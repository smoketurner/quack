//! quack's chart spec: small enough for the model to fill and for every
//! interface to render (ratatui in the terminal, `ECharts` in a browser later).

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::storage::workspace::{Cell, QueryResults};

/// Most points per series (distinct x values); beyond this the query
/// should aggregate.
pub const MAX_POINTS: usize = 200;
/// Most series on one chart: the terminal's colour cycle, and what a
/// legend still reads.
pub const MAX_SERIES: usize = 8;

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    schemars::JsonSchema,
    utoipa::ToSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum ChartKind {
    Bar,
    Line,
    Scatter,
    Pie,
}

text_enum!(ChartKind, "chart kind", {
    Bar => "bar",
    Line => "line",
    Scatter => "scatter",
    Pie => "pie",
});

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, utoipa::ToSchema)]
pub struct Axis {
    pub label: String,
    /// Category labels, one per point (or per pie slice).
    pub values: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, utoipa::ToSchema)]
pub struct Series {
    pub name: String,
    /// One value per x label; a label the series has no row for is 0.
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, utoipa::ToSchema)]
pub struct ChartSpec {
    pub title: String,
    pub kind: ChartKind,
    pub x: Axis,
    pub series: Vec<Series>,
    /// Bars and lines stacked on each other instead of beside; a stored
    /// spec from before the flag is not stacked.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub stacked: bool,
}

/// Which columns make a chart's series.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeriesColumns<'a> {
    /// One series per column, named after it.
    pub y: &'a [String],
    /// Long-format rows: one series per distinct value of this column,
    /// each taking its rows' first `y` value. Distinct x values in
    /// first-seen order become the axis.
    pub series_by: Option<&'a str>,
}

impl ChartSpec {
    /// Number of points in the longest series.
    #[must_use]
    pub fn points(&self) -> usize {
        self.series
            .iter()
            .map(|s| s.values.len())
            .max()
            .unwrap_or(0)
    }

    /// The series names, for a summary line.
    #[must_use]
    pub fn series_names(&self) -> Vec<&str> {
        self.series.iter().map(|s| s.name.as_str()).collect()
    }

    /// A `kind` chart of a result set: `x_column` supplies the labels and
    /// `columns` the series. At most [`MAX_POINTS`] distinct labels and
    /// [`MAX_SERIES`] series.
    ///
    /// # Errors
    ///
    /// Returns an error if a column is missing, a value is not numeric, no
    /// `y` column is named, or there are too many labels or series.
    pub fn from_results(
        results: &QueryResults,
        kind: ChartKind,
        x_column: &str,
        columns: SeriesColumns<'_>,
        title: &str,
    ) -> Result<Self> {
        let column = |name: &str| {
            results
                .columns
                .iter()
                .position(|c| c == name)
                .ok_or_else(|| {
                    Error::Analysis(format!("column '{name}' not found in query results"))
                })
        };
        let x_idx = column(x_column)?;
        let y_idx: Vec<usize> = columns
            .y
            .iter()
            .map(|name| column(name))
            .collect::<Result<_>>()?;
        let Some(&first_y) = y_idx.first() else {
            return Err(Error::Analysis(String::from(
                "a chart needs at least one y column",
            )));
        };
        let by_idx = columns.series_by.map(column).transpose()?;

        // Distinct labels in first-seen order, each with its position.
        let mut labels: Vec<String> = Vec::new();
        let mut positions: HashMap<String, usize> = HashMap::new();
        // Series by name, in first-seen order, each a sparse row of values.
        let mut names: Vec<String> = Vec::new();
        let mut values: HashMap<String, Vec<Option<f64>>> = HashMap::new();
        for (i, row) in results.rows.iter().enumerate() {
            let label = Cell::at(row, x_idx).label();
            let position = if let Some(&position) = positions.get(&label) {
                position
            } else {
                if labels.len() >= MAX_POINTS {
                    return Err(Error::Analysis(format!(
                        "more than {MAX_POINTS} distinct {x_column} values is too many for a chart; aggregate or limit the query"
                    )));
                }
                positions.insert(label.clone(), labels.len());
                labels.push(label);
                labels.len().saturating_sub(1)
            };
            let targets: Vec<(String, usize)> = match by_idx {
                Some(by) => vec![(Cell::at(row, by).label(), first_y)],
                None => columns
                    .y
                    .iter()
                    .cloned()
                    .zip(y_idx.iter().copied())
                    .collect(),
            };
            for (name, idx) in targets {
                if !values.contains_key(&name) {
                    if names.len() >= MAX_SERIES {
                        return Err(Error::Analysis(format!(
                            "more than {MAX_SERIES} series is too many for one chart; filter or aggregate the query"
                        )));
                    }
                    names.push(name.clone());
                    values.insert(name.clone(), Vec::new());
                }
                let y = row.get(idx).cloned().unwrap_or(serde_json::Value::Null);
                let number = Cell(&y).number().ok_or_else(|| {
                    Error::Analysis(format!(
                        "row {} of column '{}' is not numeric: {y}",
                        i.saturating_add(1),
                        results.columns.get(idx).map_or("", String::as_str)
                    ))
                })?;
                let series = values.entry(name).or_default();
                if series.len() <= position {
                    series.resize(position.saturating_add(1), None);
                }
                if let Some(slot) = series.get_mut(position) {
                    *slot = Some(number);
                }
            }
        }

        let series = names
            .into_iter()
            .map(|name| {
                let mut row = values.remove(&name).unwrap_or_default();
                row.resize(labels.len(), None);
                Series {
                    name,
                    values: row.into_iter().map(Option::unwrap_or_default).collect(),
                }
            })
            .collect();
        Ok(Self {
            title: title.to_owned(),
            kind,
            x: Axis {
                label: x_column.to_owned(),
                values: labels,
            },
            series,
            stacked: false,
        })
    }

    /// The same chart, stacked.
    #[must_use]
    pub fn stacked(mut self, stacked: bool) -> Self {
        self.stacked = stacked;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_results() -> QueryResults {
        QueryResults {
            columns: vec!["region".into(), "sales".into()],
            rows: vec![
                vec![
                    serde_json::Value::String("North".into()),
                    serde_json::Value::Number(100.into()),
                ],
                vec![
                    serde_json::Value::String("South".into()),
                    serde_json::Value::from(200.5),
                ],
                vec![serde_json::Value::Null, serde_json::Value::Null],
            ],
        }
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn builds_a_spec_with_labels_and_numeric_values() {
        let spec = ChartSpec::from_results(
            &sample_results(),
            ChartKind::Bar,
            "region",
            one("sales"),
            "Sales by Region",
        )
        .unwrap();
        assert_eq!(spec.title, "Sales by Region");
        assert_eq!(spec.kind, ChartKind::Bar);
        assert_eq!(spec.x.label, "region");
        assert_eq!(spec.x.values, vec!["North", "South", "NULL"]);
        assert_eq!(spec.series.len(), 1);
        assert_eq!(spec.series.first().unwrap().name, "sales");
        assert_eq!(spec.series.first().unwrap().values, vec![100.0, 200.5, 0.0]);
        assert_eq!(spec.points(), 3);
    }

    #[test]
    #[expect(
        clippy::unwrap_used,
        clippy::indexing_slicing,
        reason = "test asserts Ok and reads known JSON paths"
    )]
    fn spec_round_trips_through_json() {
        let spec = ChartSpec::from_results(
            &sample_results(),
            ChartKind::Pie,
            "region",
            one("sales"),
            "Share",
        )
        .unwrap();
        let json = serde_json::to_value(&spec).unwrap();
        assert_eq!(json["kind"], "pie");
        assert_eq!(json["x"]["values"][0], "North");
        assert!(
            json.get("stacked").is_none(),
            "an unstacked spec stays as it was"
        );
        let back: ChartSpec = serde_json::from_value(json).unwrap();
        assert_eq!(back, spec);
        // A stored spec from before the flag, and a stacked one.
        let old: ChartSpec = serde_json::from_str(
            r#"{"title":"t","kind":"bar","x":{"label":"x","values":["a"]},"series":[{"name":"s","values":[1.0]}]}"#,
        )
        .unwrap();
        assert!(!old.stacked);
        let stacked = spec.stacked(true);
        let json = serde_json::to_value(&stacked).unwrap();
        assert_eq!(json["stacked"], true);
        assert_eq!(serde_json::from_value::<ChartSpec>(json).unwrap(), stacked);
    }

    #[test]
    fn all_kinds_parse_and_others_do_not() {
        for (text, kind) in [
            ("bar", ChartKind::Bar),
            ("LINE", ChartKind::Line),
            (" scatter ", ChartKind::Scatter),
            ("pie", ChartKind::Pie),
        ] {
            assert!(text.parse::<ChartKind>().is_ok_and(|k| k == kind));
        }
        let err = "area".parse::<ChartKind>().err();
        assert!(err.is_some_and(|e| e.to_string().contains("bar, line, scatter, pie")));
    }

    #[test]
    fn missing_column_and_non_numeric_values_are_errors() {
        assert!(
            ChartSpec::from_results(
                &sample_results(),
                ChartKind::Bar,
                "missing",
                one("sales"),
                "t"
            )
            .is_err()
        );
        assert!(
            ChartSpec::from_results(
                &sample_results(),
                ChartKind::Bar,
                "region",
                one("missing"),
                "t"
            )
            .is_err()
        );
        assert!(
            ChartSpec::from_results(&sample_results(), ChartKind::Bar, "region", none(), "t")
                .is_err()
        );
        let text_values = QueryResults {
            columns: vec!["a".into(), "b".into()],
            rows: vec![vec![
                serde_json::Value::String("x".into()),
                serde_json::Value::String("not a number".into()),
            ]],
        };
        let err = ChartSpec::from_results(&text_values, ChartKind::Bar, "a", one("b"), "t").err();
        assert!(err.is_some_and(|e| e.to_string().contains("not numeric")));
    }

    #[test]
    fn too_many_rows_is_an_error() {
        let rows: Vec<Vec<serde_json::Value>> = (0..=MAX_POINTS)
            .map(|i| {
                vec![
                    serde_json::Value::String(i.to_string()),
                    serde_json::Value::Number(i.into()),
                ]
            })
            .collect();
        let big = QueryResults {
            columns: vec!["a".into(), "b".into()],
            rows,
        };
        let err = ChartSpec::from_results(&big, ChartKind::Line, "a", one("b"), "t").err();
        assert!(err.is_some_and(|e| e.to_string().contains("too many")));
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn several_y_columns_are_several_series() {
        let wide = QueryResults {
            columns: vec!["month".into(), "revenue".into(), "cost".into()],
            rows: vec![
                vec!["jan".into(), 10.into(), 4.into()],
                vec!["feb".into(), 12.into(), 5.into()],
            ],
        };
        let ys = [String::from("revenue"), String::from("cost")];
        let spec = ChartSpec::from_results(
            &wide,
            ChartKind::Line,
            "month",
            SeriesColumns {
                y: &ys,
                series_by: None,
            },
            "t",
        )
        .unwrap();
        assert_eq!(spec.x.values, vec!["jan", "feb"]);
        assert_eq!(spec.series_names(), vec!["revenue", "cost"]);
        assert_eq!(
            spec.series.get(1).map(|s| s.values.clone()),
            Some(vec![4.0, 5.0])
        );
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn long_rows_pivot_by_the_series_column_with_gaps_as_zero() {
        let long = QueryResults {
            columns: vec!["month".into(), "region".into(), "orders".into()],
            rows: vec![
                vec!["jan".into(), "north".into(), 3.into()],
                vec!["jan".into(), "south".into(), 5.into()],
                vec!["feb".into(), "south".into(), 7.into()],
                vec!["mar".into(), "north".into(), 1.into()],
            ],
        };
        let ys = [String::from("orders")];
        let spec = ChartSpec::from_results(
            &long,
            ChartKind::Bar,
            "month",
            SeriesColumns {
                y: &ys,
                series_by: Some("region"),
            },
            "t",
        )
        .unwrap();
        assert_eq!(spec.x.values, vec!["jan", "feb", "mar"]);
        assert_eq!(spec.series_names(), vec!["north", "south"]);
        assert_eq!(
            spec.series.first().map(|s| s.values.clone()),
            Some(vec![3.0, 0.0, 1.0])
        );
        assert_eq!(
            spec.series.get(1).map(|s| s.values.clone()),
            Some(vec![5.0, 7.0, 0.0])
        );
        assert_eq!(spec.points(), 3);
    }

    #[test]
    fn the_caps_count_distinct_labels_per_series_and_series_in_all() {
        // 400 rows but only 2 distinct labels: fine.
        let rows: Vec<Vec<serde_json::Value>> = (0..400)
            .map(|i: u32| {
                vec![
                    (i.rem_euclid(2)).to_string().into(),
                    (i.rem_euclid(3)).to_string().into(),
                    i.into(),
                ]
            })
            .collect();
        let long = QueryResults {
            columns: vec!["x".into(), "s".into(), "v".into()],
            rows,
        };
        let ys = [String::from("v")];
        let by = SeriesColumns {
            y: &ys,
            series_by: Some("s"),
        };
        let spec = ChartSpec::from_results(&long, ChartKind::Line, "x", by, "t");
        assert!(spec.is_ok_and(|s| s.series.len() == 3 && s.points() == 2));
        // Nine distinct series values is one too many.
        let rows: Vec<Vec<serde_json::Value>> = (0..9)
            .map(|i| vec!["a".into(), i.to_string().into(), i.into()])
            .collect();
        let many = QueryResults {
            columns: vec!["x".into(), "s".into(), "v".into()],
            rows,
        };
        let err = ChartSpec::from_results(&many, ChartKind::Line, "x", by, "t").err();
        assert!(err.is_some_and(|e| e.to_string().contains("series")));
    }

    fn one(name: &str) -> SeriesColumns<'static> {
        let leaked: &'static [String] = Box::leak(vec![name.to_owned()].into_boxed_slice());
        SeriesColumns {
            y: leaked,
            series_by: None,
        }
    }

    fn none() -> SeriesColumns<'static> {
        SeriesColumns {
            y: &[],
            series_by: None,
        }
    }
}
