//! quack's chart spec: small enough for the model to fill and for every
//! interface to render (ratatui in the terminal, `ECharts` in a browser later).

use crate::error::{Error, Result};
use crate::storage::workspace::QueryResults;

/// Most points per series; beyond this the query should aggregate.
pub const MAX_POINTS: usize = 200;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
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

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Axis {
    pub label: String,
    /// Category labels, one per point (or per pie slice).
    pub values: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Series {
    pub name: String,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ChartSpec {
    pub title: String,
    pub kind: ChartKind,
    pub x: Axis,
    pub series: Vec<Series>,
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
}

/// Build a spec from a result set: `x_column` supplies the labels and
/// `y_column` the values. At most [`MAX_POINTS`] rows.
///
/// # Errors
///
/// Returns an error if a column is missing, a value is not numeric, or
/// there are too many rows.
pub fn generate_chart_spec(
    results: &QueryResults,
    kind: &str,
    x_column: &str,
    y_column: &str,
    title: &str,
) -> Result<ChartSpec> {
    let kind: ChartKind = kind.parse()?;
    let column = |name: &str| {
        results
            .columns
            .iter()
            .position(|c| c == name)
            .ok_or_else(|| Error::Analysis(format!("column '{name}' not found in query results")))
    };
    let x_idx = column(x_column)?;
    let y_idx = column(y_column)?;

    if results.rows.len() > MAX_POINTS {
        return Err(Error::Analysis(format!(
            "{} rows is too many for a chart (max {MAX_POINTS}); aggregate or limit the query",
            results.rows.len()
        )));
    }

    let mut labels = Vec::with_capacity(results.rows.len());
    let mut values = Vec::with_capacity(results.rows.len());
    for (i, row) in results.rows.iter().enumerate() {
        let x = row.get(x_idx).cloned().unwrap_or(serde_json::Value::Null);
        let y = row.get(y_idx).cloned().unwrap_or(serde_json::Value::Null);
        labels.push(match x {
            serde_json::Value::String(s) => s,
            serde_json::Value::Null => String::from("NULL"),
            other => other.to_string(),
        });
        let number = match &y {
            serde_json::Value::Number(n) => n.as_f64(),
            serde_json::Value::String(s) => s.parse::<f64>().ok(),
            serde_json::Value::Null => Some(0.0),
            _ => None,
        };
        values.push(number.ok_or_else(|| {
            Error::Analysis(format!(
                "row {} of column '{y_column}' is not numeric: {y}",
                i.saturating_add(1)
            ))
        })?);
    }

    Ok(ChartSpec {
        title: title.to_owned(),
        kind,
        x: Axis {
            label: x_column.to_owned(),
            values: labels,
        },
        series: vec![Series {
            name: y_column.to_owned(),
            values,
        }],
    })
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
        let spec = generate_chart_spec(
            &sample_results(),
            "Bar",
            "region",
            "sales",
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
        let spec =
            generate_chart_spec(&sample_results(), "pie", "region", "sales", "Share").unwrap();
        let json = serde_json::to_value(&spec).unwrap();
        assert_eq!(json["kind"], "pie");
        assert_eq!(json["x"]["values"][0], "North");
        let back: ChartSpec = serde_json::from_value(json).unwrap();
        assert_eq!(back, spec);
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
        assert!(generate_chart_spec(&sample_results(), "bar", "missing", "sales", "t").is_err());
        let text_values = QueryResults {
            columns: vec!["a".into(), "b".into()],
            rows: vec![vec![
                serde_json::Value::String("x".into()),
                serde_json::Value::String("not a number".into()),
            ]],
        };
        let err = generate_chart_spec(&text_values, "bar", "a", "b", "t").err();
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
        let err = generate_chart_spec(&big, "line", "a", "b", "t").err();
        assert!(err.is_some_and(|e| e.to_string().contains("too many")));
    }
}
