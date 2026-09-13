use serde_json::json;

use crate::error::{Error, Result};
use crate::storage::workspace::QueryResults;

/// Generate an `ECharts` option spec from query results.
///
/// # Errors
///
/// Returns an error if the specified columns are not found in the results.
pub fn generate_chart_spec(
    results: &QueryResults,
    chart_type: &str,
    x_column: &str,
    y_column: &str,
    title: &str,
) -> Result<serde_json::Value> {
    let x_idx = results
        .columns
        .iter()
        .position(|c| c == x_column)
        .ok_or_else(|| {
            Error::Analysis(format!("column '{x_column}' not found in query results"))
        })?;

    let y_idx = results
        .columns
        .iter()
        .position(|c| c == y_column)
        .ok_or_else(|| {
            Error::Analysis(format!("column '{y_column}' not found in query results"))
        })?;

    let x_data: Vec<serde_json::Value> = results
        .rows
        .iter()
        .filter_map(|row| row.get(x_idx).cloned())
        .collect();

    let y_data: Vec<serde_json::Value> = results
        .rows
        .iter()
        .filter_map(|row| row.get(y_idx).cloned())
        .collect();

    match chart_type {
        "pie" => Ok(build_pie_spec(title, x_column, &x_data, &y_data)),
        "bar" | "line" | "scatter" | "area" => {
            let series_type = if chart_type == "area" {
                "line"
            } else {
                chart_type
            };
            Ok(build_cartesian_spec(
                title,
                series_type,
                chart_type == "area",
                x_column,
                y_column,
                &x_data,
                &y_data,
            ))
        }
        other => Err(Error::Analysis(format!("unsupported chart type: {other}"))),
    }
}

fn build_cartesian_spec(
    title: &str,
    series_type: &str,
    area_style: bool,
    x_label: &str,
    y_label: &str,
    x_data: &[serde_json::Value],
    y_data: &[serde_json::Value],
) -> serde_json::Value {
    let mut series = json!({
        "type": series_type,
        "data": y_data,
        "name": y_label,
    });

    if area_style && let Some(obj) = series.as_object_mut() {
        obj.insert(String::from("areaStyle"), json!({}));
    }

    json!({
        "title": { "text": title },
        "tooltip": { "trigger": "axis" },
        "xAxis": {
            "type": "category",
            "data": x_data,
            "name": x_label,
        },
        "yAxis": {
            "type": "value",
            "name": y_label,
        },
        "series": [series],
    })
}

fn build_pie_spec(
    title: &str,
    category_label: &str,
    categories: &[serde_json::Value],
    values: &[serde_json::Value],
) -> serde_json::Value {
    let data: Vec<serde_json::Value> = categories
        .iter()
        .zip(values.iter())
        .map(|(name, value)| {
            json!({
                "name": name,
                "value": value,
            })
        })
        .collect();

    json!({
        "title": { "text": title },
        "tooltip": { "trigger": "item" },
        "series": [{
            "type": "pie",
            "data": data,
            "name": category_label,
        }],
    })
}

#[cfg(test)]
#[expect(clippy::indexing_slicing, reason = "JSON path access in tests")]
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
                    serde_json::Value::Number(200.into()),
                ],
                vec![
                    serde_json::Value::String("East".into()),
                    serde_json::Value::Number(150.into()),
                ],
            ],
        }
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn generates_bar_chart() {
        let spec = generate_chart_spec(
            &sample_results(),
            "bar",
            "region",
            "sales",
            "Sales by Region",
        )
        .unwrap();
        assert_eq!(spec["title"]["text"], "Sales by Region");
        assert_eq!(spec["series"][0]["type"], "bar");
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn generates_line_chart() {
        let spec = generate_chart_spec(&sample_results(), "line", "region", "sales", "Sales Trend")
            .unwrap();
        assert_eq!(spec["series"][0]["type"], "line");
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn generates_area_chart() {
        let spec = generate_chart_spec(&sample_results(), "area", "region", "sales", "Area Chart")
            .unwrap();
        assert_eq!(spec["series"][0]["type"], "line");
        assert!(spec["series"][0]["areaStyle"].is_object());
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn generates_pie_chart() {
        let spec = generate_chart_spec(&sample_results(), "pie", "region", "sales", "Market Share")
            .unwrap();
        assert_eq!(spec["series"][0]["type"], "pie");
        assert!(spec["series"][0]["data"].is_array());
    }

    #[test]
    fn missing_column_returns_error() {
        let result = generate_chart_spec(&sample_results(), "bar", "missing", "sales", "Title");
        assert!(result.is_err());
    }

    #[test]
    fn unsupported_chart_type_returns_error() {
        let result = generate_chart_spec(&sample_results(), "radar", "region", "sales", "Title");
        assert!(result.is_err());
    }
}
