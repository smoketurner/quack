//! What a running quack reports about itself, through the `metrics`
//! facade: provider requests, their latency, the wait for a permit, the
//! retries, the server's requests by route, the jobs by state, the
//! writer queues, and the open workspaces. Labels carry ids, kinds, and
//! names only, never workspace content. `quack serve` installs the
//! Prometheus recorder once ([`install`]) and renders it at `GET /metrics`
//! ([`render`]); without a recorder every record is a no-op, which is the
//! command line's case.

use std::sync::OnceLock;
use std::time::Duration;

use metrics::{counter, gauge, histogram};
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};

static HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();

/// Seconds, for the latency and wait histograms.
const SECONDS_BUCKETS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0,
];

/// Install the Prometheus recorder, once per process. A second call, or
/// one after another recorder was installed, keeps what is there.
pub fn install() -> Option<&'static PrometheusHandle> {
    HANDLE
        .get_or_init(|| {
            PrometheusBuilder::new()
                .set_buckets_for_metric(Matcher::Suffix(String::from("_seconds")), SECONDS_BUCKETS)
                .ok()
                .and_then(|builder| builder.install_recorder().ok())
                .unwrap_or_else(|| {
                    tracing::warn!(
                        "the metrics recorder could not be installed; /metrics stays empty"
                    );
                    PrometheusBuilder::new().build_recorder().handle()
                })
        })
        .into()
}

/// The metrics in Prometheus text form, when the recorder is installed.
#[must_use]
pub fn render() -> Option<String> {
    HANDLE.get().map(PrometheusHandle::render)
}

/// The model a request named, as a label: the id, or `-` for a request
/// that named none.
struct ModelLabel<'a>(Option<&'a str>);

impl ModelLabel<'_> {
    fn text(&self) -> String {
        self.0.unwrap_or("-").to_owned()
    }
}

/// One provider request finished with `status` (the HTTP status, or
/// `error` for a transport failure) after `latency`.
pub fn provider_request(provider: &str, model: Option<&str>, status: &str, latency: Duration) {
    let labels = [
        ("provider", provider.to_owned()),
        ("model", ModelLabel(model).text()),
        ("status", status.to_owned()),
    ];
    counter!("quack_provider_requests_total", &labels).increment(1);
    histogram!("quack_provider_request_seconds", &labels).record(latency.as_secs_f64());
}

/// A request waited `wait` for a permit of its provider's gate.
pub fn provider_permit_wait(provider: &str, model: Option<&str>, wait: Duration) {
    let labels = [
        ("provider", provider.to_owned()),
        ("model", ModelLabel(model).text()),
    ];
    histogram!("quack_provider_permit_wait_seconds", &labels).record(wait.as_secs_f64());
}

/// A request to a provider is being sent again.
pub fn provider_retry(provider: &str, model: Option<&str>) {
    let labels = [
        ("provider", provider.to_owned()),
        ("model", ModelLabel(model).text()),
    ];
    counter!("quack_provider_retries_total", &labels).increment(1);
}

/// One HTTP request to `quack serve` finished: `route` is the route's
/// template, never the path.
pub fn http_request(method: &str, route: &str, status: u16, latency: Duration) {
    let labels = [
        ("method", method.to_owned()),
        ("route", route.to_owned()),
        ("status", status.to_string()),
    ];
    counter!("quack_http_requests_total", &labels).increment(1);
    histogram!("quack_http_request_seconds", &labels).record(latency.as_secs_f64());
}

/// How many jobs of `kind` are in `state` right now.
pub fn set_jobs(kind: &str, state: &str, count: usize) {
    let labels = [("kind", kind.to_owned()), ("state", state.to_owned())];
    gauge!("quack_jobs", &labels).set(Gauged(count).value());
}

/// How many closures wait for the workspaces' writers, by priority.
pub fn set_writer_waiting(priority: &str, count: usize) {
    gauge!("quack_writer_waiting", "priority" => priority.to_owned()).set(Gauged(count).value());
}

/// How many workspace files the server holds open.
pub fn set_open_workspaces(count: usize) {
    gauge!("quack_open_workspaces").set(Gauged(count).value());
}

/// A count as a gauge's value.
struct Gauged(usize);

impl Gauged {
    #[expect(
        clippy::cast_precision_loss,
        reason = "a count of jobs or workspaces is far below 2^53"
    )]
    fn value(self) -> f64 {
        self.0 as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_recorder_renders_what_was_recorded() {
        let handle = install().unwrap_or_else(|| unreachable_handle());
        provider_request("p", Some("m"), "200", Duration::from_millis(12));
        provider_retry("p", Some("m"));
        http_request("GET", "/healthz", 200, Duration::from_millis(1));
        set_jobs("ingest", "running", 2);
        set_writer_waiting("interactive", 0);
        set_open_workspaces(3);
        let text = handle.render();
        for line in [
            r#"quack_provider_requests_total{provider="p",model="m",status="200"} 1"#,
            r#"quack_provider_retries_total{provider="p",model="m"} 1"#,
            r#"quack_http_requests_total{method="GET",route="/healthz",status="200"} 1"#,
            r#"quack_jobs{kind="ingest",state="running"} 2"#,
            "quack_open_workspaces 3",
        ] {
            assert!(text.contains(line), "{line} missing from:\n{text}");
        }
        assert!(
            text.contains("quack_provider_request_seconds_bucket"),
            "{text}"
        );
        assert_eq!(render().map(|t| t.is_empty()), Some(false));
    }

    #[expect(clippy::panic, reason = "test failure path")]
    fn unreachable_handle() -> ! {
        panic!("the recorder installs")
    }
}
