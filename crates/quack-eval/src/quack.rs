//! The built `quack` binary, run as a person runs it: one process per
//! command, its own data directory, and the JSON its `-f json` flags print.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

/// How one command ended.
#[derive(Debug)]
pub(crate) struct Run {
    pub(crate) status: Option<i32>,
    pub(crate) stdout: String,
    pub(crate) stderr: String,
    pub(crate) seconds: f64,
    pub(crate) timed_out: bool,
}

impl Run {
    pub(crate) fn ok(&self) -> bool {
        self.status == Some(0)
    }

    /// The last line of stderr, which is where quack puts what went wrong.
    pub(crate) fn failure(&self) -> String {
        if self.timed_out {
            return format!("timed out after {:.0}s", self.seconds);
        }
        self.stderr
            .lines()
            .rev()
            .find(|line| !line.trim().is_empty())
            .unwrap_or("exited without a message")
            .trim()
            .to_owned()
    }
}

/// A chunk `quack search -f json` returns.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Hit {
    pub(crate) content: String,
    pub(crate) filename: String,
}

#[derive(Debug, Deserialize)]
struct Hits {
    chunks: Vec<Hit>,
}

/// One `run_sql` or `create_chart` step of an answer, with the rows it kept.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Step {
    pub(crate) tool: String,
    #[serde(default)]
    pub(crate) result: Option<StepResult>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct StepResult {
    pub(crate) rows: Vec<Vec<Value>>,
}

/// What `quack -p ... -f json` prints: the one response object every
/// interface returns.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Response {
    pub(crate) answer: String,
    #[serde(default)]
    pub(crate) steps: Vec<Step>,
    #[serde(default)]
    pub(crate) citations: Vec<Value>,
    #[serde(default)]
    pub(crate) write_refused: bool,
    pub(crate) session_id: String,
}

/// Which search legs to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SearchMode {
    Hybrid,
    Keyword,
    Vector,
}

impl SearchMode {
    fn flag(self) -> Option<&'static str> {
        match self {
            Self::Hybrid => None,
            Self::Keyword => Some("--keyword"),
            Self::Vector => Some("--vector"),
        }
    }
}

/// The binary and the data directory every command of this run shares.
pub(crate) struct Quack {
    bin: PathBuf,
    data_dir: PathBuf,
    /// How long one command may take; a turn past it is killed.
    pub(crate) timeout: Duration,
}

impl Quack {
    pub(crate) fn new(bin: PathBuf, data_dir: &Path, timeout: Duration) -> Result<Self> {
        if !bin.is_file() {
            bail!(
                "no quack binary at {}; build it first (cargo build --bin quack) or pass --quack",
                bin.display()
            );
        }
        Ok(Self {
            bin,
            data_dir: data_dir.to_path_buf(),
            timeout,
        })
    }

    /// Run `quack` with `args`, killing it past the timeout. Never fails
    /// on the command's own exit code: the `Run` carries it.
    pub(crate) fn run(&self, args: &[&str]) -> Result<Run> {
        let started = Instant::now();
        let mut child = Command::new(&self.bin)
            .args(args)
            .env("QUACK_DATA_DIR", &self.data_dir)
            .env("QUACK_CONFIG_DIR", &self.data_dir)
            .env_remove("RUST_LOG")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("starting {}", self.bin.display()))?;
        let stdout = child.stdout.take().map(read_all);
        let stderr = child.stderr.take().map(read_all);
        let mut timed_out = false;
        let status = loop {
            if let Some(status) = child.try_wait()? {
                break Some(status);
            }
            if started.elapsed() > self.timeout {
                if let Err(e) = child.kill() {
                    progress(&format!("could not stop quack after the timeout: {e}"));
                }
                let _reaped = child.wait();
                timed_out = true;
                break None;
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        let join = |reader: Option<std::thread::JoinHandle<String>>| {
            reader.map_or_else(String::new, |handle| handle.join().unwrap_or_default())
        };
        Ok(Run {
            status: status.and_then(|s| s.code()),
            stdout: join(stdout),
            stderr: join(stderr),
            seconds: started.elapsed().as_secs_f64(),
            timed_out,
        })
    }

    /// Run a command that must succeed and print JSON.
    pub(crate) fn json<T: DeserializeOwned>(&self, args: &[&str]) -> Result<T> {
        let run = self.run(args)?;
        if !run.ok() {
            bail!("quack {}: {}", args.join(" "), run.failure());
        }
        serde_json::from_str(&run.stdout)
            .with_context(|| format!("quack {} printed no JSON: {}", args.join(" "), run.stdout))
    }

    pub(crate) fn create_workspace(&self, name: &str) -> Result<()> {
        let run = self.run(&["workspace", "create", name])?;
        if !run.ok() {
            bail!("creating workspace {name}: {}", run.failure());
        }
        Ok(())
    }

    /// `quack ingest FILE -w WORKSPACE`.
    pub(crate) fn ingest(&self, path: &Path, workspace: &str) -> Result<Run> {
        let path = path.to_str().context("the fixture path is not UTF-8")?;
        self.run(&["ingest", path, "-w", workspace])
    }

    /// `quack -q SQL -w WORKSPACE -f json`: each row as an object, in
    /// column order lost, so callers compare cells, not positions.
    pub(crate) fn query(&self, sql: &str, workspace: &str) -> Result<Vec<Vec<Value>>> {
        let rows: Vec<serde_json::Map<String, Value>> =
            self.json(&["-q", sql, "-w", workspace, "-f", "json"])?;
        Ok(rows
            .into_iter()
            .map(|row| row.into_values().collect())
            .collect())
    }

    /// `quack search QUERY -w WORKSPACE -f json [--keyword|--vector] [--in FILE]`.
    pub(crate) fn search(
        &self,
        query: &str,
        workspace: &str,
        mode: SearchMode,
        within: Option<&str>,
    ) -> Result<Vec<Hit>> {
        let mut args = vec!["search", query, "-w", workspace, "-f", "json"];
        if let Some(flag) = mode.flag() {
            args.push(flag);
        }
        if let Some(file) = within {
            args.extend(["--in", file]);
        }
        let hits: Hits = self.json(&args)?;
        Ok(hits.chunks)
    }

    /// `quack -p QUESTION -w WORKSPACE -f json [-r SESSION]`. A write the
    /// turn refused exits 3 with the response printed, so the exit code
    /// alone does not decide: the JSON does, when there is any.
    pub(crate) fn ask(
        &self,
        question: &str,
        workspace: &str,
        session: Option<&str>,
    ) -> Result<Response> {
        let mut args = vec!["-p", question, "-w", workspace, "-f", "json"];
        if let Some(session) = session {
            args.extend(["-r", session]);
        }
        let run = self.run(&args)?;
        if run.timed_out {
            bail!("{}", run.failure());
        }
        match serde_json::from_str::<Response>(&run.stdout) {
            Ok(answer) => Ok(answer),
            Err(_) if !run.ok() => bail!("exit {}: {}", run.status.unwrap_or(-1), run.failure()),
            Err(e) => bail!("the answer did not parse: {e}: {}", run.stdout),
        }
    }
}

fn read_all(mut source: impl Read + Send + 'static) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let mut text = String::new();
        let _read = source.read_to_string(&mut text);
        text
    })
}

/// A line to the person running the eval, on stderr, where it stays out
/// of a report piped to a file.
pub(crate) fn progress(message: &str) {
    use std::io::Write as _;
    if writeln!(std::io::stderr(), "{message}").is_err() {
        // Nothing more can be done if stderr itself is gone.
    }
}
