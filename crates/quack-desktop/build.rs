//! Tauri's build step. The frontend's built assets (`ui/dist`) are not
//! committed; when they are absent (a `cargo check`, CI without Node) a
//! placeholder page stands in so the embedded context still builds.

use std::fs;
use std::path::Path;

fn main() {
    let dist = Path::new("ui/dist");
    if !dist.join("index.html").exists()
        && let Err(e) = fs::create_dir_all(dist).and_then(|()| {
            fs::write(
                dist.join("index.html"),
                "<!doctype html><title>quack</title><p>The frontend was not built: run <code>make ui-build</code>.</p>",
            )
        })
    {
        println!("cargo:warning=could not write the placeholder frontend: {e}");
    }
    tauri_build::build();
}
