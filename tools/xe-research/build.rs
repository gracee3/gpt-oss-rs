use std::path::PathBuf;

fn main() {
    let corpus = std::env::var_os("XE_RESEARCH_CORPUS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/home/emmy/src/xe-research"));
    let opencl = corpus.join("OpenCL-Headers");
    let level_zero = corpus.join("toolchain/level-zero-1.28.2/usr/include");

    for required in [&opencl, &level_zero] {
        if !required.is_dir() {
            panic!(
                "required cached Xe header directory is missing: {}; set XE_RESEARCH_CORPUS",
                required.display()
            );
        }
    }

    cc::Build::new()
        .file("native/xe_probe.c")
        .include(opencl)
        .include(level_zero)
        .define("CL_TARGET_OPENCL_VERSION", "300")
        .flag_if_supported("-std=gnu11")
        .flag_if_supported("-Wall")
        .flag_if_supported("-Wextra")
        .compile("xe_probe");

    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rerun-if-changed=native/xe_probe.c");
    println!("cargo:rerun-if-changed=native/xe_probe.h");
    println!("cargo:rerun-if-env-changed=XE_RESEARCH_CORPUS");
}
