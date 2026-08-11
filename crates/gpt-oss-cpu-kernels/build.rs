use std::env;

fn main() {
    println!("cargo:rerun-if-changed=native/amx_int8.cpp");
    if env::var_os("CARGO_FEATURE_AMX_INT8").is_none()
        || env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux")
        || env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("x86_64")
    {
        return;
    }

    cc::Build::new()
        .cpp(true)
        .file("native/amx_int8.cpp")
        .std("c++17")
        .flag("-mamx-tile")
        .flag("-mamx-int8")
        .warnings(true)
        .compile("gpt_oss_amx_int8");
}
