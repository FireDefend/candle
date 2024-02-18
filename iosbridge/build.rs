use std::{env, fs, io};

fn main() {
    let target = env::var("TARGET").expect("Cargo build scripts always have TARGET");
    let host = env::var("HOST").expect("Cargo build scripts always have HOST");
    let target_os = get_os_from_triple(target.as_str()).unwrap();
    // 指定静态库的路径
    if target_os.contains( "ios" ){
        println!("cargo:rustc-link-search=native=/Users/xigsun/Documents/repo/candle/iosbridge/");
        println!("cargo:rustc-link-search=framework=/Users/xigsun/Documents/repo/candle/iosbridge/");
        println!("cargo:rustc-flags=-l framework=SDL2");
    }
}

fn get_os_from_triple(triple: &str) -> Option<&str> {
    triple.splitn(3, "-").nth(2)
}