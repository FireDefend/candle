fn main() {
    // 指定静态库的路径
    println!("cargo:rustc-link-search=native=/Users/xigsun/Documents/repo/candle/iosbridge/");
    println!("cargo:rustc-link-search=framework=/Users/xigsun/Documents/repo/candle/iosbridge/");
    println!("cargo:rustc-flags=-l framework=SDL2");
}