
fn main() {

    #[cfg(target_os = "ios")]
    println!("cargo:rustc-link-search=native=/Users/xigsun/Documents/repo/candle/iosbridge/");
    #[cfg(target_os = "ios")]
    println!("cargo:rustc-link-search=framework=/Users/xigsun/Documents/repo/candle/iosbridge/");
    #[cfg(target_os = "ios")]
    println!("cargo:rustc-flags=-l framework=SDL2");

}
