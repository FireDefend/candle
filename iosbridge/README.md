# rust candle model wrapper

## build for ios
```bash
cargo lipo --release  --targets aarch64-apple-ios --features metal

```

## build for test
```bash
cargo test test_whisper --release  --features metal -- --nocapture

```