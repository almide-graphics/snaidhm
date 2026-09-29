# A headless Wayland desktop to run snaidhm's Wayland client against: sway on
# its headless backend with the pixman renderer (no GPU), grim to capture it,
# wtype for a virtual keyboard, the fonts native/fonts.almd looks for, and the
# Rust toolchain a native Almide build needs.
FROM debian:trixie
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    sway grim wtype ca-certificates curl git build-essential pkg-config \
    fonts-noto-cjk fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.94.0
ENV PATH=/root/.cargo/bin:$PATH
ENV XDG_RUNTIME_DIR=/tmp/xdg
