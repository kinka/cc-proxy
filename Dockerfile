FROM rust:1.94-bookworm AS builder

ARG CARGO_HTTP_TIMEOUT=600
ARG CARGO_NET_RETRY=10
ARG CARGO_REGISTRIES_CRATES_IO_INDEX=

ENV CARGO_HTTP_TIMEOUT=${CARGO_HTTP_TIMEOUT} \
    CARGO_NET_RETRY=${CARGO_NET_RETRY}

WORKDIR /app
COPY Cargo.toml ./Cargo.toml
COPY .cargo ./.cargo
COPY src ./src

RUN if [ -n "${CARGO_REGISTRIES_CRATES_IO_INDEX}" ]; then \
        mkdir -p /usr/local/cargo && \
        printf '[registries.crates-io]\nindex = "%s"\n' "${CARGO_REGISTRIES_CRATES_IO_INDEX}" > /usr/local/cargo/config.toml; \
    fi

RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/target/release/cc-proxy /usr/local/bin/cc-proxy
COPY config/proxy.example.yaml /app/config/proxy.example.yaml

EXPOSE 25721
CMD ["cc-proxy", "--config", "/app/config/proxy.yaml"]
