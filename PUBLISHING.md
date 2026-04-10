# Publishing

This repository is structured to be published as a standalone project.

## Suggested Repository Name

- `cc-proxy`

## Suggested Description

- `Claude-compatible proxy for OpenAI-compatible Chat Completions and Responses upstreams`

## Suggested Topics

- `claude`
- `anthropic`
- `openai`
- `proxy`
- `rust`
- `docker`
- `sse`

## Before Pushing

- Review `README.md`
- Review `LICENSE`
- Confirm no real credentials are committed
- Confirm `config/*.local.yaml` remains ignored
- Confirm local-only test containers are not relevant to docs
- Confirm `docker build` works for your target architecture

## First Push Checklist

1. Create an empty remote repository
2. Add the remote URL
3. Push `master`
4. Set repository description and topics
5. Optionally enable Issues and Discussions

## Commands

```bash
git remote add origin <your-remote-url>
git push -u origin master
```

## Notes

- `run.sh` is the quickest local start path
- `scripts/build.sh` builds local `arm64` and `amd64` images
- `scripts/test.sh` runs a container smoke test against `/health`
- Example configs are safe to commit
- Local configs should stay in `config/*.local.yaml`
