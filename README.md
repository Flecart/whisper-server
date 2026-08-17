# whisper-server

`whisper-server` is a small GPU-backed, pseudo-streaming speech-to-text server.
It exposes the subset of Deepgram's live WebSocket protocol needed by VoxKey,
while running one shared faster-whisper model. It has no batch endpoint, browser
UI, diarization, translation, or multi-backend layer.

This project is not affiliated with or endorsed by Deepgram. Compatibility here
means the documented subset below, not every Deepgram feature.

## Quick start (CUDA 12)

The default `turbo`/CUDA/`int8_float16` configuration is intended for a 4 GB
RTX 3050. Allow roughly 3 GB for model files and cache plus at least 8 GB free
system disk during the first download.

```bash
uv sync --extra cuda --frozen
./scripts/run-with-cuda-libs whisper-server
curl http://127.0.0.1:8765/ready
```

The model is loaded once during startup. Run exactly one Uvicorn process; the
CLI deliberately does not expose a worker-count option.

For CPU-only development and mocked tests:

```bash
uv sync --group dev --frozen
uv run pytest
```

## API

- `GET /health` reports process liveness without performing inference.
- `GET /ready` reports model/device/compute type, live stream count, capacity,
  inference concurrency, and queue depth.
- `WS /v1/listen` accepts little-endian signed `linear16`, 16 kHz, mono binary
  frames and JSON `KeepAlive`, `Finalize`, and `CloseStream` controls.

The socket emits Deepgram-shaped `Results`, `SpeechStarted`, `UtteranceEnd`,
`Metadata`, and `Error` objects. Results include word timestamps/confidence,
`is_final`, `speech_final`, and `from_finalize`. `Finalize` always produces a
`from_finalize=true` result, even if it has an empty transcript.

Supported query parameters:

| Parameter | Behavior |
| --- | --- |
| `encoding`, `sample_rate`, `channels` | Enforced as `linear16`, `16000`, `1` |
| `language` | BCP-47 tags such as `it-IT` normalize to Whisper's `it` |
| `interim_results`, `vad_events` | Enable corresponding live messages |
| `endpointing` | Silence in milliseconds before the quality final pass |
| `utterance_end_ms` | Delay after the last final word before `UtteranceEnd` |
| `model` | Accepted, including `nova-3`; startup model always wins |
| `punctuate`, `smart_format` | Accepted; both map to Whisper's normal text output |
| `keyterm` | Accepted and reserved; committed text is currently the only prompt context |
| `mip_opt_out` | Accepted; local inference never sends audio to a model provider |

Unknown parameters and incompatible audio-shape values are rejected.

## Streaming architecture

Each connection retains a bounded PCM window and repeats faster-whisper
inference about every 1.25 seconds. Partial passes use beam size 1; endpoint and
explicit final passes use beam size 5. LocalAgreement-2 commits the exact common
word prefix of two successive hypotheses. A 1.5-second overlap remains around
committed segment boundaries, while each inference sees at most 15 seconds.
Committed text is retained only as a bounded prompt tail. Timestamps at or
before the commit boundary and overlap phrase repetitions are suppressed.

One global FIFO scheduler serializes model access by default. A stream has at
most one queued/running partial, newer audio is coalesced, and queued final jobs
supersede partial jobs and take priority. Two live sockets are allowed by
default; further connections receive an error and close code 1013. Increasing
`INFERENCE_CONCURRENCY` can allocate extra CUDA working memory and should be
benchmarked on the target card before production use.

Silero VAD bundled with faster-whisper detects speech starts and endpoint
silence. Whisper itself is not token-streaming, so interim text can revise until
LocalAgreement commits it.

## Configuration

All options are environment variables:

| Variable | Default |
| --- | --- |
| `HOST`, `PORT` | `127.0.0.1`, `8765` |
| `WHISPER_MODEL` | `turbo` |
| `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE` | `cuda`, `int8_float16` |
| `MAX_STREAMS`, `INFERENCE_CONCURRENCY` | `2`, `1` |
| `PARTIAL_INTERVAL_SECONDS` | `1.25` |
| `ACTIVE_WINDOW_SECONDS`, `OVERLAP_SECONDS` | `15`, `1.5` |
| `MAX_BUFFER_SECONDS`, `CONTEXT_CHARACTERS` | `20`, `500` |
| `MAX_SESSION_SECONDS`, `IDLE_TIMEOUT_SECONDS` | `300`, `10` |
| `MAX_FRAME_BYTES` | `1048576` |
| `WHISPER_API_TOKEN` | unset |
| `UNSAFE_ALLOW_UNAUTHENTICATED` | `false` |

Binding beyond loopback requires `WHISPER_API_TOKEN`; clients send
`Authorization: Token …`. The explicit unsafe flag is intended only for a
container published exclusively on host loopback or disposable development.
Never expose an unauthenticated listener to a LAN or the internet. TLS is not
provided: use a private SSH tunnel or a TLS reverse proxy.

Sessions are capped at five minutes, binary frames at 1 MiB, PCM buffers at 20
seconds, and idle clients at about ten seconds without audio or `KeepAlive`.

## Docker

Install the NVIDIA Container Toolkit, then:

```bash
docker compose up --build -d
curl http://127.0.0.1:8766/ready
```

The example stages on host port 8766, persists the Hugging Face cache, requests
one GPU, and publishes only on loopback. Set `HOST_PORT=8765` only after the
staging acceptance tests pass.

## Bare metal and systemd

Copy the checkout to `/opt/whisper-server`, run `uv sync --extra cuda --frozen`,
install `deploy/systemd/whisper-server.service`, and adjust its user/path. The
provided unit stages on port 8766. `scripts/run-with-cuda-libs` adds only the
CUDA wheel library directories to `LD_LIBRARY_PATH` before starting the CLI.

`deploy/systemd/whisper-server-user.service` is a staging example for an
existing per-user faster-whisper virtual environment. It keeps the current
service and port 8765 intact while serving this repository from port 8766.

For VoxKey, make a temporary forward first:

```bash
ssh -NT -L 127.0.0.1:8766:127.0.0.1:8766 claw
VOXKEY_BACKEND=whisper \
VOXKEY_WHISPER_URL=ws://127.0.0.1:8766/v1/listen voxkey daemon
```

The included fixture client sends raw PCM in real time and reports interim and
finalization latency:

```bash
./scripts/e2e-client.py fixture.pcm --language it-IT
```

For bounded-window soak testing, `--speed 4` streams ten minutes of PCM in
about 2.5 minutes while `--event-limit 5` keeps diagnostic output compact.

After English, Italian, ten-minute, two-stream, latency, and GPU-memory checks
pass, move the existing tunnel/service from 8765 to the accepted deployment.

## Compatibility and limits

English and Italian are acceptance targets. Other languages supported by the
selected multilingual Whisper model are allowed but unverified. Audio must be
raw PCM; containers/codecs, multichannel audio, alternative sample rates,
Deepgram intelligence features, prerecorded REST transcription, diarization,
and translation are intentionally unsupported in v0.1.

Troubleshooting:

- A startup CUDA library error usually means the launcher was skipped or the
  NVIDIA driver/container runtime is unavailable.
- HTTP 503/connection refusal during startup means the model is still loading;
  watch the service log and retry `/ready`.
- Close 1013 means `MAX_STREAMS` is full; inspect `/ready` before increasing it.
- Repeated OOMs require a smaller model/compute type, one inference worker, or
  fewer streams. Do not assume increasing concurrency is free.
- A close after ten seconds indicates missing audio and `KeepAlive` traffic.

The LocalAgreement design is based on the MIT-licensed
[Whisper-Streaming](https://github.com/ufal/whisper_streaming) approach. See
[`NOTICE`](NOTICE) for attribution.
