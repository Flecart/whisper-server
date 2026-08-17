#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from urllib.parse import urlencode

from websockets.asyncio.client import connect


async def transcribe(
    endpoint: str, pcm_path: Path, language: str, token: str | None
) -> dict[str, object]:
    parameters = urlencode(
        {
            "model": "nova-3",
            "language": language,
            "encoding": "linear16",
            "sample_rate": "16000",
            "channels": "1",
            "interim_results": "true",
            "vad_events": "true",
            "endpointing": "300",
            "utterance_end_ms": "1000",
            "punctuate": "true",
            "smart_format": "true",
            "mip_opt_out": "true",
        }
    )
    headers = {"Authorization": f"Token {token}"} if token else None
    connected_at = time.monotonic()
    final_segments: list[str] = []
    interim = ""
    first_interim_seconds: float | None = None
    finalized_seconds: float | None = None
    speech_started = False

    async with connect(
        f"{endpoint}?{parameters}",
        additional_headers=headers,
        max_size=2**20,
    ) as websocket:
        messages: asyncio.Queue[tuple[float, dict[str, object]]] = asyncio.Queue()

        async def receive() -> None:
            async for payload in websocket:
                if isinstance(payload, str):
                    messages.put_nowait((time.monotonic(), json.loads(payload)))

        receiver = asyncio.create_task(receive())
        pcm = await asyncio.to_thread(pcm_path.read_bytes)
        for offset in range(0, len(pcm), 4096):
            chunk = pcm[offset : offset + 4096]
            await websocket.send(chunk)
            await asyncio.sleep(len(chunk) / 32_000)
        finalize_at = time.monotonic()
        await websocket.send(json.dumps({"type": "Finalize"}))

        while finalized_seconds is None:
            received_at, message = await asyncio.wait_for(messages.get(), timeout=15)
            message_type = message.get("type")
            if message_type == "Error":
                raise RuntimeError(str(message))
            if message_type == "SpeechStarted":
                speech_started = True
            if message_type != "Results":
                continue
            alternatives = message.get("channel", {}).get("alternatives", [])
            text = alternatives[0].get("transcript", "").strip() if alternatives else ""
            if message.get("is_final"):
                if text:
                    final_segments.append(text)
                interim = ""
            elif text:
                interim = text
                if first_interim_seconds is None:
                    first_interim_seconds = received_at - connected_at
            if message.get("from_finalize"):
                finalized_seconds = received_at - finalize_at

        await websocket.send(json.dumps({"type": "CloseStream"}))
        metadata = None
        while metadata is None:
            _, message = await asyncio.wait_for(messages.get(), timeout=5)
            if message.get("type") == "Metadata":
                metadata = message
        await receiver

    text = " ".join((*final_segments, interim)).strip()
    return {
        "language": language,
        "audio_seconds": round(len(pcm) / 32_000, 3),
        "transcript": text,
        "speech_started": speech_started,
        "first_interim_seconds": (
            round(first_interim_seconds, 3)
            if first_interim_seconds is not None
            else None
        ),
        "finalize_seconds": round(finalized_seconds, 3),
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pcm", type=Path)
    parser.add_argument("--language", required=True)
    parser.add_argument("--endpoint", default="ws://127.0.0.1:8766/v1/listen")
    parser.add_argument("--token")
    args = parser.parse_args()
    result = asyncio.run(transcribe(args.endpoint, args.pcm, args.language, args.token))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
