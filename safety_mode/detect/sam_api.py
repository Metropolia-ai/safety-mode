"""SAM 3.1 through the Meta Model API (Responses API), with a disk cache.

Cache: one JSON file per (sha256 of the source bytes, prompt, model, options)
under ``test-footage/cache/sam31/`` (gitignored). A cache hit costs nothing;
every miss is one billed request, so ``len(list(cache_dir.glob('*.json')))``
is the number of paid calls made from this machine.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .sam_parse import Detection, parse_output

BASE_URL = os.environ.get("SM_SAM_BASE_URL", "https://api.meta.ai/v1")
MODEL = os.environ.get("SM_SAM_MODEL", "sam-3.1")
REPO = Path(__file__).resolve().parents[2]
CACHE_DIR = Path(os.environ.get("SM_SAM_CACHE", REPO / "test-footage" / "cache" / "sam31"))
MAX_SIDE = 1280


@dataclass
class SamResult:
    text: str
    prompt: str
    source_size: tuple[int, int]   # (w, h) of the original image or video
    sent_size: tuple[int, int]     # (w, h) of what SAM saw
    elapsed_s: float
    cached: bool
    meta: dict

    @property
    def scale(self) -> tuple[float, float]:
        return (self.source_size[0] / self.sent_size[0], self.source_size[1] / self.sent_size[1])

    def detections(self, decode_masks: bool = True) -> list[Detection]:
        return parse_output(self.text, self.prompt, scale=self.scale, decode_masks=decode_masks)


def _key(data: bytes, prompt: str, model: str, extra: dict) -> str:
    h = hashlib.sha256(data).hexdigest()
    tail = json.dumps({"p": prompt, "m": model, **extra}, sort_keys=True, ensure_ascii=False)
    return h[:24] + "_" + hashlib.sha256(tail.encode()).hexdigest()[:12]


def prepare_image(data: bytes, max_side: int = MAX_SIDE) -> tuple[bytes, tuple[int, int], tuple[int, int]]:
    """Return (jpeg bytes to send, source size, sent size). Resizes to max_side on the long side."""
    from PIL import Image
    im = Image.open(io.BytesIO(data))
    src = im.size
    if max(src) <= max_side and im.format == "JPEG":
        return data, src, src
    im = im.convert("RGB")
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue(), src, im.size


class SamClient:
    def __init__(self, api_key: str | None = None, base_url: str = BASE_URL, model: str = MODEL,
                 cache_dir: Path | str = CACHE_DIR, mask_encoding: str | None = None,
                 include_confidence: bool = False, offline: bool = False, timeout: float = 300):
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.mask_encoding = mask_encoding
        self.include_confidence = include_confidence
        self.offline = offline
        self._api_key = api_key or os.environ.get("MODEL_API_KEY")
        self._base_url = base_url
        self._timeout = timeout
        self._client = None
        self.calls = 0          # paid requests made by this instance
        self._lock = threading.Lock()

    @property
    def client(self):
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("MODEL_API_KEY is not set")
            from openai import OpenAI
            self._client = OpenAI(base_url=self._base_url, api_key=self._api_key, timeout=self._timeout)
        return self._client

    def _metadata(self) -> dict | None:
        md = {}
        if self.mask_encoding:
            md["mask_encoding"] = self.mask_encoding
        if self.include_confidence:
            md["include_confidence"] = "true"
        return md or None

    def _cached(self, key: str) -> dict | None:
        p = self.cache_dir / f"{key}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def _store(self, key: str, rec: dict) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / f"{key}.json").write_text(json.dumps(rec, ensure_ascii=False))

    def _create(self, content: list[dict]) -> str:
        kwargs = {"model": self.model, "input": [{"role": "user", "content": content}]}
        md = self._metadata()
        if md:
            kwargs["metadata"] = md
        with self._lock:        # the batch and pipeline call from several threads
            self.calls += 1
        r = self.client.responses.create(**kwargs)
        return r.output_text or ""

    def segment_image(self, image: bytes | str | Path, prompt: str, max_side: int = MAX_SIDE) -> SamResult:
        data = Path(image).read_bytes() if isinstance(image, (str, Path)) else image
        extra = {"kind": "image", "max_side": max_side, "md": self._metadata()}
        key = _key(data, prompt, self.model, extra)
        rec = self._cached(key)
        if rec is None:
            if self.offline:
                raise RuntimeError(f"cache miss in offline mode: {prompt!r}")
            sent, src, size = prepare_image(data, max_side)
            url = "data:image/jpeg;base64," + base64.b64encode(sent).decode()
            t0 = time.time()
            text = self._create([{"type": "input_text", "text": prompt},
                                 {"type": "input_image", "image_url": url}])
            rec = {"model": self.model, "prompt": prompt, "kind": "image", "source_size": src,
                   "sent_size": size, "elapsed_s": round(time.time() - t0, 3), "created": time.time(),
                   "options": extra, "source": str(image) if not isinstance(image, bytes) else None,
                   "output_text": text}
            self._store(key, rec)
            cached = False
        else:
            cached = True
        return SamResult(rec["output_text"], prompt, tuple(rec["source_size"]), tuple(rec["sent_size"]),
                         rec["elapsed_s"], cached, rec)

    def segment_video(self, video: str | Path, prompt: str, source_size: tuple[int, int] | None = None) -> SamResult:
        """Upload an MP4 (POST /v1/files, purpose user_data) and segment it.

        ``source_size`` is the size of the original clip if ``video`` is a downscaled copy.
        """
        path = Path(video)
        data = path.read_bytes()
        extra = {"kind": "video", "md": self._metadata()}
        key = _key(data, prompt, self.model, extra)
        rec = self._cached(key)
        if rec is None:
            if self.offline:
                raise RuntimeError(f"cache miss in offline mode: {prompt!r}")
            t0 = time.time()
            with path.open("rb") as f:
                up = self.client.files.create(file=f, purpose="user_data")
            t_up = time.time() - t0
            text = self._create([{"type": "input_text", "text": prompt},
                                 {"type": "input_video", "file_id": up.id}])
            sent = _video_size(path)
            rec = {"model": self.model, "prompt": prompt, "kind": "video", "file_id": up.id,
                   "source_size": list(source_size or sent), "sent_size": list(sent),
                   "upload_s": round(t_up, 3), "elapsed_s": round(time.time() - t0, 3),
                   "created": time.time(), "options": extra, "source": str(path), "output_text": text}
            self._store(key, rec)
            cached = False
        else:
            cached = True
        return SamResult(rec["output_text"], prompt, tuple(rec["source_size"]), tuple(rec["sent_size"]),
                         rec["elapsed_s"], cached, rec)


def _video_size(path: Path) -> tuple[int, int]:
    import subprocess
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    w, h = out.split(",")[:2]
    return int(w), int(h)
