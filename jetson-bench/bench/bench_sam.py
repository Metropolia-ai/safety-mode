#!/usr/bin/env python3
"""Single bench: model -> Adapter(seg, sync, encode, decode).
seg(image: PIL.Image, prompt: str) -> (masks[N,H,W] bool numpy, scores[N] numpy).
Usage:
  python bench_sam.py --model efficientsam3 --tag efficientsam3-tv-m --ckpt .../efficientsam3_tinyvit.pt \
      --backbone tinyvit --variant 11m --clip ../footage/raw/pexels-15170997-warehouse-team.mp4
Writes JSON to ../logs/<tag>_<clip>.json, mask overlay frames and masks (npz) to ../out/<tag>/<clip>/.
Frames: 1 fps, long side 1280 (1280x720 / 720x1280), cache /tmp/bench_frames/<clip>_ls<N>.
Measurement: 5 warm-up frames + N frames per prompt; time of set_image+set_text_prompt with sync before and after.
Embedl (TensorRT INT8): --model embedl --engine .../embedl_sam3_quant.engine [--conf 0.5]; time = preprocess +
engine + postprocess up to masks at frame resolution; prompts.<p>.forward separately = engine only (H2D+execute+D2H).
SAM 3 / 3.1 (facebookresearch/sam3 on PYTHONPATH): --model sam3 --ckpt .../sam3.1_multiplex.pt [--autocast bf16];
--autocast none = FP32 (the upstream bf16 MLP is replaced with FP32), bf16|fp16 = torch.autocast around set_image/set_text_prompt.
--combo: one set_image per frame and set_text_prompt for each prompt in --prompts (whole-frame time), log logs/<tag>_<clip>.json.
Memory: tegrastats --interval 500 for the whole run (peak RAM and delta to the first sample),
torch.cuda.max_memory_allocated for torch adapters.
"""
import argparse, json, os, re, statistics, subprocess, sys, threading, time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))  # embedl_trt.py is in the same directory
MASK_PROMPTS = ("person", "high-visibility vest")   # masks for IoU against SAM 3
MASK_FRAMES = range(5, 15)                          # absolute frame indices (after the 5 warm-up frames)
SCORE_THR = 0.5


def frames_from_clip(path: str, n: int, long_side: int = 1280):
    """Returns n frames at 1 fps via ffmpeg (long side = long_side) and the source file names."""
    out_dir = Path("/tmp/bench_frames") / f"{Path(path).stem}_ls{long_side}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not any(out_dir.iterdir()):
        vf = f"fps=1,scale={long_side}:{long_side}:force_original_aspect_ratio=decrease"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-vf", vf, "-q:v", "2",
                        str(out_dir / "%03d.jpg")], check=True)
    files = sorted(out_dir.glob("*.jpg"))
    while len(files) < n:  # clips are short: loop them to get enough samples
        files = files + files
    files = files[:n]
    return [Image.open(f).convert("RGB") for f in files], [f.name for f in files]


class TegraStats:
    """tegrastats in the background: peak 'RAM x/y MB' and the first sample."""
    RE = re.compile(r"RAM (\d+)/(\d+)MB")

    def __init__(self, interval_ms=500):
        self.samples, self.total = [], None
        try:
            self.p = subprocess.Popen(["tegrastats", "--interval", str(interval_ms)], stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, text=True)
        except FileNotFoundError:
            self.p = None
            return
        self.t = threading.Thread(target=self._read, daemon=True)
        self.t.start()

    def _read(self):
        for line in self.p.stdout:
            m = self.RE.search(line)
            if m:
                self.samples.append(int(m.group(1))); self.total = int(m.group(2))

    def stop(self):
        if self.p is None:
            return {"ram_peak_mb": None}
        self.p.terminate()
        try:
            self.p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.p.kill()
        if not self.samples:
            return {"ram_peak_mb": None}
        return {"ram_first_mb": self.samples[0], "ram_peak_mb": max(self.samples),
                "ram_peak_delta_mb": max(self.samples) - self.samples[0], "ram_total_mb": self.total,
                "ram_samples": len(self.samples)}


class Adapter:
    """seg is required; encode/decode (image encoder and text+decoder separately) where possible; sync is a no-op for non-torch adapters."""
    def __init__(self, seg, sync=lambda: None, encode=None, decode=None, torch_mod=None, fwd=None):
        self.seg, self.sync, self.encode, self.decode, self.torch = seg, sync, encode, decode, torch_mod
        self.fwd = fwd  # list where seg appends the pure forward time (s), if the adapter supports it


def _np(x):
    if hasattr(x, "detach"):
        x = x.detach().float().cpu() if x.dtype != __import__("torch").bool else x.detach().cpu()
        return x.numpy()
    return np.asarray(x)


def _sam3_processor_adapter(proc, autocast="none"):
    import contextlib
    import torch
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(autocast)

    def ac():  # autocast around set_image / set_text_prompt (bf16, as in the SAM 3 examples)
        return torch.autocast("cuda", dtype=dtype) if dtype is not None else contextlib.nullcontext()

    def encode(img):
        with ac():
            return proc.set_image(img)

    def decode(st, prompt):
        with ac():
            return proc.set_text_prompt(state=st, prompt=prompt)

    def seg(img, prompt):
        return decode(encode(img), prompt)

    def sync():
        torch.cuda.synchronize()
    return Adapter(seg, sync, encode, decode, torch)


def load_adapter(name: str, args):
    if name == "efficientsam3":
        from sam3.model_builder import build_efficientsam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        model = build_efficientsam3_image_model(checkpoint_path=args.ckpt, backbone_type=args.backbone,
                                                model_name=args.variant, text_encoder_type="MobileCLIP-S0",
                                                text_encoder_context_length=16, load_from_HF=False)
        return _sam3_processor_adapter(Sam3Processor(model), args.autocast)
    if name == "sam3":
        # Original SAM 3 / 3.1 (facebookresearch/sam3): local checkpoint, no HF. For sam3.1_multiplex.pt
        # _load_checkpoint takes the detector.* keys itself (the image model does not use tracker.*).
        import sam3
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        bpe = str(Path(sam3.__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz")
        if args.autocast == "none":
            # Upstream ViT MLP (vitdet.Mlp -> perflib.fused.addmm_act) hard-casts fc1+GELU to bf16, and without autocast
            # fc2 fails (BFloat16 vs Float). For the FP32 reference we replace it with the same math in FP32.
            import torch
            import sam3.model.vitdet as vitdet

            def addmm_act_fp32(activation, linear, mat1):
                y = torch.nn.functional.linear(mat1, linear.weight, linear.bias)
                if activation in (torch.nn.functional.relu, torch.nn.ReLU):
                    return torch.relu(y)
                if activation in (torch.nn.functional.gelu, torch.nn.GELU):
                    return torch.nn.functional.gelu(y)
                raise ValueError(f"Unexpected activation {activation}")
            vitdet.addmm_act = addmm_act_fp32
        model = build_sam3_image_model(bpe_path=bpe, checkpoint_path=args.ckpt, load_from_HF=False)
        return _sam3_processor_adapter(Sam3Processor(model), args.autocast)
    if name == "embedl":
        # Embedl SAM3 INT8 TensorRT: preprocess -> engine (image + tokenized_text on every call) -> postprocess.
        from embedl_trt import TrtInferenceContext, preprocess_frame_rgb, postprocess, CONTEXT_LENGTH
        from transformers import AutoTokenizer
        ctx = TrtInferenceContext(args.engine)
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        tok.pad_token_id = 0  # CLIP-style zero padding, as the engine expects
        size, has_presence = ctx.input_size, "presence_logit_dec" in ctx.output_names
        tokens, fwd = {}, []

        def seg(img, prompt):
            if prompt not in tokens:  # tokenization on CPU, cached in production; the engine's text branch runs every time
                tokens[prompt] = tok(prompt, padding="max_length", max_length=CONTEXT_LENGTH, truncation=True,
                                     return_tensors="np")["input_ids"].astype(np.int64)
            frame = np.asarray(img)
            x = preprocess_frame_rgb(frame, size)
            t = time.perf_counter()
            out = ctx.infer({"image": x, "tokenized_text": tokens[prompt]})
            fwd.append(time.perf_counter() - t)
            h, w = frame.shape[:2]
            masks, _boxes, scores = postprocess(out["pred_masks"], out["pred_logits"], out["pred_boxes"], h, w,
                                                args.conf, presence_logit=out.get("presence_logit_dec") if has_presence else None)
            return {"masks": masks, "scores": scores}
        return Adapter(seg, fwd=fwd)
    raise SystemExit(f"unknown model {name}")


def to_masks(out, size):
    """out: dict with masks [N,1,H,W] / [N,H,W] and scores [N] -> (bool[N,H,W] at frame resolution, float[N])."""
    masks, scores = _np(out["masks"]), _np(out["scores"]).reshape(-1)
    w, h = size
    if masks.ndim == 4:
        masks = masks[:, 0]
    masks = masks.astype(bool).reshape(-1, *masks.shape[-2:]) if masks.size else np.zeros((0, h, w), bool)
    if masks.shape[1:] != (h, w):
        masks = np.stack([np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h))) > 127
                          for m in masks]) if len(masks) else np.zeros((0, h, w), bool)
    return masks, scores


def overlay(img, masks, path):
    arr = np.asarray(img).copy()
    for i, m in enumerate(masks[:8]):
        color = np.array([(255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 255, 80)][i % 4])
        arr[m] = (arr[m] * 0.5 + color * 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path, quality=90)


def pct(lat, q):
    s = sorted(lat)
    return s[min(len(s) - 1, max(0, int(round(q * len(s))) - 1))]


def stats_ms(lat):
    return {"mean_ms": round(1000 * statistics.mean(lat), 1), "p95_ms": round(1000 * pct(lat, 0.95), 1),
            "min_ms": round(1000 * min(lat), 1), "max_ms": round(1000 * max(lat), 1),
            "fps": round(1 / statistics.mean(lat), 2), "n": len(lat)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", help="name for out/ and logs/ (default = --model)")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--prompts", default="person,forklift,high-visibility vest,hard hat")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--long-side", type=int, default=1280)
    ap.add_argument("--split-frames", type=int, default=20, help="frames for the separate encoder / text+decoder measurement (0 = off)")
    ap.add_argument("--ckpt"); ap.add_argument("--backbone", default="tinyvit"); ap.add_argument("--variant", default="11m")
    ap.add_argument("--engine", help="TensorRT engine (embedl)")
    ap.add_argument("--tokenizer", default="/mnt/data/work/weights/sam3.1", help="tokenizer (embedl)")
    ap.add_argument("--conf", type=float, default=SCORE_THR, help="score threshold (embedl: sigmoid(logit)*sigmoid(presence))")
    ap.add_argument("--autocast", default="none", choices=["none", "bf16", "fp16"],
                    help="torch.autocast around set_image/set_text_prompt (torch adapters)")
    ap.add_argument("--combo", action="store_true",
                    help="'several prompts per frame' mode: set_image once, then set_text_prompt for each prompt")
    args = ap.parse_args()
    tag = args.tag or args.model
    clip = Path(args.clip).stem

    tstats = TegraStats(500)
    time.sleep(1.2)  # a couple of samples before model load = baseline
    t_load = time.perf_counter()
    ad = load_adapter(args.model, args)
    load_s = time.perf_counter() - t_load
    if ad.torch is not None:
        ad.torch.cuda.reset_peak_memory_stats()

    frames, names = frames_from_clip(args.clip, args.frames + args.warmup, args.long_side)
    prompts = [p.strip() for p in args.prompts.split(",")]
    out_dir = ROOT / "out" / tag / clip
    out_dir.mkdir(parents=True, exist_ok=True)
    wpath = args.engine if args.model == "embedl" else args.ckpt
    res = {"model": args.model, "tag": tag, "clip": Path(args.clip).name, "size": list(frames[0].size),
           "backbone": args.backbone, "variant": args.variant, "load_s": round(load_s, 1),
           "weights_bytes": os.path.getsize(wpath) if wpath and os.path.exists(wpath) else None,
           "score_thr": args.conf, "prompts": {}}
    if res["weights_bytes"]:
        res["weights_mb"] = round(res["weights_bytes"] / 2**20, 1)
    res["autocast"] = args.autocast
    print(f"[{tag}] {clip} frames={len(frames)} size={frames[0].size} load={load_s:.1f}s", flush=True)

    if args.combo:
        return run_combo(ad, args, res, frames, prompts, tag, clip, tstats)

    for p in prompts:
        pslug = p.replace(" ", "_")
        for img in frames[:args.warmup]:
            ad.seg(img, p)  # warm-up
        ad.sync()
        if ad.fwd is not None:
            ad.fwd.clear()
        lat, found = [], []
        for i, img in enumerate(frames[args.warmup:]):
            idx = i + args.warmup
            ad.sync(); t = time.perf_counter()
            out = ad.seg(img, p)
            ad.sync(); lat.append(time.perf_counter() - t)
            masks, scores = to_masks(out, img.size)
            keep = scores >= args.conf
            found.append(int(keep.sum()))
            if i < 5:
                overlay(img, masks[keep], out_dir / f"{pslug}_{i:02d}.jpg")
            if p in MASK_PROMPTS and idx in MASK_FRAMES:
                np.savez_compressed(out_dir / f"masks_{pslug}_{idx:02d}.npz", masks=masks[keep],
                                    union=masks[keep].any(0) if keep.any() else np.zeros(img.size[::-1], bool),
                                    scores=scores[keep], frame=names[idx])
        r = stats_ms(lat)
        r.update({"found_mean": round(statistics.mean(found), 2), "found_min": min(found), "found_max": max(found),
                  "frames_with_found": sum(1 for f in found if f > 0), "found_per_frame": found})
        if ad.fwd:
            r["forward"] = stats_ms(ad.fwd)  # engine only: H2D + execute + D2H, as in infer_trt.py
        res["prompts"][p] = r
        print(p, {k: v for k, v in r.items() if k != "found_per_frame"}, flush=True)

    if args.split_frames and ad.encode and ad.decode:
        enc, dec = [], []
        p = prompts[0]
        for img in frames[args.warmup:args.warmup + args.split_frames]:
            ad.sync(); t0 = time.perf_counter()
            st = ad.encode(img)
            ad.sync(); t1 = time.perf_counter()
            ad.decode(st, p)
            ad.sync(); t2 = time.perf_counter()
            enc.append(t1 - t0); dec.append(t2 - t1)
        res["split"] = {"prompt": p, "image_encoder": stats_ms(enc), "text_decoder": stats_ms(dec)}
        print("split", {k: (v["mean_ms"], v["p95_ms"]) if isinstance(v, dict) else v for k, v in res["split"].items()},
              flush=True)

    if ad.torch is not None:
        res["cuda_max_allocated_mb"] = round(ad.torch.cuda.max_memory_allocated() / 2**20, 1)
        res["cuda_max_reserved_mb"] = round(ad.torch.cuda.max_memory_reserved() / 2**20, 1)
    res.update(tstats.stop())
    print("memory", {k: res.get(k) for k in ("ram_first_mb", "ram_peak_mb", "ram_peak_delta_mb", "cuda_max_allocated_mb",
                                             "cuda_max_reserved_mb", "weights_mb")}, flush=True)
    log = ROOT / "logs" / f"{tag}_{clip}.json"
    log.write_text(json.dumps(res, indent=2))
    print("saved", log)


def run_combo(ad, args, res, frames, prompts, tag, clip, tstats):
    """Frame = set_image once + len(prompts) calls of set_text_prompt; time of the whole frame and of each part."""
    if not (ad.encode and ad.decode):
        raise SystemExit("combo: adapter has no separate encode/decode (Embedl: combo = N x full call)")
    for img in frames[:args.warmup]:
        st = ad.encode(img)
        for p in prompts:
            ad.decode(st, p)
    ad.sync()
    tot, enc, dec, found = [], [], {p: [] for p in prompts}, {p: [] for p in prompts}
    for img in frames[args.warmup:]:
        ad.sync(); t0 = time.perf_counter()
        st = ad.encode(img)
        ad.sync(); t1 = time.perf_counter()
        enc.append(t1 - t0)
        for p in prompts:
            ta = time.perf_counter()
            out = ad.decode(st, p)
            ad.sync(); dec[p].append(time.perf_counter() - ta)
            found[p].append(int((_np(out["scores"]).reshape(-1) >= args.conf).sum()))
        tot.append(t1 - t0 + sum(dec[p][-1] for p in prompts))  # without mask counting on CPU, as in seg()
    res["combo"] = {"prompts": prompts, "frame": stats_ms(tot), "image_encoder": stats_ms(enc),
                    "text_decoder": {p: stats_ms(v) for p, v in dec.items()},
                    "found_mean": {p: round(statistics.mean(v), 2) for p, v in found.items()}}
    print("combo frame", res["combo"]["frame"], "enc", res["combo"]["image_encoder"]["mean_ms"],
          {p: v["mean_ms"] for p, v in res["combo"]["text_decoder"].items()}, flush=True)
    if ad.torch is not None:
        res["cuda_max_allocated_mb"] = round(ad.torch.cuda.max_memory_allocated() / 2**20, 1)
        res["cuda_max_reserved_mb"] = round(ad.torch.cuda.max_memory_reserved() / 2**20, 1)
    res.update(tstats.stop())
    print("memory", {k: res.get(k) for k in ("ram_first_mb", "ram_peak_mb", "ram_peak_delta_mb", "cuda_max_allocated_mb",
                                             "weights_mb")}, flush=True)
    log = ROOT / "logs" / f"{tag}_{clip}.json"
    log.write_text(json.dumps(res, indent=2))
    print("saved", log)


if __name__ == "__main__":
    main()
