#!/usr/bin/env python3
"""EfficientSAM3 TensorRT engines on real frames, end to end (preprocess + engine + postprocess), torch for GPU buffers.
--mode fixed : fixed-class engine (pixel_values -> detections [1,N*200,6] fp16, masks [1,N*200,1008,1008] fp16 sigmoid);
               one call per frame = all baked classes. Postprocess on GPU: keep score > thr, bilinear resize of kept masks
               to frame size, > 0.5, copy to host as bool.
--mode open  : open-vocab vision+decoder engine fed with precomputed text embeddings (.npz from text_embeddings.py);
               one call per prompt (backbone inside the graph, so N prompts = N full calls); output = boxes + scores only.
Preprocess = Sam3Processor transform (uint8 -> Resize 1008x1008 antialias -> /255 -> (x-0.5)/0.5) on GPU, as the
PyTorch bench, so detections are comparable. Frames: same cache/loop as bench_sam.py (1 fps, long side 1280).
Writes logs/<tag>_<clip>.json, overlays out/<tag>/<clip>/, person masks of frames 5..14 for IoU vs PyTorch."""
import argparse, json, statistics, sys, time
from pathlib import Path
import numpy as np, torch, tensorrt as trt
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_sam import frames_from_clip, stats_ms, TegraStats, MASK_FRAMES

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["person", "forklift", "high-visibility vest", "hard hat"]
COLORS = [(255, 70, 70), (70, 220, 70), (255, 230, 0), (70, 140, 255)]
DT = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32, trt.int64: torch.int64,
      trt.bool: torch.bool, trt.int8: torch.int8}


class Engine:
    def __init__(self, path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.buf, self.inputs, self.outputs = {}, [], []
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(n))
            assert all(d > 0 for d in shape), f"dynamic dim in {n}{shape}: build with static --shapes"
            self.buf[n] = torch.empty(shape, dtype=DT[self.engine.get_tensor_dtype(n)], device="cuda")
            self.ctx.set_tensor_address(n, self.buf[n].data_ptr())
            (self.inputs if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT else self.outputs).append(n)
        self.stream = torch.cuda.Stream()

    def __call__(self, **inp):
        for k, v in inp.items():
            self.buf[k].copy_(v)
        torch.cuda.current_stream().synchronize()
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return {n: self.buf[n] for n in self.outputs}


def overlay(img, masks, cls, scores, path):
    arr = np.asarray(img).astype(np.float32)
    for m, c in zip(masks, cls):
        arr[m] = arr[m] * 0.45 + np.array(COLORS[c]) * 0.55
    im = Image.fromarray(arr.astype(np.uint8)); d = ImageDraw.Draw(im)
    for m, c, s in zip(masks, cls, scores):
        ys, xs = np.nonzero(m)
        if len(xs):
            d.rectangle([xs.min(), ys.min(), xs.max(), ys.max()], outline=COLORS[c], width=3)
            d.text((xs.min() + 4, max(0, ys.min() - 14)), f"{CLASSES[c]} {s:.2f}", fill=COLORS[c])
    im.save(path, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fixed", "open"], required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--emb", help="text_embeddings.npz (open)")
    ap.add_argument("--prompts", default="person,forklift,high-visibility vest,hard hat")
    ap.add_argument("--clip", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--frames", type=int, default=60); ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--conf", type=float, default=0.5); ap.add_argument("--overlays", type=int, default=3)
    a = ap.parse_args()
    clip = Path(a.clip).stem
    ts = TegraStats(500); time.sleep(1.2)
    t0 = time.perf_counter(); eng = Engine(a.engine); load_s = time.perf_counter() - t0
    from torchvision.transforms import v2
    tf = v2.Compose([v2.ToDtype(torch.uint8, scale=True), v2.Resize(size=(1008, 1008)),
                     v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
    frames, names = frames_from_clip(a.clip, a.frames + a.warmup)
    W, H = frames[0].size
    out_dir = ROOT / "out" / a.tag / clip; out_dir.mkdir(parents=True, exist_ok=True)
    prompts = [p.strip() for p in a.prompts.split(",")]
    if a.mode == "open":
        z = np.load(a.emb); idx = {p: i for i, p in enumerate(z["prompts"].tolist())}
        emb = {p: (torch.from_numpy(z["language_features"][idx[p]][:, None]).cuda(),
                   torch.from_numpy(z["language_mask"][idx[p]][None]).cuda()) for p in prompts}
    res = {"mode": a.mode, "engine_path": a.engine, "engine_mb": round(Path(a.engine).stat().st_size / 2**20, 1),
           "clip": Path(a.clip).name, "size": [W, H], "load_s": round(load_s, 2), "conf": a.conf, "prompts": prompts}
    lat, pre, run, post, found = [], [], [], [], {p: [] for p in prompts}
    for fi, img in enumerate(frames):
        torch.cuda.synchronize(); t = time.perf_counter()
        x = tf(v2.functional.to_image(img).cuda()).unsqueeze(0)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        kept = []  # (class_idx, score, mask_bool_numpy or None)
        if a.mode == "fixed":
            o = eng(pixel_values=x)
            t2 = time.perf_counter()
            det = o["detections"][0].float()
            sel = torch.nonzero(det[:, 4] > a.conf).flatten()
            if len(sel):
                m = torch.nn.functional.interpolate(o["masks"][0, sel][:, None].float(), (H, W), mode="bilinear",
                                                    align_corners=False)[:, 0] > 0.5
                m = m.cpu().numpy(); dsel = det[sel].cpu().numpy()
                kept = [(int(d[5]), float(d[4]), mm) for d, mm in zip(dsel, m)]
            run_s = t2 - t1
        else:
            run_s = 0.0
            for pi, p in enumerate(prompts):
                ta = time.perf_counter()
                o = eng(pixel_values=x, language_features=emb[p][0], language_mask=emb[p][1])
                run_s += time.perf_counter() - ta
                det = o["detections"][0]
                d = det[det[:, 4] > a.conf].cpu().numpy()
                kept += [(CLASSES.index(p) if p in CLASSES else pi, float(r[4]), None) for r in d]
        torch.cuda.synchronize(); t3 = time.perf_counter()
        if fi < a.warmup:
            continue
        i = fi - a.warmup
        lat.append(t3 - t); pre.append(t1 - t); run.append(run_s); post.append(t3 - t1 - run_s)
        for p in prompts:
            ci = CLASSES.index(p) if p in CLASSES else prompts.index(p)
            found[p].append(sum(1 for k in kept if k[0] == ci))
        if a.mode == "fixed":
            if i < a.overlays:
                overlay(img, [k[2] for k in kept], [k[0] for k in kept], [k[1] for k in kept], out_dir / f"all_{i:02d}.jpg")
            if fi in MASK_FRAMES:
                pm = [k[2] for k in kept if k[0] == 0]
                np.savez_compressed(out_dir / f"masks_person_{fi:02d}.npz", masks=np.array(pm, bool).reshape(-1, H, W),
                                    union=np.any(pm, 0) if pm else np.zeros((H, W), bool), frame=names[fi])
    res.update({"frame": stats_ms(lat), "preprocess": stats_ms(pre), "engine": stats_ms(run), "postprocess": stats_ms(post),
                "found_mean": {p: round(statistics.mean(v), 2) for p, v in found.items()},
                "frames_with_found": {p: sum(1 for f in v if f) for p, v in found.items()},
                "found_per_frame": found, "cuda_max_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)})
    res.update(ts.stop())
    print(json.dumps({k: v for k, v in res.items() if k != "found_per_frame"}, indent=1))
    (ROOT / "logs" / f"{a.tag}_{clip}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
