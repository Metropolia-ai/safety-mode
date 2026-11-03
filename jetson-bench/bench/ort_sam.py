#!/usr/bin/env python3
"""EfficientSAM3 open-vocab vision+decoder ONNX through ONNX Runtime (CPU EP as an Android stand-in, CUDA EP for
reference), fed with precomputed text embeddings (.npz from text_embeddings.py). Preprocess on CPU with cv2
(resize 1008x1008 INTER_LINEAR, (x/255-0.5)/0.5) as a phone would. One session.run = one prompt.
python ort_sam.py --onnx vision_decoder_noif.onnx --emb text_embeddings.npz --ep cpu --threads 4 --clip ... --frames 10"""
import argparse, json, statistics, sys, time
from pathlib import Path
import cv2, numpy as np, onnxruntime as ort
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_sam import frames_from_clip, stats_ms
ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--onnx", required=True); ap.add_argument("--emb", required=True)
ap.add_argument("--ep", choices=["cpu", "cuda"], default="cpu"); ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--prompt", default="person"); ap.add_argument("--clip", required=True)
ap.add_argument("--frames", type=int, default=10); ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--tag", required=True)
a = ap.parse_args()
so = ort.SessionOptions(); so.intra_op_num_threads = a.threads  # 0 = ORT default (one per physical core)
prov = ["CPUExecutionProvider"] if a.ep == "cpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
t = time.perf_counter(); sess = ort.InferenceSession(a.onnx, so, providers=prov); load_s = time.perf_counter() - t
z = np.load(a.emb); i = z["prompts"].tolist().index(a.prompt)
feats, mask = z["language_features"][i][:, None].copy(), z["language_mask"][i][None].copy()
frames, _ = frames_from_clip(a.clip, a.frames + a.warmup)
tot, run, found = [], [], []
for k, img in enumerate(frames):
    t0 = time.perf_counter()
    x = cv2.resize(np.asarray(img), (1008, 1008), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    x = ((x - 0.5) / 0.5).transpose(2, 0, 1)[None].copy()
    t1 = time.perf_counter()
    det = sess.run(None, {"pixel_values": x, "language_features": feats, "language_mask": mask})[0][0]
    t2 = time.perf_counter()
    if k >= a.warmup:
        tot.append(t2 - t0); run.append(t2 - t1); found.append(int((det[:, 4] > 0.5).sum()))
res = {"onnx": a.onnx, "ep": a.ep, "providers": sess.get_providers(), "threads": a.threads or "default",
       "prompt": a.prompt, "load_s": round(load_s, 1), "frame": stats_ms(tot), "session_run": stats_ms(run),
       "found_per_frame": found, "ort": ort.__version__}
print(json.dumps(res, indent=1))
(ROOT / "logs" / f"{a.tag}_{Path(a.clip).stem}.json").write_text(json.dumps(res, indent=2))
