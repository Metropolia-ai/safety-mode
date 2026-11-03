#!/usr/bin/env python3
"""Video mode (tracking): the whole clip with one text prompt, propagate forward from frame 0.
SAM 3.1:  PYTHONPATH=.../sam3 python video_sam.py --model sam3.1 --ckpt .../sam3.1_multiplex.pt --clip ../footage/raw/X.mp4
Writes logs/<tag>_<clip>.json (time per phase, fps, peak RAM, ids and their lifetimes) and
out/<tag>/<clip>/frame_XXXX.jpg: a few frames across the clip with masks coloured by id and id labels."""
import argparse, json, statistics, sys, time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_sam import TegraStats  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


PALETTE = np.array([(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
                    (240, 50, 230), (210, 245, 60), (0, 128, 128), (170, 110, 40), (128, 0, 0), (0, 0, 128)])  # RGB


def color(obj_id):
    return PALETTE[int(obj_id) % len(PALETTE)]


def render(frame_bgr, masks, ids, path, long_side=1280):
    arr = frame_bgr.copy()
    for m, i in zip(masks, ids):
        if not m.any():
            continue
        c = color(i)[::-1]
        arr[m] = (arr[m] * 0.45 + c * 0.55).astype(np.uint8)
        cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(arr, cnt, -1, tuple(int(v) for v in c), 3)
    h, w = arr.shape[:2]
    s = long_side / max(h, w)
    arr = cv2.resize(arr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    for m, i in zip(masks, ids):
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        x, y = int(xs.mean() * s), int(ys.min() * s) + 18
        cv2.putText(arr, f"id {int(i)}", (x - 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(arr, f"id {int(i)}", (x - 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(str(path), arr, [cv2.IMWRITE_JPEG_QUALITY, 88])


def build(args):
    if args.model == "sam3.1":
        import sam3
        from sam3.model_builder import build_sam3_multiplex_video_predictor
        bpe = str(Path(sam3.__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz")
        # use_fa3=False: FlashAttention 3 is Hopper only; async_loading_frames=False: all frames load
        # inside start_session, so that load time does not mix with propagate.
        pred = build_sam3_multiplex_video_predictor(checkpoint_path=args.ckpt, bpe_path=bpe, use_fa3=False,
                                                    async_loading_frames=False)
        # Upstream (7f473cf): Sam3BasePredictor.start_session passes offload_state_to_cpu, but
        # Sam3MultiplexTrackingWithInteractivity.init_state does not accept it (TypeError) -> drop the extra kwargs.
        import inspect
        orig = pred.model.init_state
        allowed = set(inspect.signature(orig).parameters)
        pred.model.init_state = lambda **kw: orig(**{k: v for k, v in kw.items() if k in allowed})
        return pred
    if args.model == "efficientsam3":
        import sam3
        from sam3.model_builder import build_efficientsam3_video_predictor
        bpe = str(Path(sam3.__file__).parent.parent / "assets" / "bpe_simple_vocab_16e6.txt.gz")
        return build_efficientsam3_video_predictor(
            gpus_to_use=[0], checkpoint_path=args.ckpt, bpe_path=bpe, backbone_type=args.backbone,
            model_name=args.variant, text_encoder_type="MobileCLIP-S0", text_encoder_context_length=16,
            text_encoder_pos_embed_table_size=16, interpolate_pos_embed=False, strict_state_dict_loading=False)
    raise SystemExit(f"unknown model {args.model}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["sam3.1", "efficientsam3"])
    ap.add_argument("--tag")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--prompt", default="person")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone", default="efficientvit"); ap.add_argument("--variant", default="b1")
    ap.add_argument("--render", type=int, default=5, help="how many frames across the clip to render")
    ap.add_argument("--min-free-gb", type=float, default=8.0, help="stop propagate if free memory drops below this")
    ap.add_argument("--no-evict", action="store_true",
                    help="keep finished frames in cached_frame_outputs (default evict_cached_frame_outputs=True: "
                         "otherwise full-resolution masks of all frames pile up on the GPU)")
    args = ap.parse_args()
    tag = args.tag or f"{args.model}-video"
    clip = Path(args.clip).stem
    out_dir = ROOT / "out" / tag / clip
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.clip)
    n_frames_cv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps_src = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    render_idx = set(np.linspace(0, max(n_frames_cv - 1, 0), args.render).round().astype(int).tolist())

    import torch
    tstats = TegraStats(500)
    time.sleep(1.2)
    t = time.perf_counter()
    pred = build(args)
    load_s = time.perf_counter() - t
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    sid = pred.handle_request({"type": "start_session", "resource_path": args.clip})["session_id"]
    torch.cuda.synchronize(); t1 = time.perf_counter()
    r = pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": args.prompt})
    torch.cuda.synchronize(); t2 = time.perf_counter()
    ids0 = r["outputs"]["out_obj_ids"].tolist() if "outputs" in r else []
    per_frame, saved, stamps, memtrace, aborted = {}, {}, [], [], None
    for resp in pred.handle_stream_request({"type": "propagate_in_video", "session_id": sid,
                                            "propagation_direction": "forward",
                                            "evict_cached_frame_outputs": not args.no_evict}):
        fi = resp["frame_index"]; o = resp["outputs"]
        ids = np.asarray(o["out_obj_ids"]).tolist()
        masks = np.asarray(o["out_binary_masks"])
        nonempty = [int(i) for i, m in zip(ids, masks) if m.any()]
        per_frame[fi] = nonempty
        stamps.append(time.perf_counter())
        free_b, _ = torch.cuda.mem_get_info()  # unified memory: free = free in the whole system
        if fi % 10 == 0:
            memtrace.append((fi, round(torch.cuda.memory_allocated() / 2**20), round(free_b / 2**20)))
        if free_b < args.min_free_gb * 2**30:  # guard against the OOM killer (GUI, Ollama and t3 also run on the board)
            aborted = fi
            print(f"ABORT: free memory {free_b / 2**30:.1f} GB < {args.min_free_gb} GB at frame {fi}", flush=True)
            break
        if fi in render_idx:
            saved[fi] = (masks.astype(bool), ids)
    torch.cuda.synchronize(); t3 = time.perf_counter()
    pred.handle_request({"type": "close_session", "session_id": sid})
    mem = {"cuda_max_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
           "cuda_max_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20, 1)}
    mem.update(tstats.stop())

    n = len(per_frame)
    life = {}
    for fi in sorted(per_frame):
        for i in per_frame[fi]:
            d = life.setdefault(i, {"first": fi, "last": fi, "frames": 0})
            d["last"] = fi; d["frames"] += 1
    counts = [len(v) for v in per_frame.values()]
    dt = np.diff(stamps) if len(stamps) > 1 else np.array([0.0])
    res = {"model": args.model, "tag": tag, "clip": Path(args.clip).name, "prompt": args.prompt,
           "frames_cv": n_frames_cv, "frames_out": n, "src_fps": round(fps_src, 2), "load_s": round(load_s, 1),
           "start_session_s": round(t1 - t0, 1), "add_prompt_s": round(t2 - t1, 2),
           "propagate_s": round(t3 - t2, 1), "total_s": round(t3 - t0, 1),
           "fps_track": round(n / (t3 - t1), 2), "fps_total": round(n / (t3 - t0), 2),
           "per_frame_ms_median": round(1000 * float(np.median(dt)), 1),
           "per_frame_ms_p95": round(1000 * float(np.percentile(dt, 95)), 1),
           "ids_frame0": ids0, "ids_total": len(life), "ids_max_per_frame": max(counts, default=0),
           "ids_mean_per_frame": round(statistics.mean(counts), 2) if counts else 0,
           "frames_with_ids": sum(1 for c in counts if c), "id_life": {str(k): v for k, v in life.items()},
           "short_ids_lt15": sum(1 for v in life.values() if v["frames"] < 15),
           "evict_cached_frame_outputs": not args.no_evict, "aborted_at_frame": aborted,
           "memtrace_frame_alloc_free_mb": memtrace, "rendered": sorted(saved), **mem}
    print(json.dumps({k: v for k, v in res.items() if k not in ("id_life", "memtrace_frame_alloc_free_mb")}), flush=True)
    print("memtrace", memtrace, flush=True)
    print("id_life", life, flush=True)

    cap = cv2.VideoCapture(args.clip)
    fi = 0
    while saved:
        ok, frame = cap.read()
        if not ok:
            break
        if fi in saved:
            masks, ids = saved.pop(fi)
            np.savez_compressed(out_dir / f"masks_{fi:04d}.npz", masks=masks, ids=np.asarray(ids))
            if masks.shape[1:] != frame.shape[:2]:
                masks = np.stack([cv2.resize(m.astype(np.uint8), frame.shape[1::-1]) > 0 for m in masks]) \
                    if len(masks) else masks
            render(frame, masks, ids, out_dir / f"frame_{fi:04d}.jpg")
        fi += 1
    log = ROOT / "logs" / f"{tag}_{clip}.json"
    log.write_text(json.dumps(res, indent=2))
    print("saved", log)


if __name__ == "__main__":
    main()
