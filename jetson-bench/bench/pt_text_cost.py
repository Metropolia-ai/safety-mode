#!/usr/bin/env python3
"""How much of PyTorch EfficientSAM3 set_text_prompt is the text encoder: per frame set_image once, then for each prompt
(a) set_text_prompt as in bench_sam.py (forward_text + forward_grounding + mask postprocess),
(b) forward_text alone, (c) _forward_grounding with the text outputs precomputed (cached once per prompt).
FP32 with torch defaults (cuDNN TF32 on), torch.cuda.synchronize around each part. Same frames as bench_sam.py."""
import argparse, json, statistics, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_sam import frames_from_clip, stats_ms
ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--clip", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--prompts", default="person,forklift,high-visibility vest")
ap.add_argument("--frames", type=int, default=20); ap.add_argument("--warmup", type=int, default=5)
a = ap.parse_args()
from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
model = build_efficientsam3_image_model(checkpoint_path=a.ckpt, backbone_type="efficientvit", model_name="b1",
    text_encoder_type="MobileCLIP-S0", text_encoder_context_length=16, load_from_HF=False)
proc = Sam3Processor(model)
prompts = [p.strip() for p in a.prompts.split(",")]
frames, _ = frames_from_clip(a.clip, a.frames + a.warmup)
S = torch.cuda.synchronize
with torch.inference_mode():
    cache = {p: model.backbone.forward_text([p], device="cuda") for p in prompts}
    enc, full, txt, dec = [], {p: [] for p in prompts}, {p: [] for p in prompts}, {p: [] for p in prompts}
    for k, img in enumerate(frames):
        S(); t0 = time.perf_counter(); st = proc.set_image(img); S(); t1 = time.perf_counter()
        for p in prompts:
            S(); ta = time.perf_counter(); proc.set_text_prompt(state=st, prompt=p); S(); tb = time.perf_counter()
            model.backbone.forward_text([p], device="cuda"); S(); tc = time.perf_counter()
            st["backbone_out"].update(cache[p]); proc._forward_grounding(st); S(); td = time.perf_counter()
            if k >= a.warmup:
                full[p].append(tb - ta); txt[p].append(tc - tb); dec[p].append(td - tc)
        if k >= a.warmup:
            enc.append(t1 - t0)
allf = [v for p in prompts for v in full[p]]; allt = [v for p in prompts for v in txt[p]]; alld = [v for p in prompts for v in dec[p]]
res = {"prompts": prompts, "frames": a.frames, "image_encoder": stats_ms(enc), "set_text_prompt": stats_ms(allf),
       "text_encoder_only": stats_ms(allt), "grounding_with_precomputed_text": stats_ms(alld),
       "per_prompt": {p: {"set_text_prompt": stats_ms(full[p])["mean_ms"], "text": stats_ms(txt[p])["mean_ms"],
                          "grounding": stats_ms(dec[p])["mean_ms"]} for p in prompts}}
n = len(prompts)
res["frame_3prompt_ms"] = {"set_image+N*set_text_prompt": round(1000 * (statistics.mean(enc) + n * statistics.mean(allf)), 1),
                           "set_image+N*grounding(precomputed)": round(1000 * (statistics.mean(enc) + n * statistics.mean(alld)), 1)}
print(json.dumps(res, indent=1))
(ROOT / "logs" / f"{a.tag}_{Path(a.clip).stem}.json").write_text(json.dumps(res, indent=2))
