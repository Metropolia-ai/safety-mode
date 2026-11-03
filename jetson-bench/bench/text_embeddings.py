"""Precompute EfficientSAM3 text embeddings ("phone trick"): SimpleTokenizer (Python) -> text_encoder.onnx (ORT CPU EP)
for a prompt list, saved batch-first to .npz; checked against the PyTorch text encoder (CPU, FP32).
The vision+decoder graph wants language_features [16,1,256] and language_mask [1,16] per prompt:
  feats = npz["language_features"][i][:, None]; mask = npz["language_mask"][i][None]
python text_embeddings.py --onnx .../text_encoder.onnx --ckpt .../efficientsam3_efficientvit.pt --out .../text_embeddings.npz"""
import argparse, os, time, numpy as np, onnxruntime as ort, torch
PROMPTS = ["person", "forklift", "high-visibility vest", "hard hat",
           "pallet", "pallet jack", "reach truck", "safety goggles", "gloves", "ladder", "fire extinguisher",
           "emergency exit", "cone", "barrier", "truck", "box", "conveyor", "pedestrian", "worker lying down",
           "face mask", "ear protection", "safety boots", "spill", "cable", "door", "tool", "chain", "hook",
           "scissor lift", "crane"]
ap = argparse.ArgumentParser()
ap.add_argument("--onnx", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--backbone", default="efficientvit"); ap.add_argument("--variant", default="b1"); ap.add_argument("--out", required=True)
a = ap.parse_args()
from sam3.model_builder import build_efficientsam3_image_model
model = build_efficientsam3_image_model(checkpoint_path=a.ckpt, backbone_type=a.backbone, model_name=a.variant,
    text_encoder_type="MobileCLIP-S0", text_encoder_context_length=16, load_from_HF=False, device="cpu")
tok = model.backbone.language_backbone.tokenizer
sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
ids = tok(PROMPTS, context_length=16).numpy().astype(np.int64)        # [N,16]
sess.run(None, {"tokenized_ids": ids[:1]})                               # warm-up
t = time.perf_counter(); per = [sess.run(None, {"tokenized_ids": ids[i:i+1]}) for i in range(len(PROMPTS))]
dt = (time.perf_counter() - t) / len(PROMPTS)
mask = np.concatenate([p[0] for p in per], 0)                           # [N,16] bool, True = padding
feats = np.concatenate([p[1] for p in per], 1).transpose(1, 0, 2)       # [N,16,256]
embeds = np.concatenate([p[2] for p in per], 1).transpose(1, 0, 2)      # [N,16,512]
with torch.inference_mode():
    ref = model.backbone.forward_text(PROMPTS, device="cpu")
d_f = float(np.abs(feats - ref["language_features"].transpose(0, 1).numpy()).max())
d_e = float(np.abs(embeds - ref["language_embeds"].transpose(0, 1).numpy()).max())
assert np.array_equal(mask, ref["language_mask"].numpy())
np.savez(a.out, prompts=np.array(PROMPTS), tokenized_ids=ids, language_mask=mask, language_features=feats,
         language_embeds=embeds)
n = len(PROMPTS)
print(f"{n} prompts, ORT CPU text encoder {1000*dt:.1f} ms/prompt; max|ORT-torch| features {d_f:.2e} embeds {d_e:.2e}")
print(f"tokens per prompt (non-pad): {[int((~m).sum()) for m in mask]}")
print(f"bytes/prompt fp32: features {feats[0].nbytes} + mask {mask[0].nbytes} + embeds {embeds[0].nbytes} "
      f"(+ids {ids[0].nbytes}); needed by vision graph = features+mask {feats[0].nbytes + mask[0].nbytes}")
print(f"npz {a.out}: {os.path.getsize(a.out)} bytes ({os.path.getsize(a.out)/2**10:.0f} KiB) for {n} prompts; "
      f"features+mask only: {n*(feats[0].nbytes+mask[0].nbytes)/2**10:.0f} KiB fp32, {n*(feats[0].nbytes//2+mask[0].nbytes)/2**10:.0f} KiB fp16")
print(f"text_encoder.onnx on disk: {os.path.getsize(a.onnx)/2**20:.1f} MiB (not shipped to the phone with precomputed embeddings)")
