"""TensorRT engine vs PyTorch FP32 on one real frame (same preprocessed tensor): NaN count, scores, kept queries.
python trt_check.py --ckpt ... --emb text_embeddings.npz --open <vision_decoder.plan>... --fixed <fixed4.plan>..."""
import argparse, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trt_sam import Engine, CLASSES
from PIL import Image
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--backbone", default="efficientvit"); ap.add_argument("--variant", default="b1"); ap.add_argument("--emb", required=True)
ap.add_argument("--open", nargs="*", default=[]); ap.add_argument("--fixed", nargs="*", default=[])
ap.add_argument("--frame", default="/tmp/bench_frames/pexels-15170997-warehouse-team_ls1280/006.jpg")
a = ap.parse_args()
from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from torchvision.transforms import v2
model = build_efficientsam3_image_model(checkpoint_path=a.ckpt, backbone_type=a.backbone, model_name=a.variant,
    text_encoder_type="MobileCLIP-S0", text_encoder_context_length=16, load_from_HF=False)
proc = Sam3Processor(model)
x = proc.transform(v2.functional.to_image(Image.open(a.frame).convert("RGB")).cuda()).unsqueeze(0)
z = np.load(a.emb); idx = {p: i for i, p in enumerate(z["prompts"].tolist())}
ref = {}
with torch.inference_mode():
    bo = model.backbone.forward_image(x)
    for p in CLASSES:
        b2 = dict(bo); b2.update(model.backbone.forward_text([p], device="cuda"))
        o = model.forward_grounding(backbone_out=b2, find_input=proc.find_stage, geometric_prompt=model._get_dummy_prompt(), find_target=None)
        ref[p] = ((o["pred_logits"].sigmoid() * o["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)[0].cpu().numpy(),
                  o["pred_masks"][0])
print(f"frame {a.frame}; PyTorch FP32 kept (score>0.5): " + ", ".join(f"{p}={int((ref[p][0] > 0.5).sum())} top={ref[p][0].max():.3f}" for p in CLASSES))
for path in a.open:
    e = Engine(path)
    for p in CLASSES:
        f = torch.from_numpy(z["language_features"][idx[p]][:, None]).cuda(); m = torch.from_numpy(z["language_mask"][idx[p]][None]).cuda()
        d = e(pixel_values=x, language_features=f, language_mask=m)["detections"][0].float().cpu().numpy()
        s = d[:, 4]; r = ref[p][0]
        print(f"{Path(path).name} [{p}] NaN={int(np.isnan(d).sum())} kept={int((s > 0.5).sum())} top={np.nanmax(s) if not np.isnan(s).all() else float('nan'):.3f} "
              f"max|score-torch|={np.nanmax(np.abs(s - r)) if not np.isnan(s).all() else float('nan'):.3f}")
    del e; torch.cuda.empty_cache()
for path in a.fixed:
    e = Engine(path)
    o = e(pixel_values=x); d = o["detections"][0].float().cpu().numpy()
    for ci, p in enumerate(CLASSES):
        s = d[ci * 200:(ci + 1) * 200, 4]; r, rm = ref[p]
        k = np.nonzero(r > 0.5)[0]
        iou = []
        for q in k:  # mask IoU at 1008x1008 for queries PyTorch keeps
            tm = torch.nn.functional.interpolate(rm[q][None, None].float(), (1008, 1008), mode="bilinear", align_corners=False)[0, 0].sigmoid() > 0.5
            em = o["masks"][0, ci * 200 + q] > 0.5
            iou.append(float((tm & em).sum() / max(1, (tm | em).sum())))
        print(f"{Path(path).name} [{p}] NaN={int(np.isnan(d[ci*200:(ci+1)*200]).sum())} kept={int((s > 0.5).sum())} top={np.nanmax(s):.3f} "
              f"max|score-torch|={np.nanmax(np.abs(s - r)):.3f} maskIoU(kept by torch)={np.round(iou, 3).tolist()}")
