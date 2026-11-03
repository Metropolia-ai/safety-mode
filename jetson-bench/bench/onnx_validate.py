"""PyTorch vs ONNX Runtime (CUDA EP, FP32) on one real frame: same preprocessed tensor fed to both."""
import sys, time, numpy as np, torch, onnxruntime as ort
from PIL import Image
from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model import box_ops
D = "/mnt/data/work/export/efficientsam3-ev-m"
CK = "/mnt/data/work/weights/efficientsam3/efficientsam3_efficientvit.pt"
FR = "/tmp/bench_frames/pexels-15170997-warehouse-team_ls1280/006.jpg"
PROMPTS = ["person", "forklift", "high-visibility vest", "hard hat"]
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
model = build_efficientsam3_image_model(checkpoint_path=CK, backbone_type="efficientvit", model_name="b1",
    text_encoder_type="MobileCLIP-S0", text_encoder_context_length=16, load_from_HF=False)
proc = Sam3Processor(model)
img = Image.open(FR).convert("RGB")
from torchvision.transforms import v2
x = proc.transform(v2.functional.to_image(img).to("cuda")).unsqueeze(0)
tok = model.backbone.language_backbone.tokenizer
prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
so = ort.SessionOptions()
te = ort.InferenceSession(f"{D}/open_vocab/text_encoder.onnx", so, providers=prov)
vd = ort.InferenceSession(f"{D}/open_vocab/vision_decoder.onnx", so, providers=prov)
fx = ort.InferenceSession(f"{D}/fixed4/efficientsam3_ev-m_fixed4_mask.onnx", so, providers=prov)
print("providers", te.get_providers(), vd.get_providers(), fx.get_providers())
xn = x.cpu().numpy()
with torch.inference_mode():
    bo = model.backbone.forward_image(x)
    fo = fx.run(None, {"pixel_values": xn})
    for ci, p in enumerate(PROMPTS):
        t = model.backbone.forward_text([p], device="cuda")
        ids = tok([p], context_length=16).numpy().astype(np.int64)
        m_o, f_o, e_o = te.run(None, {"tokenized_ids": ids})
        dt = lambda a, b: float(np.abs(a - b.detach().float().cpu().numpy()).max())
        print(f"[{p}] text: mask eq={np.array_equal(m_o, t['language_mask'].cpu().numpy())} "
              f"feat maxabs={dt(f_o, t['language_features']):.2e} embeds maxabs={dt(e_o, t['language_embeds']):.2e}")
        b2 = dict(bo); b2.update(t)
        out = model.forward_grounding(backbone_out=b2, find_input=proc.find_stage,
                                      geometric_prompt=model._get_dummy_prompt(), find_target=None)
        sc = (out["pred_logits"].sigmoid() * out["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)[0].cpu().numpy()
        bx = (box_ops.box_cxcywh_to_xyxy(out["pred_boxes"]) * 1008)[0].cpu().numpy()
        det = vd.run(None, {"pixel_values": xn, "language_features": f_o, "language_mask": m_o})[0][0]
        fd = fo[0][0, ci*200:(ci+1)*200].astype(np.float32)
        fm = fo[1][0, ci*200:(ci+1)*200]
        pm = out["pred_masks"][0]
        k = sc > 0.5
        print(f"   torch keep={int(k.sum())} top={np.sort(sc)[-3:][::-1].round(3)} | ORT open-vocab keep={int((det[:,4]>0.5).sum())} "
              f"score maxabs={np.abs(det[:,4]-sc).max():.2e} box maxabs={np.abs(det[:,:4]-bx).max():.2e} (score>0.3: {np.abs(det[sc>0.3,:4]-bx[sc>0.3]).max(initial=0):.2f}) | ORT fixed keep={int((fd[:,4]>0.5).sum())} "
              f"score maxabs={np.abs(fd[:,4]-sc).max():.2e} box maxabs={np.abs(fd[:,:4]-bx).max():.2e} (score>0.3: {np.abs(fd[sc>0.3,:4]-bx[sc>0.3]).max(initial=0):.2f})")
        if k.any():
            tm = (torch.nn.functional.interpolate(pm[torch.from_numpy(k).cuda()][:, None], (1008, 1008), mode="bilinear", align_corners=False).sigmoid() > 0.5)[:, 0].cpu().numpy()
            om = fm[k] > 0.5
            iou = [(a & b).sum() / max(1, (a | b).sum()) for a, b in zip(tm, om)]
            print(f"   fixed masks IoU vs torch (kept): {np.round(iou, 4)}")
