# Models under test

| Model | Where | Access | License | What it does | Requirements |
|---|---|---|---|---|---|
| EfficientSAM3 (distilled SAM 3) | https://github.com/SimonZeng7108/efficientsam3 ; weights https://huggingface.co/Simon7108528/EfficientSAM3 | open | Apache-2.0 (upstream parts under their own licenses) | text prompts via MobileCLIP-S0/S1, masks, scores; video tracking announced (README_stage3) | Python ≥3.10, torch ≥2.0, CUDA ≥11.8 |
| Embedl SAM 3 INT8/FP16 | https://huggingface.co/embedl/sam3 | login + accept terms | Embedl Models Community Licence v1.0 (no hosting as a service) + SAM License | text prompts, video; ONNX `embedl_sam3_quant.onnx` + ~3.1 GB `.onnx.data`, script `infer_trt.py` | TensorRT 10.1 or 10.3 (newer gives wrong masks), `trtexec --fp16 --int8` |
| SAM 3.1 original | code https://github.com/facebookresearch/sam3 ; weights https://huggingface.co/facebook/sam3.1 (and facebook/sam3) | on request on HF | SAM License | 848M parameters, detector + tracker, text/image/video | Python ≥3.12, torch ≥2.7, CUDA ≥12.6 |

## Installation and minimal call

### EfficientSAM3
```bash
git clone https://github.com/SimonZeng7108/efficientsam3 && cd efficientsam3
pip install -e ".[stage1]"
# weights (example: the three M backbones + LiteText text encoders):
hf download Simon7108528/EfficientSAM3 --include "efficientsam3_ft/*" --local-dir weights
hf download Simon7108528/EfficientSAM3 --include "sam3_litetext/*" --local-dir weights
```
```python
from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image
model = build_efficientsam3_image_model(
    checkpoint_path="weights/efficientsam3_ft/efficientsam3_tinyvit.pt",
    backbone_type="tinyvit", model_name="11m",
    text_encoder_type="MobileCLIP-S0", text_encoder_context_length=16, load_from_HF=False)
proc = Sam3Processor(model)
state = proc.set_image(Image.open("frame.jpg").convert("RGB"))
state = proc.set_text_prompt("person", state)
masks, scores = state["masks"], state["scores"]
```
Variants: `backbone_type` ∈ {repvit, tinyvit, efficientvit}, `model_name` per the README (RepViT M0.9/M1.1/M2.3, TinyViT 5M/11M/21M, EfficientViT B0/B1/B2). Claimed quality: COCO mIoU 61.6–66.3 %, LiteText cgF1 53.6–54.0 against 54.1 for SAM 3.

ONNX/TensorRT export: PR #49 in the repository (linked from the README). The README has no instructions, so reproduce it from the PR.

### Embedl SAM 3 INT8
```bash
hf auth login   # a Hugging Face token with the accepted terms
hf download embedl/sam3 --local-dir embedl-sam3
cd embedl-sam3
trtexec --onnx=embedl_sam3_quant.onnx --fp16 --int8 --saveEngine=sam3_int8.plan   # check the flags in the repo README
pip install opencv-python-headless transformers av
python infer_trt.py --help
```
Claimed on AGX Orin: 462 ms/request INT8 (2.17 qps) against 763 ms FP16; cgF1 53.77 against 55.56 for FP32.

### SAM 3.1 original
```bash
# separate environment with Python 3.12
uv venv --python 3.12 ~/venv-sam3 && . ~/venv-sam3/bin/activate
pip install torch torchvision --index-url https://pypi.jetson-ai-lab.io/jp6/cu126   # needs cp312 wheels on the index
git clone https://github.com/facebookresearch/sam3.git && cd sam3 && pip install -e .
hf auth login && hf download facebook/sam3.1 --local-dir weights
```
```python
from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
from sam3.model.sam3_image_processor import Sam3Processor
model = build_sam3_image_model(); proc = Sam3Processor(model)
state = proc.set_image(img); out = proc.set_text_prompt(state=state, prompt="person")
masks, boxes, scores = out["masks"], out["boxes"], out["scores"]
# video
vp = build_sam3_video_predictor()
r = vp.handle_request(request=dict(type="start_session", resource_path="clip.mp4"))
r = vp.handle_request(request=dict(type="add_prompt", session_id=r["session_id"], frame_index=0, text="person"))
```

## Other candidates (not measured)
- EdgeTAM (Meta, SAM 2 tracking for mobile): https://github.com/facebookresearch/EdgeTAM. Click/box prompt, 16 fps on iPhone 15 Pro Max. A candidate for the tracker in a "light text detector + EdgeTAM" pipeline.
- PyTorch wheels for JetPack: https://forums.developer.nvidia.com/t/pytorch-and-torchvision-for-jetpack-6-2/325257
