# Jetson bench: how the benchmark was run

On-device measurements of SAM 3, SAM 3.1 and a distilled SAM 3 (EfficientSAM3) on an NVIDIA Jetson AGX Orin 64 GB, plus the Muse Glimmer 30B rule compiler (`glimmer.md`). Results: `results.md`. Raw logs: `logs/`. Mask overlays: `samples/`.

Paths under `/mnt/data/work/...` in this folder are the working directory on the board's USB drive. Replace them with your own paths on another board.

## Questions
1. **SAM 3 / 3.1 on Orin:** frames per second and memory with one text prompt at 720p, in the original precision and in INT8 (Embedl build, TensorRT).
2. **EfficientSAM3 on Orin:** the same for three backbones, plus mask quality on the test clips, by eye and by IoU against SAM 3.1.
3. **Export:** does EfficientSAM3 export to ONNX, does a TensorRT engine build from it, file sizes, and what an Android phone would need (ONNX Runtime Mobile / NNAPI). The phone itself was not measured.
4. **Cameras per board** for safety rules: one frame every 2 s, 3 prompts (person, forklift, high-visibility vest) per frame.

## Environment
- Board: Jetson AGX Orin 64 GB Developer Kit, JetPack 6.2.1 (L4T R36.4.7), Ubuntu 22.04, Python 3.10, CUDA 12.6, TensorRT 10.3. Machine passport: `logs/00_machine_passport.txt`, clocks: `logs/01_jetson_clocks.txt`.
- Power mode MAXN with `jetson_clocks`. Memory was read with `tegrastats --interval 500` (peak RAM of the whole system); for the Embedl and SAM 3.1 runs the page cache was dropped first (`sync; echo 3 > /proc/sys/vm/drop_caches`).
- PyTorch 2.8.0 and torchvision 0.23 from the NVIDIA JetPack 6 index (`https://pypi.jetson-ai-lab.io/jp6/cu126`), ONNX 1.23.2, ONNX Runtime GPU 1.23.0, all in one venv:
  ```bash
  python3 -m venv venv-sam && . venv-sam/bin/activate
  pip install -U pip wheel
  pip install torch torchvision --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
  pip install onnx onnxruntime-gpu --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
  pip install opencv-python-headless pillow av transformers huggingface_hub
  ```
- Version constraints that shaped the order: the original SAM 3 asks for Python ≥3.12 and PyTorch ≥2.7 (it ran on Python 3.10 without changes, see `results.md`); the Embedl INT8 build is validated on TensorRT 10.1/10.3 and gives wrong masks on newer versions.

## Models
Links, licences and minimal calls: `models.md`.
1. EfficientSAM3 (Apache-2.0, open weights), backbones RepViT, TinyViT and EfficientViT, M size.
2. Embedl SAM 3 INT8 (Hugging Face, accepted terms), built into a TensorRT engine on the board.
3. SAM 3.1 (`facebook/sam3.1`, access on request on Hugging Face), official `facebookresearch/sam3` code.

## Test clips
Three Pexels clips, downloaded by `footage/download.sh` into `footage/raw/` (not in git). What each clip shows: `footage/README.md`.

## Measurement protocol
`bench/bench_sam.py` runs all three models through one adapter interface. For each model and clip:
- 5 warm-up frames, then 60 frames at 1 fps with the long side 1280 px, prompts in turn: `person`, `forklift`, `high-visibility vest`, `hard hat`.
- Recorded: mean and p95 latency per frame per prompt, frames per second, peak RAM, weight size on disk.
- 5 frames with mask overlays per clip (`out/<model>/<clip>/`, a selection in `samples/`).
- Quality: IoU of the EfficientSAM3 and Embedl masks against SAM 3.1 FP32 on the same frames (`bench/iou.py`), for `person` and `high-visibility vest`.
- Three prompts per frame (`--combo`): one image encoding, then one text prompt after another.
- Video mode: SAM 3.1 on the whole clip with one `person` prompt (`bench/video_sam.py`): whether object ids persist, time per clip, memory.

## Export
- EfficientSAM3 to ONNX with the scripts from PR #49 of the EfficientSAM3 repository (opset 17), then `trtexec --fp16` for each graph (`bench/export_trtexec_build.sh`).
- Three graph fixes were needed for TensorRT 10.3 (`bench/onnx_inline_squeeze_if.py`, `bench/onnx_fp16_safe_linear_att.py`, `bench/onnx_presence_fusion_break.py`); `bench/trt_check.py` compares the engines with PyTorch on real frames, `bench/trt_sam.py` measures them end to end.
- Precomputed text embeddings (`bench/text_embeddings.py`): the device does not need the text encoder when the prompt list is fixed.
- Android stand-in: the ONNX graph through ONNX Runtime on the Orin CPU (`bench/ort_sam.py`) and the official ORT mobile usability check for NNAPI.

## Cameras per board
cameras = (1 / latency per frame with 3 prompts, in s) × 2 s, for each model variant; the table is in `results.md`, section "Capacity".
