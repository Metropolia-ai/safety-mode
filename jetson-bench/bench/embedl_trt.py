# Adapted from embedl/sam3 infer_trt.py (Copyright (C) 2026 Embedl AB): TrtInferenceContext,
# preprocess_frame_rgb, postprocess copied verbatim-ish; top-level demo removed, libcudart path fallback added.
"""Embedl SAM3 INT8 TensorRT engine: pure TensorRT + numpy + cv2 (no torch).

Inputs: image [1,3,924,924] float32, tokenized_text [1,32] int64.
Outputs: pred_masks [1,200,264,264], pred_logits [1,200,1], pred_boxes [1,200,4] (cxcywh norm), presence_logit_dec [1,1].
"""
import ctypes
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt

CONTEXT_LENGTH = 32
MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

try:
    _cudart = ctypes.CDLL("libcudart.so")
except OSError:
    _cudart = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")

_TRT_NP_DTYPE = {
    trt.float32: np.float32,
    trt.float16: np.float16,
    trt.int32: np.int32,
    trt.int64: np.int64,
    trt.int8: np.int8,
    trt.bool: np.bool_,
}


def _cuda_malloc(nbytes: int) -> int:
    ptr = ctypes.c_void_p()
    ret = _cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes))
    if ret != 0:
        raise RuntimeError(f"cudaMalloc failed with error {ret}")
    return ptr.value


def _cuda_free(ptr: int) -> None:
    _cudart.cudaFree(ctypes.c_void_p(ptr))


def _cuda_memcpy_h2d(dst: int, src: np.ndarray) -> None:
    _cudart.cudaMemcpy(ctypes.c_void_p(dst), src.ctypes.data_as(ctypes.c_void_p),
                       ctypes.c_size_t(src.nbytes), ctypes.c_int(1))  # cudaMemcpyHostToDevice


def _cuda_memcpy_d2h(dst: np.ndarray, src: int) -> None:
    _cudart.cudaMemcpy(dst.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(src),
                       ctypes.c_size_t(dst.nbytes), ctypes.c_int(2))  # cudaMemcpyDeviceToHost


def _cuda_stream_create() -> ctypes.c_void_p:
    stream = ctypes.c_void_p()
    ret = _cudart.cudaStreamCreate(ctypes.byref(stream))
    if ret != 0:
        raise RuntimeError(f"cudaStreamCreate failed with error {ret}")
    return stream


def _cuda_stream_synchronize(stream: ctypes.c_void_p) -> None:
    _cudart.cudaStreamSynchronize(stream)


class TrtInferenceContext:
    """Synchronous TensorRT inference with buffers pre-allocated once (Embedl AB)."""

    def __init__(self, engine_path):
        engine_path = Path(engine_path)

        class _Logger(trt.ILogger):
            def log(self, severity, msg):
                if severity <= trt.Logger.WARNING:
                    print(f"[TRT] {msg.rstrip()}")

        with engine_path.open("rb") as f, trt.Runtime(_Logger()) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = _cuda_stream_create()

        self.input_names, self.output_names = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        self._d_inputs, self._d_outputs, self._h_outputs = {}, {}, {}
        for name in self.input_names:
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = _TRT_NP_DTYPE[self.engine.get_tensor_dtype(name)]
            ptr = _cuda_malloc(int(np.prod(shape) * np.dtype(dtype).itemsize))
            self._d_inputs[name] = ptr
            self.context.set_tensor_address(name, ptr)
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = _TRT_NP_DTYPE[self.engine.get_tensor_dtype(name)]
            ptr = _cuda_malloc(int(np.prod(shape) * np.dtype(dtype).itemsize))
            self._d_outputs[name] = ptr
            self.context.set_tensor_address(name, ptr)
            self._h_outputs[name] = np.empty(shape, dtype=dtype)

    def __del__(self):
        for ptr in list(self._d_inputs.values()) + list(self._d_outputs.values()):
            _cuda_free(ptr)

    @property
    def input_size(self) -> int:
        return self.engine.get_tensor_shape("image")[2]

    def synchronize(self):
        _cuda_stream_synchronize(self.stream)

    def infer(self, inputs: dict) -> dict:
        for name, arr in inputs.items():
            arr = np.ascontiguousarray(arr, dtype=_TRT_NP_DTYPE[self.engine.get_tensor_dtype(name)])
            _cuda_memcpy_h2d(self._d_inputs[name], arr)
        self.context.execute_async_v3(self.stream.value)
        _cuda_stream_synchronize(self.stream)
        for name in self.output_names:
            _cuda_memcpy_d2h(self._h_outputs[name], self._d_outputs[name])
        return self._h_outputs


def preprocess_frame_rgb(frame_rgb: np.ndarray, input_size: int) -> np.ndarray:
    """Resize, normalise and convert an RGB frame to model input [1,3,H,W]."""
    resized = cv2.resize(frame_rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    img = resized.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return img.transpose(2, 0, 1)[np.newaxis]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def box_cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return np.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], axis=-1)


def postprocess(pred_masks, pred_logits, pred_boxes, img_h, img_w, confidence, presence_logit=None):
    scores = sigmoid(pred_logits[0, :, 0])
    if presence_logit is not None:
        scores = scores * sigmoid(presence_logit[0])
    keep = scores > confidence
    scores = scores[keep]
    boxes = pred_boxes[0][keep]
    masks = pred_masks[0][keep]
    if len(scores) == 0:
        return (np.zeros((0, img_h, img_w), dtype=bool), np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32))
    boxes_xyxy = box_cxcywh_to_xyxy(boxes) * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    binary_masks = np.stack([sigmoid(cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_LINEAR)) > 0.5
                             for m in masks])
    return binary_masks, boxes_xyxy, scores
