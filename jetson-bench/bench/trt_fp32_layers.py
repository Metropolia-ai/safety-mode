"""Print trtexec --layerPrecisions/--layerOutputTypes spec pinning to FP32 every TensorRT layer whose name contains
MATCH (default 'context_module', EfficientViT ReLU linear attention that overflows FP16 -- PR #49) and whose outputs
are float; same selection as build_mixed_precision_engine() in export_onnx_tensorrt.py.
python trt_fp32_layers.py <graph.onnx> [match] > spec.txt"""
import sys
import tensorrt as trt
onnx_path = sys.argv[1]; match = sys.argv[2] if len(sys.argv) > 2 else "context_module"
logger = trt.Logger(trt.Logger.ERROR)
b = trt.Builder(logger)
net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
p = trt.OnnxParser(net, logger)
assert p.parse(open(onnx_path, "rb").read()), [p.get_error(i) for i in range(p.num_errors)]
names = []
for i in range(net.num_layers):
    l = net.get_layer(i)
    if match in l.name and all(l.get_output(k).dtype in (trt.float32, trt.float16) for k in range(l.num_outputs)):
        names.append(l.name)
assert not any("," in n or ":" in n for n in names), "layer name with , or : breaks trtexec spec"
print(f"{len(names)}/{net.num_layers} layers", file=sys.stderr)
print(",".join(f"{n}:fp32" for n in names))
