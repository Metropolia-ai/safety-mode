"""FP16-safe EfficientViT ReLU linear attention (LiteMLA.relu_linear_att) in an exported EfficientSAM3 ONNX graph.
out = (pad(v,1) @ relu(k)^T) @ relu(q); out = out[:-1] / (out[-1] + eps). On real frames |out| reaches ~2e5 (> FP16 max
65504; measured at 1008x1008, stages 2-3) -> TensorRT FP16 gives NaN, and TensorRT 10.3 fuses these MatMuls into a Myelin
region that ignores per-layer FP32 constraints (PR #49 workaround has no effect here). Fix: multiply relu(k)^T by 2^-6
(exact in FP32/FP16): numerator and denominator scale identically, so the ratio is unchanged; eps 1e-15 (0 in FP16 ->
0/0 = NaN where the denominator is 0) becomes 1e-4 in the scaled domain (= 6.4e-3 unscaled; observed non-zero
denominators >= 7). python onnx_fp16_safe_linear_att.py in.onnx out.onnx"""
import sys, numpy as np, onnx
from onnx import numpy_helper, helper
SCALE, EPS = 2.0 ** -6, 1e-4
src, dst = sys.argv[1:3]
m = onnx.load(src); g = m.graph
consts = {n.output[0]: n for n in g.node if n.op_type == "Constant"}
new, n_mul, n_eps = [], 0, 0
g.initializer.append(numpy_helper.from_array(np.array(SCALE, np.float32), "linear_att_k_scale"))
for n in g.node:
    if "context_module/main/" in n.name and n.op_type == "Transpose" and n.input[0].endswith("kernel_func_1/Relu_output_0"):
        new.append(n)
        out = n.output[0] + "_scaled"
        new.append(helper.make_node("Mul", [n.output[0], "linear_att_k_scale"], [out], name=n.name + "_kscale"))
        for c in g.node:  # rewire the consumer MatMul (v @ k^T)
            if c.op_type == "MatMul":
                for i, x in enumerate(c.input):
                    if x == n.output[0]:
                        c.input[i] = out
        n_mul += 1
        continue
    if "context_module/main/" in n.name and n.op_type == "Add" and n.name.endswith("main/Add") and n.input[1] in consts:
        a = numpy_helper.to_array(consts[n.input[1]].attribute[0].t)
        if a.size == 1 and float(a) < 1e-10:
            name = n.name + "_eps"
            g.initializer.append(numpy_helper.from_array(np.array(EPS, np.float32), name))
            n.input[1] = name; n_eps += 1
    new.append(n)
del g.node[:]; g.node.extend(new)
onnx.checker.check_model(m); onnx.save(m, dst)
print(f"k scaled by {SCALE} in {n_mul} LiteMLA blocks, eps -> {EPS} in {n_eps}; saved {dst}")
