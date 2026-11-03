"""Work around a TensorRT 10.3 miscompile of the decoder presence path (presence_token_out_norm + presence_token_head,
MLP 256-256-256-1 on the presence token, once per decoder layer): in the built engine the final presence logit comes out
wrong (FP32 and FP16 engines alike; e.g. ORT 6.97 vs TRT -0.32 on frame 006, prompt person -> every score x0.42) although
the presence-token input matches ORT to 3e-3. Exposing the intermediate tensors of that path as graph outputs stops the
offending fusion: exposing only the activation before the last Linear fixed the open-vocab graph but not the fixed-class
graph; exposing every LayerNormalization/Add/Relu output of the path fixes both (scores match ORT to <0.07 in FP16).
The extra outputs are [1,1,256] or smaller and are ignored by the caller. python onnx_presence_fusion_break.py in out"""
import sys, onnx
from onnx import helper, TensorProto
src, dst = sys.argv[1:3]
m = onnx.load(src)
have = {o.name for o in m.graph.output}
names = [n.output[0] for n in m.graph.node
         if ("presence_token_head" in n.name or "presence_token_out_norm" in n.name)
         and n.op_type in ("LayerNormalization", "Add", "Relu") and n.output[0] not in have]
for x in names:
    m.graph.output.append(helper.make_tensor_value_info(x, TensorProto.FLOAT, ["batch", 1, None]))
onnx.checker.check_model(m); onnx.save(m, dst)
print(f"exposed {len(names)} presence-path tensors as extra outputs (e.g. {names[:2]}); saved {dst}")
