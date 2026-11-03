"""TensorRT 10.3 rejects the exporter's `If` around `scores.squeeze(-1)` (then = Squeeze -> [B,200], else = Identity ->
[B,200,1]: "IIfConditionalOutputLayer inputs must have the same shape"). The squeezed dim is always 1 (pred_logits
[B,Q,1] * presence [B,1,1]), so the then-branch is inlined into the main graph and the If removed.
python onnx_inline_squeeze_if.py in.onnx out.onnx"""
import sys, onnx
src, dst = sys.argv[1:3]
m = onnx.load(src)
g = m.graph
new, n_fixed = [], 0
for n in g.node:
    if n.op_type == "If":
        br = {a.name: a.g for a in n.attribute}
        th, el = br["then_branch"], br["else_branch"]
        ok = ([x.op_type for x in th.node] == ["Constant", "Squeeze"] and [x.op_type for x in el.node] == ["Identity"]
              and th.node[1].input[0] == el.node[0].input[0])
        if not ok:
            raise SystemExit(f"unexpected If {n.name}: {[x.op_type for x in th.node]} / {[x.op_type for x in el.node]}")
        c, sq = th.node
        axes = onnx.numpy_helper.to_array(c.attribute[0].t).tolist()
        assert axes in ([-1], -1), axes
        cname = n.name + "_axes"
        new.append(onnx.helper.make_node("Constant", [], [cname], value=c.attribute[0].t, name=cname))
        new.append(onnx.helper.make_node("Squeeze", [sq.input[0], cname], [n.output[0]], name=n.name + "_squeeze"))
        n_fixed += 1
        continue
    new.append(n)
del g.node[:]
g.node.extend(new)
onnx.checker.check_model(m)
onnx.save(m, dst)
print(f"inlined {n_fixed} If node(s) -> {dst}")
