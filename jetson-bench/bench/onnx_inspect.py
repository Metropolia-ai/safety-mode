# python bench/onnx_inspect.py <graph.onnx>...: inputs/outputs, opset, onnx.checker (full_check), Counter(op_type).
import sys, os, collections, onnx
for p in sys.argv[1:]:
    m = onnx.load(p, load_external_data=False)
    def shp(v):
        t = v.type.tensor_type
        return [d.dim_param or d.dim_value for d in t.shape.dim], onnx.TensorProto.DataType.Name(t.elem_type)
    print(f"== {p}  size={os.path.getsize(p)/2**20:.1f} MiB  opset={[ (o.domain or 'ai.onnx', o.version) for o in m.opset_import]}  ir={m.ir_version} producer={m.producer_name} {m.producer_version}")
    ext = [i.name for i in m.graph.initializer if i.data_location == onnx.TensorProto.EXTERNAL]
    print("  external data tensors:", len(ext))
    for v in m.graph.input: print("  in ", v.name, *shp(v))
    for v in m.graph.output: print("  out", v.name, *shp(v))
    print("  nodes", len(m.graph.node), "initializers", len(m.graph.initializer))
    try:
        onnx.checker.check_model(p, full_check=True); print("  checker: OK (full_check)")
    except Exception as e: print("  checker: FAIL", type(e).__name__, str(e)[:300])
    c = collections.Counter(n.op_type for n in m.graph.node)
    print("  ops:", dict(sorted(c.items())))
