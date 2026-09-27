# Micro-benchmark: stacked Linear layers (LLM-sized matmuls) quantized to integer, profiled on Snapdragon 7 Gen 4 via AI Hub.
import sys, time, json, numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper
import qai_hub as hub

T, D, L = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
wq = sys.argv[4]  # int8 | int4
out = f"T{T}_D{D}_L{L}_{wq}"
rng = np.random.default_rng(0)
nodes, inits = [], []
x = "x"
for i in range(L):
    W = (rng.standard_normal((D, D)) / np.sqrt(D)).astype(np.float32)
    b = np.zeros(D, np.float32)
    inits += [numpy_helper.from_array(W, f"W{i}"), numpy_helper.from_array(b, f"b{i}")]
    nodes += [helper.make_node("MatMul", [x, f"W{i}"], [f"m{i}"]),
              helper.make_node("Add", [f"m{i}", f"b{i}"], [f"a{i}"]),
              helper.make_node("Relu", [f"a{i}"], [f"y{i}"])]
    x = f"y{i}"
g = helper.make_graph(nodes, "mm", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, T, D])],
                      [helper.make_tensor_value_info(x, TensorProto.FLOAT, [1, T, D])], inits)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]); m.ir_version = 8
onnx.checker.check_model(m)
path = out + ".onnx"; onnx.save(m, path)
flops = 2 * T * D * D * L
print(f"model {path} flops {flops/1e9:.2f} GFLOP", flush=True)

dev = hub.Device("Snapdragon 7 Gen 4 QRD")
calib = {"x": [rng.standard_normal((1, T, D)).astype(np.float32) for _ in range(4)]}
wdt = hub.QuantizeDtype.INT4 if wq == "int4" else hub.QuantizeDtype.INT8
qj = hub.get_job(sys.argv[5]) if len(sys.argv) > 5 else hub.submit_quantize_job(path, calib, weights_dtype=wdt, activations_dtype=hub.QuantizeDtype.INT16, name=f"mm-{T}-{D}-{L}-{wq}a16")
print("quantize job", qj.url, flush=True)
qm = qj.get_target_model()
if qm is None:
    print("quantize failed", qj.get_status()); sys.exit(1)
cj, pj = hub.submit_compile_and_profile_jobs(qm, dev, name=f"mm-{T}-{D}-{L}-{wq}a16", compile_options="--target_runtime qnn_dlc --quantize_io")
print("compile job", cj.url, "\nprofile job", pj.url, flush=True)
prof = pj.download_profile()
if prof is None:
    print("profile failed", pj.get_status()); sys.exit(1)
json.dump(prof, open(out + "_profile.json", "w"), indent=1)
es = prof["execution_summary"]
t_us = es["estimated_inference_time"]
print(f"inference {t_us/1000:.2f} ms  ->  {flops/(t_us*1e-6)/1e9:.0f} GFLOPS  (tokens/s through {L} layers: {T/(t_us*1e-6):.0f})", flush=True)
units = {}
for op in prof.get("execution_detail", []):
    units[op.get("compute_unit")] = units.get(op.get("compute_unit"), 0) + 1
print("ops per compute unit:", units, flush=True)
