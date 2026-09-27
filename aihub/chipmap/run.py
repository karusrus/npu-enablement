# Map HTP fp16 support across Snapdragon chipsets: tiny float model, compile+profile on one device per chipset,
# then read the QNN runtime log line "Detected Qualcomm SOC=... Hexagon arch=... fp16=...".
import json, re, sys, time, numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper
import qai_hub as hub
W = numpy_helper.from_array(np.eye(32, dtype=np.float32), "W")
g = helper.make_graph([helper.make_node("MatMul", ["x", "W"], ["y"])], "tiny",
                      [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 32])],
                      [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 32])], [W])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]); m.ir_version = 8
onnx.save(m, "tiny.onnx")
seen, devs = set(), []
for d in hub.get_devices():
    cs = [a for a in d.attributes if a.startswith("chipset:")]
    key = cs[0] if cs else d.name
    if key in seen: continue
    seen.add(key); devs.append(d)
print(len(devs), "chipsets", flush=True)
jobs = []
for d in devs:
    try:
        cj, pj = hub.submit_compile_and_profile_jobs("tiny.onnx", d, name="chipmap", compile_options="--target_runtime qnn_dlc")
        jobs.append((d, cj, pj)); print("submitted", d.name, flush=True)
    except Exception as e:
        print("submit failed", d.name, str(e)[:120], flush=True)
res = []
for d, cj, pj in jobs:
    line = ""
    try:
        if pj is not None:
            pj.wait()
            paths = pj.download_job_logs("logs_" + pj.job_id)
            for p in paths:
                for l in open(p, errors="ignore"):
                    if "Detected Qualcomm SOC" in l or "fp16=" in l:
                        line = l.strip(); break
        st = pj.get_status() if pj is not None else cj.get_status()
        res.append({"device": d.name, "attrs": [a for a in d.attributes if a.startswith(("chipset:", "hexagon:"))],
                    "status": st.code, "detected": line[line.find("Detected"):] if line else "", "msg": (st.message or "")[:160]})
    except Exception as e:
        res.append({"device": d.name, "error": str(e)[:200]})
    print(json.dumps(res[-1]), flush=True)
json.dump(res, open("chipmap.json", "w"), indent=1)
print("done", flush=True)
