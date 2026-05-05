"""
tests/verify_zip.py - Final ZIP verification before deployment.

Generates a fresh pothole ZIP, extracts to isolated temp dir,
runs all checks, prints PASS/FAIL for each.
"""
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import zipfile
import zlib

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from api.brain.network_zip_generator import NetworkZipGenerator

SEP = "=" * 60


def make_png(w=64, h=64):
    def chunk(name, data):
        c = name + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes([i % 256, (i * 2) % 256, 128]) * w for i in range(h))
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw))
    png += chunk(b"IEND", b"")
    return png


def parse_json_from_output(text):
    for i, line in enumerate(text.splitlines()):
        if line.strip().startswith("{"):
            try:
                return json.loads("\n".join(text.splitlines()[i:]))
            except Exception:
                pass
    return None


results = {}


def check(name, passed, detail=""):
    verdict = "PASS" if passed else f"FAIL{(' — ' + detail) if detail else ''}"
    results[name] = verdict
    return passed


# ── 1. Generate ZIP ───────────────────────────────────────────────────────────
print(SEP)
print("Generating fresh pothole ZIP...")
print(SEP)

gen = NetworkZipGenerator()
zip_bytes = gen.generate(
    problem="detect potholes in road surface",
    topology={"agents": ["image"], "topology": "sequential", "connections": []},
)
print(f"ZIP size: {len(zip_bytes):,} bytes\n")

# ── 2. Extract to isolated temp dir ──────────────────────────────────────────
tmp = tempfile.mkdtemp(prefix="final_verify_")
with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
    names = sorted(zf.namelist())
    zf.extractall(tmp)

print(f"Extracted to: {tmp}")
print("Contents:")
for name in names:
    size = os.path.getsize(os.path.join(tmp, name))
    print(f"  {name:<55} {size:>12,} bytes")

# Write test image
img_path = os.path.join(tmp, "road.png")
with open(img_path, "wb") as f:
    f.write(make_png())

env = os.environ.copy()
env["PYTHONPATH"] = ""

# ── Check 1: predict.py --image ───────────────────────────────────────────────
print(f"\n{SEP}")
print("CHECK 1  python predict.py --image road.png")
print(SEP)

r1 = subprocess.run(
    [sys.executable, "predict.py", "--image", "road.png"],
    cwd=tmp, capture_output=True, text=True, env=env, timeout=120,
)
print(r1.stdout.strip())
if r1.stderr.strip():
    print("[stderr]", r1.stderr.strip())

if r1.returncode != 0:
    check("1_image_predict", False, f"exit {r1.returncode}")
else:
    out = parse_json_from_output(r1.stdout)
    if out and "label" in out and "confidence" in out:
        print(f"\n  label      : {out['label']}")
        print(f"  confidence : {out['confidence']}")
        print(f"  action     : {out.get('action', '?')}")
        check("1_image_predict", True)
    else:
        check("1_image_predict", False, "no valid JSON in output")

# ── Check 2: predict.py --text (wrong domain — should not crash) ──────────────
print(f"\n{SEP}")
print('CHECK 2  python predict.py --text "there is a big hole in the road"')
print(SEP)

r2 = subprocess.run(
    [sys.executable, "predict.py", "--text", "there is a big hole in the road"],
    cwd=tmp, capture_output=True, text=True, env=env, timeout=120,
)
print(r2.stdout.strip())
if r2.stderr.strip():
    print("[stderr]", r2.stderr.strip())

if r2.returncode != 0:
    check("2_text_on_image_agent", False, f"exit {r2.returncode}")
else:
    out2 = parse_json_from_output(r2.stdout)
    if out2 and "label" in out2:
        print(f"\n  label      : {out2['label']}")
        print(f"  confidence : {out2.get('confidence', '?')}")
        check("2_text_on_image_agent", True, "ran without crash, returned JSON")
    else:
        check("2_text_on_image_agent", r2.returncode == 0, "ran without crash")

# ── Check 3: README.md ────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("CHECK 3  README.md")
print(SEP)

readme_path = os.path.join(tmp, "README.md")
if not os.path.exists(readme_path):
    check("3_readme", False, "file missing")
else:
    readme = open(readme_path, encoding="utf-8").read()
    print(readme)
    reqs = {
        "Install section":         "pip install" in readme,
        "predict.py mentioned":    "predict.py" in readme,
        "--image usage":           "--image" in readme,
        "retrain.py mentioned":    "retrain.py" in readme,
        "problem description":     "pothole" in readme.lower(),
    }
    print("\nChecklist:")
    all_ok = True
    for label, ok in reqs.items():
        print(f"  [{'OK' if ok else '!!'}] {label}")
        if not ok:
            all_ok = False
    check("3_readme", all_ok, "" if all_ok else str([k for k, v in reqs.items() if not v]))

# ── Check 4: requirements.txt ─────────────────────────────────────────────────
print(f"\n{SEP}")
print("CHECK 4  requirements.txt")
print(SEP)

req_path = os.path.join(tmp, "requirements.txt")
if not os.path.exists(req_path):
    check("4_requirements", False, "file missing")
else:
    req = open(req_path, encoding="utf-8").read()
    print(req)
    pkgs = ["torch", "torchvision", "pillow", "numpy", "requests", "flask"]
    req_lower = req.lower()
    all_ok2 = True
    print("Package check:")
    for pkg in pkgs:
        ok = pkg in req_lower
        print(f"  [{'OK' if ok else '!!'}] {pkg}")
        if not ok:
            all_ok2 = False
    check("4_requirements", all_ok2, "" if all_ok2 else str([p for p in pkgs if p not in req_lower]))

# ── Check 5: model actually loaded (not random weights) ───────────────────────
print(f"\n{SEP}")
print("CHECK 5  model weights loaded (not random)")
print(SEP)

model_pth = os.path.join(tmp, "models", "potholes_road_agent_model.pth")
model_exists = os.path.exists(model_pth)
model_size   = os.path.getsize(model_pth) if model_exists else 0
print(f"  models/potholes_road_agent_model.pth: {'exists' if model_exists else 'MISSING'}")
print(f"  size: {model_size:,} bytes")

agent_py = os.path.join(tmp, "agents", "potholes_road_agent.py")
if os.path.exists(agent_py):
    code = open(agent_py, encoding="utf-8").read()
    for line in code.splitlines():
        if "MODEL_PATH" in line and "model.pth" in line:
            print(f"  MODEL_PATH line: {line.strip()}")
    path_ok = "potholes_road_agent_model.pth" in code
    size_ok = model_size > 1_000_000
    check("5_model_weights", path_ok and size_ok,
          "" if (path_ok and size_ok) else f"path_ok={path_ok} size_ok={size_ok}")
else:
    check("5_model_weights", False, "agent .py missing")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("FINAL SUMMARY")
print(SEP)

all_pass = True
for name, verdict in results.items():
    icon = "PASS" if verdict.startswith("PASS") else "FAIL"
    if icon == "FAIL":
        all_pass = False
    print(f"  {icon}  {name.split('_', 1)[1].replace('_', ' '):<35} {verdict}")

print()
overall = "PASS -- ready for deployment" if all_pass else "FAIL -- see checks above"
print(f"  OVERALL: {overall}")
print(SEP)

sys.exit(0 if all_pass else 1)
