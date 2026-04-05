# AutoArchitect Agent Network
**Problem:** detect illegal dumping in Oakland cameras and classify severity
**Generated:** 2026-03-26 21:14

## Quick Start (3 steps)

**Step 1 — Install**
```
pip install -r requirements.txt
```

**Step 2 — Run**
```
python run_network.py input/
```

**Step 3 — Drop files into input/ folder**
The network processes them automatically. Forever.

---

## Your Agent Network

**Topology:** `sequential`
**Pipeline:** `image → severity → report`

| Agent | What it does |
|-------|-------------|
| IMAGE | Detect and classify visual patterns in: detect ill |
| SEVERITY | Classify severity as HIGH / MEDIUM / LOW |
| REPORT | Generate report and send alerts automatically |

---

## Usage Options

### Autonomous (recommended)
```python
python run_network.py my_data_folder/
```
Watches folder every 10 seconds. Processes every new file.
Retrains itself every 50 examples. Gets smarter over time.

### Single prediction
```python
python run_network.py my_file.jpg
```

### REST API
```
python api_server.py
# POST http://localhost:8000/predict
# body: {"input": "path/to/file.jpg"}
```

### Python
```python
from network import AgentNetwork
net = AgentNetwork()
result = net.predict("my_file.jpg")
print(result)
```

---

## How It Gets Smarter
Every prediction is stored in `memory_*.jsonl`.
Every 50 predictions → agents retrain on YOUR data.
The more you run it → the more accurate it gets.

---
*Built with AutoArchitect AI — The ChatGPT for AI Agents*
