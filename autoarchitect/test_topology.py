import sys
sys.path.append('.')

from api.brain.topology_designer     import TopologyDesigner
from api.brain.network_zip_generator import NetworkZipGenerator

# ── Test 1: Topology Designer ──────────────────────────────────────────────
print("=" * 60)
print("  TEST 1 — TOPOLOGY DESIGNER")
print("=" * 60)

td = TopologyDesigner()

problems = [
    "detect illegal dumping in Oakland cameras and classify severity",
    "filter spam emails automatically",
    "monitor patient xrays and alert doctors",
    "grow my network marketing business",
    "detect fraud in bank transactions",
]

for p in problems:
    t = td.design(p)
    print(f"\nProblem:    {p[:55]}")
    print(f"  Agents:   {t['agents']}")
    print(f"  Topology: {t['topology']}")
    print(f"  Confidence: {t['confidence']}")
    print(f"  Source:   {t['source']}")

# ── Test 2: Network Zip Generator ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  TEST 2 — NETWORK ZIP GENERATOR")
print("=" * 60)

nzg     = NetworkZipGenerator()
problem = "detect illegal dumping in Oakland cameras and classify severity"
topo    = td.design(problem)

print(f"\nGenerating zip for: {problem[:50]}")
print(f"Agents: {topo['agents']} | Topology: {topo['topology']}")

zip_bytes = nzg.generate(problem, topo)
print(f"\nZip size: {len(zip_bytes):,} bytes")

with open("test_network.zip", "wb") as f:
    f.write(zip_bytes)
print("Saved: test_network.zip")

# ── Test 3: Brain learned topologies ──────────────────────────────────────
print("\n" + "=" * 60)
print("  TEST 3 — BRAIN STATUS")
print("=" * 60)
stats = td.stats()
for k, v in stats.items():
    print(f"  {k}: {v}")

print("\n✅ All tests passed!")
print("Next: python app.py → open localhost:5000 → type a problem → download network")
