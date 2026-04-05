"""
fix_nzg.py — patches all self.agent_name references inside
the generated code strings in network_zip_generator.py
Run once: python fix_nzg.py
"""

path = "api/brain/network_zip_generator.py"

with open(path, "r", encoding="utf-8") as f:
    code = f.read()

# All the bad patterns and their fixes
replacements = [
    (
        'print(f"   ⚡ [{self.agent_name}] HIGH CONFIDENCE: {{label}} ({{conf:.0%}})")',
        'print("   ⚡ [" + self.agent_name + "] HIGH CONFIDENCE: " + str(label) + " (" + str(round(conf*100)) + "%)")'
    ),
    (
        'mem_file = Path(f"memory_{self.agent_name}.jsonl")',
        'mem_file = Path("memory_" + self.agent_name + ".jsonl")'
    ),
    (
        'print(f"   [{self.agent_name}] Need more examples to retrain "\n                  f"({{len(self.memory)}}/20)")',
        'print("   [" + self.agent_name + "] Need more examples — " + str(len(self.memory)) + "/20")'
    ),
    (
        'print(f"   [{self.agent_name}] Retraining on {{len(self.memory)}} examples...")',
        'print("   [" + self.agent_name + "] Retraining on " + str(len(self.memory)) + " examples...")'
    ),
    (
        'print(f"   [{self.agent_name}] ✅ Retrain complete")',
        'print("   [" + self.agent_name + "] ✅ Retrain complete")'
    ),
]

fixed = 0
for old, new in replacements:
    if old in code:
        code = code.replace(old, new)
        print(f"✅ Fixed: {old[:60]}...")
        fixed += 1
    else:
        print(f"⚠️  Not found (may already be fixed): {old[:60]}...")

with open(path, "w", encoding="utf-8") as f:
    f.write(code)

print(f"\n✅ Done — {fixed} replacements made")
print("Now run: python test_topology.py")
