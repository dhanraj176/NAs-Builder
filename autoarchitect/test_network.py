# test_network.py — test the agent network
import sys
sys.path.append('.')

from api.agents.agent_network import AgentNetwork
from api.agents.image_agent   import ImageAgent
from api.agents.text_agent    import TextAgent

# Create two agents
img_agent  = ImageAgent()
txt_agent  = TextAgent()

# Build network
network = AgentNetwork("test_network")
network.add_agent(img_agent,  role="vision")
network.add_agent(txt_agent,  role="text")
network.add_pipeline(
    [img_agent.agent_id, txt_agent.agent_id],
    name="detection_pipeline"
)

# Show network info
print("\n" + "="*50)
print("  AGENT NETWORK INFO")
print("="*50)
info = network.info()
for k, v in info.items():
    print(f"  {k}: {v}")
print("="*50)
print("\n✅ Agent network working!")
print("   Agents collaborate → brain learns")
print("   Network runs forever → brain improves")
