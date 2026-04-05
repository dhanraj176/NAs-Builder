"""
run_network.py — One-click launcher
AutoArchitect Agent Network

Usage:
    python run_network.py                  # watches input/ folder
    python run_network.py my_data/         # custom folder
    python run_network.py image.jpg        # single file
"""
import sys
from network import AgentNetwork

if __name__ == "__main__":
    net = AgentNetwork()

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        import os
        if os.path.isfile(arg):
            # Single file
            result = net.predict(arg)
            print("\nResult:", result)
        else:
            # Folder — run forever
            net.run(source=arg)
    else:
        net.run(source="input/")
