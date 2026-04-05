"""
api_server.py — REST API for your agent network
Usage: python api_server.py
Then: POST http://localhost:8000/predict
"""
from flask import Flask, request, jsonify
from network import AgentNetwork

app = Flask(__name__)
net = AgentNetwork()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    inp  = data.get("input", "")
    if not inp:
        return jsonify({"error": "provide input field"}), 400
    result = net.predict(inp)
    return jsonify(result)

@app.route("/status")
def status():
    return jsonify(net.status())

@app.route("/")
def index():
    return jsonify({
        "name":    "AutoArchitect Agent Network",
        "problem": "detect illegal dumping in Oakland cameras and classify sever",
        "agents":  ['image', 'severity', 'report'],
        "endpoints": ["/predict (POST)", "/status (GET)"]
    })

if __name__ == "__main__":
    print("🚀 Agent Network API running on http://localhost:8000")
    app.run(port=8000, debug=False)
