from flask import Flask, request, jsonify
import torch
import io
from utils import SimpleModel, federated_averaging

app = Flask(__name__)

# Global state
global_model = SimpleModel()
collected_updates = []
MIN_CLIENTS = 2  # Wait for 2 clients before updating global model

@app.route('/get_model', methods=['GET'])
def get_model():
    """Clients call this to get the current global weights."""
    buffer = io.BytesIO()
    torch.save(global_model.state_dict(), buffer)
    return buffer.getvalue()

@app.route('/send_update', methods=['POST'])
def receive_update():
    """Clients upload their local weights here."""
    global collected_updates, global_model
    
    # Load weights from the binary data sent
    client_weights_bytes = request.data
    client_weights = torch.load(io.BytesIO(client_weights_bytes))
    
    collected_updates.append(client_weights)
    print(f"Received update. Total: {len(collected_updates)}/{MIN_CLIENTS}")

    if len(collected_updates) >= MIN_CLIENTS:
        print("Averaging weights...")
        new_weights = federated_averaging(collected_updates)
        global_model.load_state_dict(new_weights)
        collected_updates = [] # Clear for next round
        return jsonify({"status": "Global model updated"}), 200

    return jsonify({"status": "Update received, waiting for more clients"}), 200

if __name__ == '__main__':
    app.run(port=5000)