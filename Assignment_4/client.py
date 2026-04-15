import torch
import torch.optim as optim
import requests
import io
from utils import SimpleModel

SERVER_URL = "http://127.0.0.1:5000"

def train_locally(model, data, target):
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    print(f"Local training loss: {loss.item()}")
    return model.state_dict()

def main():
    # 1. Download global model
    print("Fetching global model...")
    response = requests.get(f"{SERVER_URL}/get_model")
    model = SimpleModel()
    model.load_state_dict(torch.load(io.BytesIO(response.content)))

    # 2. Local Data (Mocking a single batch)
    local_data = torch.randn(5, 10)
    local_target = torch.randint(0, 2, (5,))

    # 3. Local Training
    updated_weights = train_locally(model, local_data, local_target)

    # 4. Upload updated weights
    buffer = io.BytesIO()
    torch.save(updated_weights, buffer)
    resp = requests.post(f"{SERVER_URL}/send_update", data=buffer.getvalue())
    print(resp.json())

if __name__ == "__main__":
    main()