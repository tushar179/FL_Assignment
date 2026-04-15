import torch
import torch.nn as nn

# Simple Model Architecture
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2) # Example: 10 input features, 2 classes

    forward = lambda self, x: self.fc(x)

def federated_averaging(weights_list):
    """
    Averages a list of state_dicts (model parameters).
    """
    avg_weights = {}
    num_clients = len(weights_list)
    
    for key in weights_list[0].keys():
        # Sum the same parameter across all clients
        avg_weights[key] = sum([w[key] for w in weights_list]) / num_clients
        
    return avg_weights