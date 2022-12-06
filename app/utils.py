import os
import torch
import json


def load_model(path, device='cpu'):
    from net import Net, GAT
    with open(os.path.join(path, 'config.json')) as f:
        config = json.load(f)
    net = GAT(**config)
    net.load_state_dict(torch.load(os.path.join(path, 'model.pth'), map_location=device))
    net.eval()
    return net
