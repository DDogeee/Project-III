import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GATv2Conv, SAGEConv


class Net(torch.nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        super().__init__()
        self.conv1 = GraphConv(num_inputs,
                               num_hiddens,
                               aggr='max')
        self.conv2 = GraphConv(num_hiddens,
                               num_hiddens,
                               aggr='max')

        self.lin = torch.nn.Linear(num_hiddens, num_outputs)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = torch.relu(self.conv1(x, edge_index, edge_attr))
        x = torch.relu(self.conv2(x, edge_index, edge_attr))
        x = self.lin(x)
        return x

class GAT(torch.nn.Module):
  """Graph Attention Network"""
  def __init__(self, num_inputs, num_hiddens, num_outputs, heads=8):
    super().__init__()
    self.gat1 = GATv2Conv(num_inputs, num_hiddens, heads=heads)

    self.gat2 = GATv2Conv(num_hiddens*heads, num_outputs, heads=heads)

    self.lin = torch.nn.Linear(num_outputs*heads, num_outputs)

  def forward(self, data):
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    h = self.gat1(x, edge_index)
    h = F.elu(h)
    h = F.dropout(h, p=0.6, training=self.training)
    h = self.gat2(h, edge_index)
    h = F.elu(h)
    h = self.lin(h)
    return h
  

class Hybrid(torch.nn.Module):
  def __init__(self, num_inputs, num_hiddens, num_outputs, heads=8):
    super().__init__()
    self.conv1 = GraphConv(num_inputs,
                              num_hiddens,
                              aggr='max')
    self.conv2 = GraphConv(num_hiddens,
                              num_hiddens,
                              aggr='max')
    
    self.gat1 = GATv2Conv(num_hiddens, num_hiddens, heads=heads)

    self.gat2 = GATv2Conv(num_hiddens*heads, num_outputs, heads=heads)

    self.lin = torch.nn.Linear(num_outputs*heads, num_outputs)

  def forward(self, data):
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    h = torch.relu(self.conv1(x, edge_index, edge_attr))
    h = torch.relu(self.conv2(h, edge_index, edge_attr))
    h = self.gat1(h, edge_index)
    h = F.elu(h)
    h = F.dropout(h, p=0.6, training=self.training)
    h = self.gat2(h, edge_index)
    h = F.elu(h)
    h = self.lin(h)
    return h

