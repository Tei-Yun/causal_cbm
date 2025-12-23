import numpy as np
import torch
from torch.utils.data import Dataset
import networkx as nx

# --- Data Generation Logic (Provided by User) ---
def generate_xor_data(size):
    x = np.random.uniform(0, 1, (size, 2))
    c = np.stack([x[:, 0] > 0.5, x[:, 1] > 0.5]).T
    y = np.logical_xor(c[:, 0], c[:, 1])
    return torch.FloatTensor(x), torch.FloatTensor(c), torch.FloatTensor(y)

def generate_trig_data(size):
    h = np.random.normal(0, 2, (size, 3))
    x, y, z = h[:, 0], h[:, 1], h[:, 2]
    input_features = np.stack([
        np.sin(x) + x, np.cos(x) + x, np.sin(y) + y, np.cos(y) + y,
        np.sin(z) + z, np.cos(z) + z, x ** 2 + y ** 2 + z ** 2,
    ]).T
    concepts = np.stack([x > 0, y > 0, z > 0]).T
    downstream_task = (x + y + z) > 1
    return torch.FloatTensor(input_features), torch.FloatTensor(concepts), torch.FloatTensor(downstream_task)

def generate_dot_data(size):
    emb_size = 2
    v1 = np.random.randn(size, emb_size) * 2
    v2 = np.ones(emb_size)
    v3 = np.random.randn(size, emb_size) * 2
    v4 = -np.ones(emb_size)
    x = np.hstack([v1+v3, v1-v3])
    c = np.stack([np.dot(v1, v2).ravel() > 0, np.dot(v3, v4).ravel() > 0]).T
    y = ((v1*v3).sum(axis=-1) > 0).astype(np.int64)
    return torch.FloatTensor(x), torch.FloatTensor(c), torch.Tensor(y)

# --- Wrapper for Project Compatibility ---
class DictTensorDataset(Dataset):
    def __init__(self, x, c, y):
        self.x = x
        self.c = c
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        # Preprocessing code expects a dictionary
        return {'x': self.x[idx], 'c': self.c[idx], 'y': self.y[idx]}
    
    def register_graph(self, graph):
        pass # Placeholder

class SimpleSyntheticDataset:
    def __init__(self, dataset_name, dataset_n_samples=10000, **kwargs):
        self.dataset_name = dataset_name.lower()
        self.n_samples = dataset_n_samples
        
        # 1. Generate Data
        if self.dataset_name == "xor":
            x, c, y = generate_xor_data(self.n_samples)
            self.concept_names = ['c1', 'c2']
            self.task_name = ['xor_y']
            # Graph: C1 -> Y, C2 -> Y
            self.true_graph = nx.DiGraph()
            self.true_graph.add_edges_from([('c1', 'xor_y'), ('c2', 'xor_y')])
            
        elif self.dataset_name in ["trig", "trigonometry"]:
            x, c, y = generate_trig_data(self.n_samples)
            self.concept_names = ['c1', 'c2', 'c3']
            self.task_name = ['trig_y']
            # Graph: C1->Y, C2->Y, C3->Y
            self.true_graph = nx.DiGraph()
            self.true_graph.add_edges_from([('c1', 'trig_y'), ('c2', 'trig_y'), ('c3', 'trig_y')])
            
        elif self.dataset_name in ["vector", "dot"]:
            x, c, y = generate_dot_data(self.n_samples)
            self.concept_names = ['dot_c1', 'dot_c2']
            self.task_name = ['dot_y']
            self.true_graph = nx.DiGraph()
            self.true_graph.add_edges_from([('dot_c1', 'dot_y'), ('dot_c2', 'dot_y')])
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # 2. Split Data (70/10/20)
        n_train = int(0.7 * self.n_samples)
        n_val = int(0.1 * self.n_samples)
        n_test = self.n_samples - n_train - n_val
        
        self.data = {}
        self.data['train'] = DictTensorDataset(x[:n_train], c[:n_train], y[:n_train])
        self.data['val'] = DictTensorDataset(x[n_train:n_train+n_val], c[n_train:n_train+n_val], y[n_train:n_train+n_val])
        self.data['test'] = DictTensorDataset(x[n_train+n_val:], c[n_train+n_val:], y[n_train+n_val:])

        # 3. Metadata (Required by pipeline)
        self.c_info = {
            'names': self.concept_names,
            'cardinality': [2] * len(self.concept_names)
        }
        # Some parts of code check c_info_complete
        self.c_info_complete = self.c_info 
        
        self.y_info = {
            'names': self.task_name,
            'cardinality': [2]
        }

    def load_ground_truth_graph(self):
        return self.true_graph