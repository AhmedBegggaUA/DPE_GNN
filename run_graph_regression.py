from attrdict import AttrDict
from torch_geometric.datasets import ZINC
from ogb.graphproppred import PygGraphPropPredDataset
import warnings
from experiments.graph_regression import Experiment
import time
import torch
import numpy as np
import pandas as pd
from hyperparams import get_args_from_input
import random
import os
from torch_geometric.utils import degree
warnings.filterwarnings("ignore")
seed = 42
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def log_to_file(message, filename="results/graph_regression.txt"):
    print(message)
    file = open(filename, "a")
    file.write(message)
    file.close()
def process_pe(dataset, pe, k):
    if pe == "RWPE":
        for graph in dataset:
            graph.pe = RWPE(graph.edge_index, k)
    elif pe == "LPE":
        for graph in dataset:
            graph.pe = get_k_smallest_eigenvectors(graph.edge_index, k)
    elif pe == "pump":
        # pe is the node degree
        for graph in dataset:
            graph.pe = degree(graph.edge_index[0]).unsqueeze(1)
    elif pe == "None":
        for graph in dataset:
            graph.pe = torch.zeros((graph.num_nodes, k))
    return dataset
   
def get_k_smallest_eigenvectors(adj_matrix: torch.Tensor, k: int) -> torch.Tensor:
    """
    Computa los k autovectores más pequeños de la matriz normalizada.
    
    Args:
        adj_matrix (torch.Tensor): Matriz de adyacencia
        k (int): Número de autovectores a retornar
        
    Returns:
        torch.Tensor: Matriz con los k autovectores más pequeños como columnas
    """
    # Aseguramos que la matriz sea simétrica
    adj_matrix = adj_matrix
    
    # Calculamos el grado de cada nodo
    degrees = torch.sum(adj_matrix, dim=1)
    
    # Creamos la matriz D^(-1/2)
    D_inv_sqrt = torch.diag(torch.pow(degrees, -0.5))
    
    # Normalizamos la matriz: D^(-1/2) A D^(-1/2)
    normalized_matrix = torch.mm(torch.mm(D_inv_sqrt, adj_matrix), D_inv_sqrt)
    
    # Calculamos los autovalores y autovectores usando numpy
    import numpy as np
    # Nos aseguramos de que cojamos la componente conectada más grande
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(normalized_matrix.cpu().numpy())
        # Pasamos a torch
        eigenvalues = torch.Tensor(eigenvalues).to(adj_matrix.device)
        eigenvectors = torch.Tensor(eigenvectors).to(adj_matrix.device)
        #eigenvalues, eigenvectors = torch.linalg.eigh(normalized_matrix)
    except:
        print("Error en la matriz")
        eigenvectors = torch.zeros(adj_matrix.shape[0], k, device=adj_matrix.device)
    
    # Seleccionamos los k autovectores correspondientes a los k autovalores más pequeños
    # eigenvalues ya viene ordenado de menor a mayor en PyTorch
    k_smallest_eigenvectors = eigenvectors[:, :k]
    
    return k_smallest_eigenvectors

def RWPE(g, pos_enc_dim):
    """
    Initializes positional encoding for an adjacency matrix using Random Walk Positional Encoding (RWPE)
    
    Args:
        g: torch.Tensor - Adjacency matrix of shape (n x n)
        pos_enc_dim: int - Dimension of positional encoding
        type_init: str - Type of initialization (currently supports 'rand_walk')
    
    Returns:
        torch.Tensor: Positional encoding matrix of shape (n_nodes, pos_enc_dim)
    """
    
    n = g.size(0)  # Number of nodes
    
    # Calculate D^-1 (inverse degree matrix)
    degrees = g.sum(dim=1)  # Sum rows to get degrees
    degrees = degrees.clamp(min=1.0)  # Avoid division by zero
    d_inv = torch.diag(1.0 / degrees)
    
    # Calculate random walk matrix: RW = A * D^-1
    RW = torch.mm(g, d_inv)
    M = RW
    
    # Initialize list to store positional encodings
    PE = [torch.diag(M)]  # Get diagonal of M
    M_power = M
    
    # Compute powers of random walk matrix
    for _ in range(pos_enc_dim-1):
        M_power = torch.mm(M_power, M)
        PE.append(torch.diag(M_power))
    
    # Stack all positional encodings
    PE = torch.stack(PE, dim=-1)
    
    return PE

default_args = AttrDict({
    "dropout": 0,
    "num_layers": 4,
    "hidden_dim": 64,
    "learning_rate": 1e-3,
    "layer_type": "GIN",
    "display": True,
    "num_trials": 25,
    "eval_every": 1,
    "patience": 100,
    "output_dim": 1,  # Changed to 1 for regression
    "dataset": None,
    "last_layer_fa": False,
    "positional_encoding": "pump",
    "k": 10,
    "dataset_name": "zinc"
})

hyperparams = {
    "zinc": AttrDict({
        "output_dim": 1,
        "num_features": 21  # ZINC atom features
    })
}

def load_dataset(dataset_name):
    if dataset_name == "zinc":
        zinc = list(ZINC(root="data/ZINC"))
        return process_pe(zinc, "pump", 10)
        #return list(ZINC(root="data/ZINC", subset=True))
    elif dataset_name in ["ogbg-moltox21", "ogbg-molpcba"]:
        dataset = PygGraphPropPredDataset(name=dataset_name, root=f"data/{dataset_name}")
        return list(dataset)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

results = []
args = default_args
args += get_args_from_input()

# Load dataset
datasets = {}
try:
    dataset = load_dataset(args.dataset_name)
    datasets = {args.dataset_name: dataset}
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

if args.dataset:
    # restricts to just the given dataset if this mode is chosen
    name = args.dataset
    datasets = {name: datasets[name]}

name_of_the_dataset = None
for key in datasets:
    args += hyperparams[key]
    train_mses = []
    validation_mses = []
    test_mses = []
    dataset = datasets[key]
    name_of_the_dataset = key
    
    print('TRAINING STARTED...')
    start = time.time()
    
    for trial in range(args.num_trials):
        print(f"Trial {trial + 1}/{args.num_trials}")
        train_mse, validation_mse, test_mse = Experiment(args=args, dataset=dataset).run()
        train_mses.append(train_mse)
        validation_mses.append(validation_mse)
        test_mses.append(test_mse)
    
    end = time.time()
    run_duration = end - start

    # Calculate statistics
    train_mean = np.mean(train_mses)
    val_mean = np.mean(validation_mses)
    test_mean = np.mean(test_mses)
    train_ci = 2 * np.std(train_mses)/(args.num_trials ** 0.5)
    val_ci = 2 * np.std(validation_mses)/(args.num_trials ** 0.5)
    test_ci = 2 * np.std(test_mses)/(args.num_trials ** 0.5)
    
    log_to_file(f"RESULTS FOR {key} ({args.layer_type}), {args.positional_encoding} PE:\n")
    log_to_file(f"average MSE: {test_mean}\n")
    log_to_file(f"plus/minus:  {test_ci}\n\n")
    
    results.append({
        "dataset": key,
        "layer_type": args.layer_type,
        "positional_encoding": args.positional_encoding,
        "test_mean": test_mean,
        "test_ci": test_ci,
        "val_mean": val_mean,
        "val_ci": val_ci,
        "train_mean": train_mean,
        "train_ci": train_ci,
        "last_layer_fa": args.last_layer_fa,
        "run_duration": run_duration,
    })

    # Log every time a dataset is completed
    df = pd.DataFrame(results)
    with open(f'results/graph_regression_{args.layer_type}_{args.positional_encoding}.csv', 'a') as f:
        df.to_csv(f, mode='a', header=f.tell()==0)