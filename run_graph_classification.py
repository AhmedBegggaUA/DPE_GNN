from torch_geometric.utils import degree
from attrdict import AttrDict
from torch_geometric.datasets import TUDataset
import warnings
warnings.filterwarnings("ignore")
from experiments.graph_classification import Experiment
import time
import torch
import numpy as np
import pandas as pd
from hyperparams import get_args_from_input
import random
import os
from ogb.graphproppred import PygGraphPropPredDataset
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
def log_to_file(message, filename="results/graph_classification.txt"):
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
    "layer_type": "GCN",
    "display": True,
    "num_trials": 25,
    "eval_every": 1,
    "patience": 100,
    "output_dim": 2,
    "dataset": None,
    "last_layer_fa": False,
    "positional_encoding": "pump",
    "k": 15,
    "dataset_name": "mutag"
})

hyperparams = {
    "mutag": AttrDict({"output_dim": 2}),
    "bzr": AttrDict({"output_dim": 2}),
    "cox2": AttrDict({"output_dim": 2}),
    "mutagenicity": AttrDict({"output_dim": 2}),
    "ptcfm": AttrDict({"output_dim": 2}),
    "ptcmm": AttrDict({"output_dim": 2}),
    "proteins": AttrDict({"output_dim": 2}),
    "enzymes": AttrDict({"output_dim": 6}),
    "msrc21": AttrDict({"output_dim": 20}),
    "imdb": AttrDict({"output_dim": 2}),
    "collab": AttrDict({"output_dim": 3}),
    "reddit": AttrDict({"output_dim": 2}),
    "ogbg-moltox21": AttrDict({
        "output_dim": 12,  # Multiple regression targets
        "num_features": 9  # OGB atcom features
    }),
    "ogbg-molpcba": AttrDict({
        "output_dim": 128,  # Multiple regression targets
        "num_features": 9  # OGB atom features
    })
}

results = []
args = default_args
args += get_args_from_input()
# Let's parse dataset_name
datasets = {}
if args.dataset_name == "mutag":
    mutag = list(TUDataset(root="data", name="MUTAG"))
    mutag = process_pe(mutag, args.positional_encoding, args.k)
    datasets = {"mutag": mutag}
elif args.dataset_name == "bzr":
    bzr = list(TUDataset(root="data", name="BZR"))
    bzr = process_pe(bzr, args.positional_encoding, args.k)
    datasets = {"bzr": bzr}
elif args.dataset_name == "cox2":
    cox2 = list(TUDataset(root="data", name="COX2"))
    cox2 = process_pe(cox2, args.positional_encoding, args.k)
    datasets = {"cox2": cox2}
elif args.dataset_name == "mutagenicity":
    mutagenicity = list(TUDataset(root="data", name="Mutagenicity"))
    mutagenicity = process_pe(mutagenicity, args.positional_encoding, args.k)
    datasets = {"mutagenicity": mutagenicity}
elif args.dataset_name == "ptcfm":
    ptcfm = list(TUDataset(root="data", name="PTC_FM"))
    ptcfm = process_pe(ptcfm, args.positional_encoding, args.k)
    datasets = {"ptcfm": ptcfm}
elif args.dataset_name == "ptcmm":
    ptcmm = list(TUDataset(root="data", name="PTC_MM"))
    ptcmm = process_pe(ptcmm, args.positional_encoding, args.k)
    datasets = {"ptcmm": ptcmm}
elif args.dataset_name == "proteins":
    proteins = list(TUDataset(root="data", name="PROTEINS"))
    proteins = process_pe(proteins, args.positional_encoding, args.k)
    datasets = {"proteins": proteins}
elif args.dataset_name == "enzymes":
    enzymes = list(TUDataset(root="data", name="ENZYMES"))
    enzymes = process_pe(enzymes, args.positional_encoding, args.k)
    datasets = {"enzymes": enzymes}
elif args.dataset_name == "msrc21":
    msrc21 = list(TUDataset(root="data", name="MSRC_21"))
    msrc21 = process_pe(msrc21, args.positional_encoding, args.k)
    datasets = {"msrc21": msrc21}
elif args.dataset_name == "imdb":
    imdb = list(TUDataset(root="data", name="IMDB-BINARY"))
    imdb = process_pe(imdb, args.positional_encoding, args.k)
    datasets = {"imdb": imdb}
elif args.dataset_name == "collab":
    collab = list(TUDataset(root="data", name="COLLAB"))
    collab = process_pe(collab, args.positional_encoding, args.k)
    datasets = {"collab": collab}
elif args.dataset_name == "reddit":
    reddit = list(TUDataset(root="data", name="REDDIT-BINARY"))
    reddit = process_pe(reddit, args.positional_encoding, args.k)
    datasets = {"reddit": reddit}
elif args.dataset_name in ["ogbg-moltox21", "ogbg-molpcba"]:
    ogbg = PygGraphPropPredDataset(name=args.dataset_name, root=f"data/{args.dataset_name}")
    # Las etiquetas de cada grafo están en one-hot encoding, hay que convertirlas a enteros
    for graph in ogbg:
        graph.y = torch.argmax(graph.y, dim=1)
    ogbg = process_pe(ogbg, args.positional_encoding, args.k)
    datasets = {args.dataset_name: list(ogbg)}
else:
    print("Dataset not found")
for key in datasets:
    if key in ["reddit", "imdb", "collab"]:
        for graph in datasets[key]:
            n = graph.num_nodes
            graph.x = torch.ones((n,1))
            graph.num_nodes = n
            graph.num_features = 1
if args.dataset:
    # restricts to just the given dataset if this mode is chosen
    name = args.dataset
    datasets = {name: datasets[name]}
name_of_the_dataset = None
for key in datasets:
    args += hyperparams[key]
    train_accuracies = []
    validation_accuracies = []
    test_accuracies = []
    dataset = datasets[key]
    name_of_the_dataset = key
    print('TRAINING STARTED...')
    start = time.time()
    for trial in range(args.num_trials):
        print(f"Trial {trial + 1}/{args.num_trials}")
        train_acc, validation_acc, test_acc = Experiment(args=args, dataset=dataset).run()
        train_accuracies.append(train_acc)
        validation_accuracies.append(validation_acc)
        test_accuracies.append(test_acc)
    end = time.time()
    run_duration = end - start

    train_mean = 100 * np.mean(train_accuracies)
    val_mean = 100 * np.mean(validation_accuracies)
    test_mean = 100 * np.mean(test_accuracies)
    train_ci = 2 * np.std(train_accuracies)/(args.num_trials ** 0.5)
    val_ci = 2 * np.std(validation_accuracies)/(args.num_trials ** 0.5)
    test_ci = 2 * np.std(test_accuracies)/(args.num_trials ** 0.5)*100
    log_to_file(f"RESULTS FOR {key} ({args.layer_type}), {args.positional_encoding} PE:\n")
    log_to_file(f"average acc: {test_mean}\n")
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
        "run_duration" : run_duration,
    })

    # Log every time a dataset is completed
    df = pd.DataFrame(results)
    with open(f'results/graph_classification_{args.layer_type}_{args.positional_encoding}.csv', 'a') as f:
        df.to_csv(f, mode='a', header=f.tell()==0)
