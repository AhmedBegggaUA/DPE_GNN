import torch
import numpy as np
from measure_smoothing import dirichlet_normalized
from attrdict import AttrDict
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from math import inf
from sklearn.metrics import mean_absolute_error, r2_score
from models.graph_model import GNN

default_args = AttrDict({
    "learning_rate": 1e-3,
    "max_epochs": 1000000,
    "display": True,
    "device": None,
    "eval_every": 1,
    "stopping_criterion": "validation",
    "stopping_threshold": 1.01,
    "patience": 20,
    "train_fraction": 0.8,
    "validation_fraction": 0.1,
    "test_fraction": 0.1,
    "dropout": 0.0,
    "weight_decay": 1e-5,
    "input_dim": None,
    "hidden_dim": 32,
    "output_dim": 1,
    "hidden_layers": None,
    "num_layers": 1,
    "batch_size": 64,
    "layer_type": "R-GCN",
    "num_relations": 2,
    "last_layer_fa": False
})

class Experiment:
    def __init__(self, args=None, dataset=None, train_dataset=None, validation_dataset=None, test_dataset=None):
        self.args = default_args + args
        self.dataset = dataset
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.loss_fn = torch.nn.L1Loss()  # Cambiado a L1Loss (MAE) para regresión

        if self.args.device is None:
            self.args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.args.hidden_layers is None:
            self.args.hidden_layers = [self.args.hidden_dim] * self.args.num_layers
        if self.args.input_dim is None:
            self.args.input_dim = self.dataset[0].x.shape[1]
            
        # Manejo de tipos de aristas
        for graph in self.dataset:
            if not "edge_type" in graph.keys():
                num_edges = graph.edge_index.shape[1]
                graph.edge_type = torch.zeros(num_edges, dtype=int)
                
        if self.args.num_relations is None:
            if self.args.rewiring == "None":
                self.args.num_relations = 1
            else:
                self.args.num_relations = 2

        # División del dataset si no se proporciona
        if self.test_dataset is None:
            dataset_size = len(self.dataset)
            train_size = int(self.args.train_fraction * dataset_size)
            validation_size = int(self.args.validation_fraction * dataset_size)
            test_size = dataset_size - train_size - validation_size
            self.train_dataset, self.validation_dataset, self.test_dataset = random_split(
                self.dataset, [train_size, validation_size, test_size]
            )
        elif self.validation_dataset is None:
            train_size = int(self.args.train_fraction * len(self.train_dataset))
            validation_size = len(self.train_dataset) - train_size
            self.train_dataset, self.validation_dataset = random_split(
                self.train_dataset, [train_size, validation_size]
            )

    def run(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
        validation_loader = DataLoader(self.validation_dataset, batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=True)

        
        self.model = GNN(self.args).to(self.args.device)
        
        print(self.model)
        print("Number of parameters: ", sum(p.numel() for p in self.model.parameters()))
        
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        scheduler = ReduceLROnPlateau(optimizer, mode='min')  # Para minimizar MAE

        best_validation_mae = float('inf')
        best_train_mae = float('inf')
        best_test_mae = float('inf')
        train_goal = float('inf')
        validation_goal = float('inf')
        epochs_no_improve = 0

        for epoch in range(1, 1 + self.args.max_epochs):
            self.model.train()
            total_loss = 0
            optimizer.zero_grad()

            for graph in train_loader:
                graph = graph.to(self.args.device)
                y = graph.y.to(self.args.device)
                if len(y.shape) == 1:
                    y = y.unsqueeze(1)  # Asegurar dimensión correcta para regresión

                out, extra_loss = self.model(graph)
                loss = self.loss_fn(out, y.float()) + extra_loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            new_best_str = ''
            scheduler.step(total_loss)

            if epoch % self.args.eval_every == 0:
                train_mae = self.eval(loader=train_loader)
                validation_mae = self.eval(loader=validation_loader)
                test_mae = self.eval(loader=test_loader)

                if self.args.stopping_criterion == "train":
                    if train_mae < train_goal:
                        best_train_mae = train_mae
                        best_validation_mae = validation_mae
                        best_test_mae = test_mae
                        epochs_no_improve = 0
                        train_goal = train_mae / self.args.stopping_threshold
                        new_best_str = ' (new best train)'
                    elif train_mae < best_train_mae:
                        best_train_mae = train_mae
                        best_validation_mae = validation_mae
                        best_test_mae = test_mae
                        epochs_no_improve += 1
                    else:
                        epochs_no_improve += 1
                elif self.args.stopping_criterion == 'validation':
                    if validation_mae < validation_goal:
                        best_train_mae = train_mae
                        best_validation_mae = validation_mae
                        best_test_mae = min(test_mae, best_test_mae)
                        epochs_no_improve = 0
                        validation_goal = validation_mae / self.args.stopping_threshold
                        new_best_str = ' (new best validation)'
                    elif validation_mae < best_validation_mae:
                        best_train_mae = train_mae
                        best_validation_mae = validation_mae
                        best_test_mae = min(test_mae, best_test_mae)
                        epochs_no_improve += 1
                    else:
                        epochs_no_improve += 1

                if self.args.display:
                    print(f'Epoch {epoch}, Train MAE: {train_mae:.6f}, '
                          f'Validation MAE: {validation_mae:.6f}{new_best_str}, '
                          f'Test MAE: {test_mae:.6f}')

                if epochs_no_improve > self.args.patience:
                    if self.args.display:
                        print(f'{self.args.patience} epochs without improvement, stopping training')
                        print(f'Best train MAE: {best_train_mae:.6f}, '
                              f'Best validation MAE: {best_validation_mae:.6f}, '
                              f'Best test MAE: {best_test_mae:.6f}')
                    return best_train_mae, best_validation_mae, best_test_mae

        if self.args.display:
            print('Reached max epoch count, stopping training')
            print(f'Best train MAE: {best_train_mae:.6f}, '
                  f'Best validation MAE: {best_validation_mae:.6f}, '
                  f'Best test MAE: {best_test_mae:.6f}')
        return best_train_mae, best_validation_mae, best_test_mae

    def eval(self, loader):
        self.model.eval()
        total_mae = 0
        num_samples = 0
        
        with torch.no_grad():
            for graph in loader:
                graph = graph.to(self.args.device)
                y = graph.y.to(self.args.device)
                if len(y.shape) == 1:
                    y = y.unsqueeze(1)
                
                out, _ = self.model(graph)
                mae = self.loss_fn(out, y.float()).item()
                total_mae += mae * graph.num_graphs
                num_samples += graph.num_graphs
                
        return total_mae / num_samples