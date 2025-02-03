import torch
import torch.nn as nn
from measure_smoothing import dirichlet_normalized
from torch.nn import ModuleList, Dropout, ReLU
from torch_geometric.nn import GCNConv, RGCNConv, SAGEConv, GATConv, GatedGraphConv, GINConv, FiLMConv, global_mean_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch, unbatch_edge_index, unbatch
from torch_scatter import scatter_add
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import os
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import DataLoader
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add
class RGATConv(torch.nn.Module):
    def __init__(self, in_features, out_features, num_relations):
        super(RGATConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.self_loop_conv = torch.nn.Linear(in_features, out_features)
        convs = []
        for i in range(self.num_relations):
            convs.append(GATConv(in_features, out_features))
        self.convs = ModuleList(convs)
    def forward(self, x, edge_index, edge_type):
        x_new = self.self_loop_conv(x)
        for i, conv in enumerate(self.convs):
            rel_edge_index = edge_index[:, edge_type==i]
            x_new += conv(x, rel_edge_index)
        return x_new

class RGINConv(torch.nn.Module):
    def __init__(self, in_features, out_features, num_relations):
        super(RGINConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.self_loop_conv = torch.nn.Linear(in_features, out_features)
        convs = []
        for i in range(self.num_relations):
            convs.append(GINConv(nn.Sequential(nn.Linear(in_features, out_features),nn.BatchNorm1d(out_features), nn.ReLU(),nn.Linear(out_features, out_features))))
        self.convs = ModuleList(convs)
    def forward(self, x, edge_index, edge_type):
        x_new = self.self_loop_conv(x)
        for i, conv in enumerate(self.convs):
            rel_edge_index = edge_index[:, edge_type==i]
            x_new += conv(x, rel_edge_index)
        return x_new


class GatedGCNLayer(MessagePassing):
    def __init__(self, input_dim, output_dim, dropout=0.5, batch_norm=True, 
                 residual=False, graph_norm=True):
        super().__init__(aggr='add')
        
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.graph_norm = graph_norm
        self.residual = residual
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)

    def forward(self, x, edge_index, edge_attr=None, batch=None, snorm_n=None):
        """
        Solo devuelve h_out
        """
        # x, edge_index = data.x, data.edge_index
        # edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        # batch = data.batch if hasattr(data, 'batch') else None
        # snorm_n = data.snorm_n if hasattr(data, 'snorm_n') else None

        h_in = x

        # Transformaciones de nodos
        Ah = self.A(x)
        Bh = self.B(x)
        Dh = self.D(x)
        Eh = self.E(x)

        # Procesar características de aristas
        if edge_attr is None:
            edge_attr = torch.ones((edge_index.size(1), self.in_channels), 
                                 device=edge_index.device)
        
        Ce = self.C(edge_attr)

        # Calcular características de aristas
        row, col = edge_index
        e_out = Dh[row] + Eh[col] + Ce
        sigma = torch.sigmoid(e_out)

        # Propagar mensajes
        h_out = Ah + self.propagate(edge_index, 
                                  size=(x.size(0), x.size(0)),
                                  sigma=sigma,
                                  Bh=Bh)

        # Normalización del grafo
        if self.graph_norm and snorm_n is not None:
            h_out = h_out * snorm_n.view(-1, 1)

        # Batch normalization para nodos
        if self.batch_norm:
            h_out = self.bn_node_h(h_out)

        # Activación no lineal
        h_out = F.relu(h_out)

        # Conexión residual
        if self.residual:
            h_out = h_in + h_out

        # Dropout
        h_out = F.dropout(h_out, self.dropout, training=self.training)

        return h_out

    def message(self, Bh_j, sigma):
        return sigma * Bh_j

    def aggregate(self, inputs, index, dim_size=None):
        # Agregación con normalización
        out = scatter_add(inputs, index, dim=self.node_dim, dim_size=dim_size)
        
        # Calcular denominador
        ones = torch.ones_like(inputs[..., :1])
        sigma_sum = scatter_add(ones, index, dim=self.node_dim, dim_size=dim_size)
        
        # Normalizar
        out = out / (sigma_sum + 1e-6)
        
        return out

    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels)
class GNN(torch.nn.Module):
    def __init__(self, args):
        super(GNN, self).__init__()
        self.pe = args.positional_encoding
        self.args = args
        self.k = args.k
        self.num_relations = args.num_relations
        self.layer_type = args.layer_type
        num_features = [args.input_dim] + list(args.hidden_layers) + [args.output_dim]
        self.num_layers = len(num_features) - 1
        layers = []
        for i, (in_features, out_features) in enumerate(zip(num_features[:-1], num_features[1:])):
            if i == 0 and (self.pe != "None"):
                print("USING POSITIONAL ENCODING: ", self.pe)
                layers.append(self.get_layer(in_features + self.k, out_features))
            else:
                layers.append(self.get_layer(in_features, out_features))
        self.layers = ModuleList(layers)
        self.dropout = Dropout(p=args.dropout)
        self.act_fn = ReLU()
        self.gnn_pump = GCNConv(1, 64)
        self.gnn_pump2 = GCNConv(64, self.k)
        self.MLP_Eigen = nn.Linear(self.k, self.k)
        if self.args.last_layer_fa:
            if self.layer_type == "R-GCN" or self.layer_type == "GCN":
                self.last_layer_transform = torch.nn.Linear(self.args.hidden_dim, self.args.output_dim)
            elif self.layer_type == "R-GIN" or self.layer_type == "GIN":
                self.last_layer_transform = nn.Sequential(nn.Linear(self.args.hidden_dim, self.args.hidden_dim),nn.BatchNorm1d(self.args.hidden_dim), nn.ReLU(),nn.Linear(self.args.hidden_dim, self.args.output_dim))
            else:
                raise NotImplementedError

    def get_layer(self, in_features, out_features):
        if self.layer_type == "GCN":
            return GCNConv(in_features, out_features)
        elif self.layer_type == "R-GCN":
            return RGCNConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "R-GAT":
            return RGATConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "R-GIN":
            return RGINConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "GIN":
            return GINConv(nn.Sequential(nn.Linear(in_features, out_features),nn.BatchNorm1d(out_features), nn.ReLU(),nn.Linear(out_features, out_features)))
        elif self.layer_type == "SAGE":
            return SAGEConv(in_features, out_features)
        elif self.layer_type == "FiLM":
            return FiLMConv(in_features, out_features)
        elif self.layer_type == "GatedGCN":
            return GatedGCNLayer(in_features, out_features)

    def forward(self, graph, measure_dirichlet=False):
        x, edge_index,pe, ptr, batch = graph.x, graph.edge_index,graph.pe, graph.ptr, graph.batch
        x = x.float()
        for i, layer in enumerate(self.layers):
            if self.layer_type in ["R-GCN", "R-GAT", "R-GIN", "FiLM"]:
                x_new = layer(x, edge_index, edge_type=graph.edge_type)
            if i == 0:
                if self.pe == "pump":
                    # Convert to dense batch efficiently
                    x_dense, mask = to_dense_batch(x, batch)
                    adj_matrices = to_dense_adj(edge_index, batch)
                    batch_size = adj_matrices.size(0)
                    x_final = []
                    total_pump_loss = 0
                    total_ortho_loss = 0
                    for b in range(batch_size):
                        adj = adj_matrices[b]
                        degree_matrix = adj.sum(dim=-1).unsqueeze(-1)
                        s = self.gnn_pump(degree_matrix, adj.nonzero().t())
                        s = F.relu(s)
                        s = self.gnn_pump2(s, adj.nonzero().t())
                        s, pump_loss, ortho_loss = pump(adj, s)
                        s = s.squeeze(0)
                        # concatenamos con x
                        num_nodes = mask[b].sum()
                        s = s[:num_nodes]
                        x_i = x_dense[b, :num_nodes]
                        x_final.append(torch.cat([x_i, s], dim=1))
                        # Average the losses
                        total_pump_loss += pump_loss#total_pump_loss / batch_size
                        total_ortho_loss += ortho_loss#total_ortho_loss / batch_size
                    total_pump_loss = total_pump_loss / batch_size
                    total_ortho_loss = total_ortho_loss / batch_size
                    x = torch.cat(x_final, dim=0)

                    
                    # # Calculate degree matrices for GNN
                    # degree_matrices = adj_matrices.sum(dim=-1).unsqueeze(-1)  # [batch_size, num_nodes, 1]
                    
                    # # Get edge indices for each graph in batch
                    # batch_edge_indices = []
                    # for i in range(batch_size):
                    #     edge_index_batch = adj_matrices[i].nonzero()
                    #     batch_edge_indices.append(edge_index_batch)
                    # # Stackeamos todos los edge_index
                    # batch_edge_indices = torch.vstack(batch_edge_indices).t()
                    # # Process through GNN in batch
                    # s = self.gnn_pump(degree_matrices, batch_edge_indices)
                    # s = F.relu(s)
                    # s = self.gnn_pump2(s, batch_edge_indices)
                    
                    # # Apply batched pump function
                    # s_processed, pump_loss, ortho_loss = batched_pump(adj_matrices, s, mask)
                    
                    # # Process final output considering mask
                    # if mask is not None:
                    #     x_final = []
                    #     for i in range(batch_size):
                    #         num_nodes = mask[i].sum()
                    #         s_i = s_processed[i, :num_nodes]
                    #         x_i = x_dense[i, :num_nodes]
                    #         x_final.append(torch.cat([x_i, s_i], dim=1))
                    #     x = torch.cat(x_final, dim=0)
                    # else:
                    #     # If no mask, reshape directly
                    #     x = torch.cat([x_dense, s_processed], dim=-1)
                    #     x = x.view(-1, x.size(-1))
                    
                    
                    
                    # Apply final layer
                    x_new = layer(x, edge_index)
                elif self.pe == "eigen":
                    x = unbatch(x, batch)
                    # # Now we parse to batch again
                    # x = torch.vstack(x)
                    edge_index_nuevo = unbatch_edge_index(edge_index, batch)
                    edge_indexes = []
                    for i in range(len(edge_index_nuevo)):
                        edge_inde = to_dense_adj(edge_index_nuevo[i]).squeeze(0)
                        edge_indexes.append(edge_inde)
                    total_pump_loss = 0
                    total_ortho_loss = 0
                    x_final = []
                    for i in range(len(edge_indexes)):
                        adj = edge_indexes[i]
                        # Nos quedamos con los k menores
                        k = self.k
                        s = get_k_smallest_eigenvectors(adj, k)
                        # Hacemos padding para que sea igual a k
                        if s.shape[1] < k:
                            s = torch.cat((s, torch.zeros(s.shape[0], k - s.shape[1],device=s.device)), dim=-1)
                        # Si tiene menos nodos que x[i].shape[0] hacemos padding
                        if s.shape[0] < x[i].shape[0]:
                            s = torch.cat((s, torch.zeros(x[i].shape[0] - s.shape[0], s.shape[1],device=s.device)), dim=0)
                        #s = self.MLP_Eigen(s)
                        x_s = torch.cat((x[i],s),dim=1)
                        x_final.append(x_s.clone())
                    x = torch.vstack(x_final)
                    x_new = layer(x, edge_index)
                elif self.pe == "LaPE":
                    x = unbatch(x, batch)
                    # # Now we parse to batch again
                    # x = torch.vstack(x)
                    edge_index_nuevo = unbatch_edge_index(edge_index, batch)
                    edge_indexes = []
                    for i in range(len(edge_index_nuevo)):
                        edge_inde = to_dense_adj(edge_index_nuevo[i]).squeeze(0)
                        edge_indexes.append(edge_inde)
                    total_pump_loss = 0
                    total_ortho_loss = 0
                    x_final = []
                    for i in range(len(edge_indexes)):
                        adj = edge_indexes[i]
                        # Nos quedamos con los k menores
                        k = self.k
                        s = get_k_smallest_eigenvectors(adj, k)
                        # Hacemos padding para que sea igual a k
                        if s.shape[1] < k:
                            s = torch.cat((s, torch.zeros(s.shape[0], k - s.shape[1],device=s.device)), dim=-1)
                        # Si tiene menos nodos que x[i].shape[0] hacemos padding
                        if s.shape[0] < x[i].shape[0]:
                            s = torch.cat((s, torch.zeros(x[i].shape[0] - s.shape[0], s.shape[1],device=s.device)), dim=0)
                        s = self.MLP_Eigen(s)
                        s, pump_loss, ortho_loss = pump_lape(adj, s.unsqueeze(0))
                        s = s.squeeze(0)
                        total_pump_loss += pump_loss
                        total_ortho_loss += ortho_loss
                        x_s = torch.cat((x[i],s),dim=1)
                        x_final.append(x_s.clone())
                    x = torch.vstack(x_final)
                    x_new = layer(x, edge_index)
                elif self.pe == "RWPE":
                    x = unbatch(x, batch)
                    # # Now we parse to batch again
                    # x = torch.vstack(x)
                    edge_index_nuevo = unbatch_edge_index(edge_index, batch)
                    edge_indexes = []
                    for i in range(len(edge_index_nuevo)):
                        edge_inde = to_dense_adj(edge_index_nuevo[i]).squeeze(0)
                        edge_indexes.append(edge_inde)
                    total_pump_loss = 0
                    total_ortho_loss = 0
                    x_final = []
                    for i in range(len(edge_indexes)):
                        adj = edge_indexes[i]
                        # Nos quedamos con los k menores
                        k = self.k
                        s = RWPE(adj, k)
                        # Hacemos padding para que sea igual a k
                        if s.shape[1] < k:
                            s = torch.cat((s, torch.zeros(s.shape[0], k - s.shape[1],device=s.device)), dim=-1)
                        # Si tiene menos nodos que x[i].shape[0] hacemos padding
                        if s.shape[0] < x[i].shape[0]:
                            s = torch.cat((s, torch.zeros(x[i].shape[0] - s.shape[0], s.shape[1],device=s.device)), dim=0)
                        s = self.MLP_Eigen(s)
                        s, pump_loss, ortho_loss = pump_lape(adj, s.unsqueeze(0))
                        s = s.squeeze(0)
                        total_pump_loss += pump_loss
                        total_ortho_loss += ortho_loss
                        x_s = torch.cat((x[i],s),dim=1)
                        x_final.append(x_s.clone())
                    x = torch.vstack(x_final)
                    x_new = layer(x, edge_index)
                elif self.pe == "eigen_mlp":
                    x = unbatch(x, batch)
                    # # Now we parse to batch again
                    # x = torch.vstack(x)
                    edge_index_nuevo = unbatch_edge_index(edge_index, batch)
                    edge_indexes = []
                    for i in range(len(edge_index_nuevo)):
                        edge_inde = to_dense_adj(edge_index_nuevo[i]).squeeze(0)
                        edge_indexes.append(edge_inde)
                    total_pump_loss = 0
                    total_ortho_loss = 0
                    x_final = []
                    for i in range(len(edge_indexes)):
                        adj = edge_indexes[i]
                        # Nos quedamos con los k menores
                        k = self.k
                        s = get_k_smallest_eigenvectors(adj, k)
                        # Hacemos padding para que sea igual a k
                        if s.shape[1] < k:
                            s = torch.cat((s, torch.zeros(s.shape[0], k - s.shape[1],device=s.device)), dim=-1)
                        # Si tiene menos nodos que x[i].shape[0] hacemos padding
                        if s.shape[0] < x[i].shape[0]:
                            s = torch.cat((s, torch.zeros(x[i].shape[0] - s.shape[0], s.shape[1],device=s.device)), dim=0)
                        s = self.MLP_Eigen(s)
                        x_s = torch.cat((x[i],s),dim=1)
                        x_final.append(x_s.clone())
                    x = torch.vstack(x_final)
                    x_new = layer(x, edge_index)
                else:
                    total_pump_loss = 0
                    total_ortho_loss = 0
                    x_new = layer(x, edge_index)                        
            else:
                x_new = layer(x, edge_index)
                if i != self.num_layers - 1:
                    x_new = self.act_fn(x_new)
                    x_new = self.dropout(x_new)
                if i == self.num_layers - 1 and self.args.last_layer_fa:
                    combined_values = global_mean_pool(x, batch)
                    combined_values = self.last_layer_transform(combined_values)
                    if self.layer_type in ["R-GCN", "R-GIN"]:
                        x_new += combined_values[batch]
                    else:
                        x_new = combined_values[batch]
            x = x_new 
        if measure_dirichlet:
            energy = dirichlet_normalized(x.cpu().numpy(), graph.edge_index.cpu().numpy())
            return energy
        x = global_mean_pool(x, batch)
        return x, total_pump_loss+ total_ortho_loss

# Trace of a tensor [1,k,k]
def _rank3_trace(x):
    return torch.einsum('ijj->i', x)

# Diagonal version of a tensor [1,n] -> [1,n,n]
def _rank3_diag(x):
    eye = torch.eye(x.size(1)).type_as(x)
    out = eye * x.unsqueeze(2).expand(*x.size(), x.size(1)) 
    return out

def pump(adj, s):
    # Ensure correct dimensions
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s
    k = s.size(-1)
    
    # Apply tanh activation
    s = torch.tanh(s)
    
    # Calculate degree matrix
    d_flat = torch.einsum('ijk->ij', adj)
    d = _rank3_diag(d_flat)
    
    # Calculate losses using matrix operations
    CT_num = _rank3_trace(torch.matmul(torch.matmul(s.transpose(1, 2), adj), s))
    CT_den = _rank3_trace(torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
    
    # Calculate CT loss
    CT_loss = -(CT_num / CT_den)
    CT_loss = torch.mean(CT_loss)
    
    # Calculate orthogonality loss
    ss = torch.matmul(s.transpose(1, 2), s)
    i_s = torch.eye(k, device=ss.device, dtype=ss.dtype)
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) - i_s
    )
    ortho_loss = torch.mean(ortho_loss)
    
    # Clean up memory
    del d, d_flat
    
    return s, CT_loss, ortho_loss
def pump_lape(adj, s): 
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj # adj torch.Size([20, N, N]) N=Mmax
    s = s.unsqueeze(0) if s.dim() == 2 else s # s torch.Size([20, N, k])
    k = s.size(-1)
    s = torch.tanh(s) # torch.Size([20, N, k]) One k for each N of each graph
    d_flat = torch.einsum('ijk->ij', adj) # torch.Size([20, N]) 
    d = _rank3_diag(d_flat) # d torch.Size([20, N, N]) 
    L = d - adj
    CT_num = _rank3_trace(torch.matmul(torch.matmul(s.transpose(1, 2),L), s)) # Tr(S^T A S) 
    
    # Mask with adjacency if proceeds 
    CT_loss = (CT_num) # Tr(S^T A S) / Tr(S^T D S)
    CT_loss = torch.mean(CT_loss)
    
    # Orthogonality regularization.
    ss = torch.matmul(s.transpose(1, 2), s)  
    i_s = torch.eye(k).type_as(ss) 
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
        i_s )  
    ortho_loss = torch.mean(ortho_loss)
    # print(adj.shape)
    # vol = _rank3_trace(d) # Vol(G)
    # adj = (adj) / vol.unsqueeze(1).unsqueeze(1) # Distance matrix normalized
    del d
    #del vol
    del d_flat
    return s, CT_loss, ortho_loss

def batched_pump(adj_matrices, s_batch, mask=None):
    """
    Vectorized implementation of pump for batch processing
    
    Args:
        adj_matrices: [batch_size, num_nodes, num_nodes] adjacency matrices
        s_batch: [batch_size, num_nodes, k] node features
        mask: [batch_size, num_nodes] boolean mask for valid nodes
    """
    # Apply tanh activation
    s_batch = torch.tanh(s_batch)
    
    # Calculate degree matrices for all graphs in batch
    d_flat = torch.sum(adj_matrices, dim=-1)  # [batch_size, num_nodes]
    # Convert to diagonal matrices
    d_matrices = torch.diag_embed(d_flat)  # [batch_size, num_nodes, num_nodes]
    
    # Compute CT numerator and denominator for all graphs
    # s_batch.transpose: [batch_size, k, num_nodes]
    s_T = s_batch.transpose(-2, -1)
    
    # CT numerator: Tr(S^T A S) for all graphs
    CT_num = torch.matmul(torch.matmul(s_T, adj_matrices), s_batch)
    CT_num = torch.diagonal(CT_num, dim1=-2, dim2=-1).sum(-1)  # [batch_size]
    
    # CT denominator: Tr(S^T D S) for all graphs
    CT_den = torch.matmul(torch.matmul(s_T, d_matrices), s_batch)
    CT_den = torch.diagonal(CT_den, dim1=-2, dim2=-1).sum(-1)  # [batch_size]
    
    # Calculate CT loss
    CT_loss = -(CT_num / CT_den)
    CT_loss = torch.mean(CT_loss)
    
    # Calculate orthogonality loss
    ss = torch.matmul(s_T, s_batch)  # [batch_size, k, k]
    i_s = torch.eye(ss.size(-1), device=ss.device, dtype=ss.dtype)
    i_s = i_s.unsqueeze(0).expand(ss.size(0), -1, -1)
    
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) - i_s,
        dim=(-1, -2)
    )
    ortho_loss = torch.mean(ortho_loss)
    
    return s_batch, CT_loss, ortho_loss