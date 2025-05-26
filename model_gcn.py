import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius, radius_graph
from torch_scatter import scatter_mean
from torch_geometric.nn import GCNConv
from common.get_infor_sdf import lig_feature_dims


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class AtomEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        feature_dims = lig_feature_dims
        self.atom_embedding_list = nn.ModuleList()
        self.num_categorical_features = len(feature_dims[0])
        self.num_scalar_features = feature_dims[1]

        for dim in feature_dims[0]:
            emb = nn.Embedding(dim, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())
        return x_embedding


class RecEncoder(nn.Module):
    def __init__(self, emb_dim=97):
        super().__init__()
        self.emb1 = nn.Embedding(20, emb_dim)
        self.emb2 = nn.Linear(199, emb_dim)

    def forward(self, x):
        return self.emb1(x[:, 0].long()) + self.emb2(x[:, 1:].float())


class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        out = self.gcn(x, edge_index)
        out = self.bn(out)
        out = F.relu(out)
        return self.dropout(out)


class GCNModel(nn.Module):
    def __init__(self, device, emb_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.num_layers = num_layers

        self.lig_distance_exp = GaussianSmearing(0.0, 5.0, 32)
        self.rec_distance_exp = GaussianSmearing(0.0, 30.0, 32)

        self.lig_node_encoder = AtomEncoder(emb_dim)
        self.rec_node_encoder = RecEncoder(emb_dim)

        self.lig_edge_encoder = nn.Sequential(
            nn.Linear(4 + 32, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.rec_edge_encoder = nn.Sequential(
            nn.Linear(32, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim)
        )

        self.lig_layers = nn.ModuleList([
            GCNLayer(emb_dim, emb_dim, dropout) for _ in range(num_layers)
        ])
        self.rec_layers = nn.ModuleList([
            GCNLayer(emb_dim, emb_dim, dropout) for _ in range(num_layers)
        ])

        self.output = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1)
        )

    def build_graph(self, pos, edge_index, edge_attr, distance_expansion):
        src, dst = edge_index
        edge_vec = pos[dst] - pos[src]
        edge_len_emb = distance_expansion(edge_vec.norm(dim=-1))
        return torch.cat([edge_attr, edge_len_emb], dim=-1)

    def forward(self, batch):
        # Ligand graph
        lig_node_feat = self.lig_node_encoder(batch['ligand'].x)
        lig_edge_index = radius_graph(batch['ligand'].pos, 5.0, batch['ligand'].batch)

        # Receptor graph
        rec_node_feat = self.rec_node_encoder(batch['receptor'].x)
        rec_edge_index = batch['receptor', 'receptor'].edge_index

        for l in range(self.num_layers):
            lig_node_feat = self.lig_layers[l](lig_node_feat, lig_edge_index)
            rec_node_feat = self.rec_layers[l](rec_node_feat, rec_edge_index)

        lig_feat = scatter_mean(lig_node_feat, batch['ligand'].batch, dim=0)
        rec_feat = scatter_mean(rec_node_feat, batch['receptor'].batch, dim=0)
        concat_feat = torch.cat([lig_feat, rec_feat], dim=1)

        return self.output(concat_feat).squeeze(-1)
