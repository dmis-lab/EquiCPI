import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius, radius_graph
from torch_scatter import scatter, scatter_mean
from e3nn import o3
from e3nn.nn import BatchNorm
from common.get_infor_sdf import lig_feature_dims


class GaussianSmearing(torch.nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        feature_dims = lig_feature_dims
        self.atom_embedding_list = torch.nn.ModuleList()
        self.num_categorical_features = len(feature_dims[0])
        self.num_scalar_features = feature_dims[1]

        for _, dim in enumerate(feature_dims[0]):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        assert x.shape[1] == self.num_categorical_features + self.num_scalar_features 
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())
        return x_embedding


class RecEncoder(torch.nn.Module):
    def __init__(self, emb_dim = 97):
        super(RecEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        first_emb = torch.nn.Embedding(20 , emb_dim)
        last_emb = torch.nn.Linear(199,emb_dim)

        self.atom_embedding_list.extend([first_emb, last_emb])
        self.num_categorical_features = 2

    def forward(self, x):
        x_embedding = 0
        x_embedding = self.atom_embedding_list[0](x[:, 0].long())
        x_embedding = x_embedding + self.atom_embedding_list[1](x[:,1:].float())
        return x_embedding

class TensorProductConvLayer(torch.nn.Module):
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=True, batch_norm=True, dropout=0.0,
                 hidden_features=None):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        if hidden_features is None:
            hidden_features = n_edge_features

        self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, tp.weight_numel)
        )
        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean'):

        edge_src, edge_dst = edge_index
        tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)

        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded

        if self.batch_norm:
            out = self.batch_norm(out)
        return out

class TensorProductModel_one_hot3(torch.nn.Module):
    def __init__(self, device, in_lig_edge_features=4, sh_lmax=2,
             ns=16, nv=4, num_conv_layers=2, lig_max_radius=5, rec_max_radius=30, cross_max_distance=250,
             distance_embed_dim=32, cross_distance_embed_dim=32,
             use_second_order_repr=False, batch_norm=True,
             dropout=0.0,
             confidence_dropout=0, confidence_no_batchnorm=False, num_confidence_outputs=1,
             morgan_net=[2048,512,256], 
             task_ml= 'classification', 
             interaction_net = [256,256,256]):
        super(TensorProductModel_one_hot3, self).__init__()
        self.confidence_no_batchnorm = confidence_no_batchnorm
        self.confidence_dropout = confidence_dropout
        self.task_ml = task_ml
        self.in_lig_edge_features = in_lig_edge_features
        self.lig_max_radius = lig_max_radius
        self.rec_max_radius = rec_max_radius
        self.cross_max_distance = cross_max_distance
        self.distance_embed_dim = distance_embed_dim
        self.cross_distance_embed_dim = cross_distance_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.device = device
        self.num_conv_layers = num_conv_layers

        self.lig_distance_expansion = GaussianSmearing(0.0, lig_max_radius, distance_embed_dim)
        self.rec_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)

        self.lig_node_embedding = AtomEncoder(emb_dim=ns)
        self.lig_edge_embedding = nn.Sequential(nn.Linear(in_lig_edge_features + distance_embed_dim, ns),nn.ReLU(),nn.Dropout(dropout),nn.Linear(ns, ns))

        self.rec_node_embedding = RecEncoder(emb_dim=ns)
        self.rec_edge_embedding = nn.Sequential(nn.Linear(distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.cross_edge_embedding = nn.Sequential(nn.Linear(cross_distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))
        self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_distance, cross_distance_embed_dim)

        
        if use_second_order_repr == '1':
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]
        if use_second_order_repr == '2':
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {ns}x0o',
                f'{ns}x0e + {ns}x0o + {nv}x1e',
                f'{ns}x0e + {ns}x0o + {nv}x1e + {nv}x1o',
                f'{ns}x0e + {ns}x0o + {nv}x1e',
                f'{ns}x0e + {ns}x0o',
                f'{ns}x0e'
            ]
        if use_second_order_repr == '3':
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e'
            ]
        if use_second_order_repr == '4':
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o' ,
                f'{ns}x0e'
            ]

        if use_second_order_repr == '5':
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e']        
        # convolutional layers
        lig_conv_layers, rec_conv_layers, rec_to_lig_conv_layers, lig_to_rec_conv_layers = [], [], [], []
        ecfp_nets, interaction_net_ne = [], []
        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': self.sh_irreps,
                'out_irreps': out_irreps,
                'n_edge_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }
            lig_layer = TensorProductConvLayer(**parameters)
            lig_conv_layers.append(lig_layer)
            rec_layer = TensorProductConvLayer(**parameters)
            rec_conv_layers.append(rec_layer)
            rec_to_lig_layer = TensorProductConvLayer(**parameters)
            rec_to_lig_conv_layers.append(rec_to_lig_layer)
            lig_to_rec_layer = TensorProductConvLayer(**parameters)
            lig_to_rec_conv_layers.append(lig_to_rec_layer)

        self.conv_lig_layers = nn.ModuleList(lig_conv_layers)
        self.conv_rec_layers = nn.ModuleList(rec_conv_layers)
        self.rec_to_lig_conv_layers = nn.ModuleList(rec_to_lig_conv_layers)
        self.lig_to_rec_conv_layers = nn.ModuleList(lig_to_rec_conv_layers)

        for i in range(len(morgan_net)-1):
            ecfp_nets.extend(self.sequential_linear_layer(morgan_net[i],morgan_net[i+1]))

        interaction_net_ne.extend(self.sequential_linear_layer(ns*4 + morgan_net[-1], interaction_net[0]))
        for i in range(len(interaction_net)-1):
            interaction_net_ne.extend(self.sequential_linear_layer(interaction_net[i], interaction_net[i+1]))
        interaction_net_ne.append(nn.Linear(interaction_net[-1], num_confidence_outputs))
        
        self.ecfp_nets = nn.Sequential(*ecfp_nets)
        self.interaction_net_ne = nn.Sequential(*interaction_net_ne)

    def sequential_linear_layer(self , hid_in, hid_out):
        return [nn.Linear(hid_in, hid_out),
                nn.BatchNorm1d( hid_out) if not self.confidence_no_batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(self.confidence_dropout)]
    
    def build_lig_conv_graph(self, batch):
        radius_edges = radius_graph(batch['ligand'].pos, self.lig_max_radius, batch['ligand'].batch)
        
        edge_index = torch.cat([batch['ligand', 'ligand'].edge_index, radius_edges], 1).long()
        edge_attr = torch.cat([batch['ligand', 'ligand'].edge_attr, torch.zeros(radius_edges.shape[-1], self.in_lig_edge_features, device=batch['ligand'].x.device)], 0)

        node_attr = batch['ligand'].x
        src, dst = edge_index
        edge_vec = batch['ligand'].pos[dst.long()] - batch['ligand'].pos[src.long()]
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = torch.cat([edge_attr, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return node_attr, edge_index, edge_attr, edge_sh

    def build_rec_conv_graph(self, batch):
        # builds the receptor initial node and edge embeddings
        node_attr = batch['receptor'].x
        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = batch['receptor', 'receptor'].edge_index
        src, dst = edge_index.long()
        edge_vec = batch['receptor'].pos[dst.long()] - batch['receptor'].pos[src.long()]
        edge_length_emb = self.rec_distance_expansion(edge_vec.norm(dim=-1))
        edge_attr = edge_length_emb
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return node_attr, edge_index, edge_attr, edge_sh

    def build_cross_conv_graph(self, data, cross_distance_cutoff):
        # builds the cross edges between ligand and receptor
        if torch.is_tensor(cross_distance_cutoff):
            # different cutoff for every graph (depends on the diffusion time)
            edge_index = radius(data['receptor'].pos / cross_distance_cutoff[data['receptor'].batch],
                                data['ligand'].pos / cross_distance_cutoff[data['ligand'].batch], 1,
                                data['receptor'].batch, data['ligand'].batch, max_num_neighbors=100)
        else:
            edge_index = radius(data['receptor'].pos, data['ligand'].pos, cross_distance_cutoff,
                            data['receptor'].batch, data['ligand'].batch, max_num_neighbors=100)

        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['ligand'].pos[src.long()]

        edge_length_emb = self.cross_distance_expansion(edge_vec.norm(dim=-1))
        # edge_sigma_emb = data['ligand'].node_sigma_emb[src.long()]
        edge_attr =  edge_length_emb
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return edge_index, edge_attr, edge_sh
        
    def forward(self, batch):
        # build ligand graph
        lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh = self.build_lig_conv_graph(batch)
        lig_node_attr = self.lig_node_embedding(lig_node_attr)
        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)

        rec_node_attr, rec_edge_index, rec_edge_attr, rec_edge_sh = self.build_rec_conv_graph(batch)
        rec_src, rec_dst = rec_edge_index
        rec_node_attr = self.rec_node_embedding(rec_node_attr)
        rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)

        # build cross graph
        cross_cutoff = self.cross_max_distance
        cross_edge_index, cross_edge_attr, cross_edge_sh = self.build_cross_conv_graph(batch, cross_cutoff)
        cross_lig, cross_rec = cross_edge_index
        cross_edge_attr = self.cross_edge_embedding(cross_edge_attr)

        for l in range(self.num_conv_layers):
            # LIGAND updates
            lig_edge_attr_ = torch.cat([lig_edge_attr, lig_node_attr[lig_edge_index[0], :self.ns], lig_node_attr[lig_edge_index[1], :self.ns]], -1)
            lig_intra_update = self.conv_lig_layers[l](lig_node_attr, lig_edge_index, lig_edge_attr_, lig_edge_sh)
            
            # Cross update
            rec_to_lig_edge_attr_ = torch.cat([cross_edge_attr, lig_node_attr[cross_lig, :self.ns], rec_node_attr[cross_rec, :self.ns]], -1)
            lig_inter_update = self.rec_to_lig_conv_layers[l](rec_node_attr, cross_edge_index, rec_to_lig_edge_attr_, cross_edge_sh,
                                                              out_nodes=lig_node_attr.shape[0])
            
            # REC update
            rec_edge_attr_ = torch.cat([rec_edge_attr, rec_node_attr[rec_src, :self.ns], rec_node_attr[rec_dst, :self.ns]], -1)
            rec_intra_update = self.conv_rec_layers[l](rec_node_attr, rec_edge_index, rec_edge_attr_, rec_edge_sh)

            lig_to_rec_edge_attr_ = torch.cat([cross_edge_attr, lig_node_attr[cross_lig, :self.ns], rec_node_attr[cross_rec, :self.ns]], -1)
            rec_inter_update = self.lig_to_rec_conv_layers[l](lig_node_attr, torch.flip(cross_edge_index, dims=[0]), lig_to_rec_edge_attr_,
                                                                  cross_edge_sh, out_nodes=rec_node_attr.shape[0])

            # padding original features and update features with residual updates LIG
            lig_node_attr = F.pad(lig_node_attr, (0, lig_intra_update.shape[-1] - lig_node_attr.shape[-1]))
            lig_node_attr = (lig_node_attr + lig_intra_update + lig_inter_update)/3

            rec_node_attr = F.pad(rec_node_attr, (0, rec_intra_update.shape[-1] - rec_node_attr.shape[-1]))
            rec_node_attr = (rec_node_attr + rec_intra_update + rec_inter_update)/3

        scalar_lig_attr = torch.cat([lig_node_attr[:,:self.ns],lig_node_attr[:,-self.ns:]], dim=1) if self.num_conv_layers >= 3 else lig_node_attr[:,:self.ns]
        lig_feat_final = scatter_mean(scalar_lig_attr, batch['ligand'].batch, dim=0)

        scalar_rec_attr = torch.cat([rec_node_attr[:,:self.ns],rec_node_attr[:,-self.ns:]], dim=1) if self.num_conv_layers >= 3 else rec_node_attr[:,:self.ns]
        rec_feat_final = scatter_mean(scalar_rec_attr, batch['receptor'].batch, dim=0)

        ecpf_infor = self.ecfp_nets(batch['ligand'].morgan_fingerprint)
        cpi_feat = torch.cat((rec_feat_final, lig_feat_final, ecpf_infor), dim=1)
        if self.task_ml == 'classification':
            output = torch.sigmoid(self.interaction_net_ne(cpi_feat).squeeze(dim=-1))
        else:
            output = self.interaction_net_ne(cpi_feat).squeeze(dim=-1)

        return output
