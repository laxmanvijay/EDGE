import torch
import numpy as np
import torch.nn as nn
import torch_geometric as pyg
from torch_scatter import scatter
from torch.nn import functional as F
from ogb.utils.features import get_bond_feature_dims
import torch_sparse
from torch_scatter import scatter, scatter_max, scatter_add
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.utils import maybe_num_nodes, remove_self_loops, add_remaining_self_loops, add_self_loops
from torch_scatter import scatter
from torch_geometric.graphgym.models.layer import BatchNorm1dNode, new_layer_config
import warnings
from yacs.config import CfgNode as CN
import opt_einsum as oe

from grit import RRWPLinearNodeEncoder, RRWPLinearEdgeEncoder

cfg = {
    "out_dir": "results",
    "metric_best": "mae",
    "metric_agg": "argmin",
    "tensorboard_each_run": True,
    "accelerator": "cuda:0",
    "mlflow": {
        "use": False,
        "project": "Exp",
        "name": "zinc-GRIT-RRWP"
    },
    "wandb": {
        "use": False,
        "project": "ZINC"
    },
    "dataset": {
        "name": "subset",
        "task": "graph",
        "task_type": "regression",
        "transductive": False,
        "node_encoder": True,
        "node_encoder_name": "TypeDictNode",
        "node_encoder_num_types": 21,
        "node_encoder_bn": False,
        "edge_encoder": True,
        "edge_encoder_name": "TypeDictEdge",
        "edge_encoder_num_types": 4,
        "edge_encoder_bn": False
    },
    "posenc_RRWP": {
        "enable": True,
        "ksteps": 21,
        "add_identity": True,
        "add_node_attr": False,
        "add_inverse": False
    },
    "train": {
        "mode": "custom",
        "batch_size": 32,
        "eval_period": 1,
        "enable_ckpt": True,
        "ckpt_best": True,
        "ckpt_clean": True
    },
    "model": {
        "type": "GritTransformer",
        "loss_fun": "l1",
        "edge_decoding": "dot",
        "graph_pooling": "add"
    },
    "gt": {
        "layer_type": "GritTransformer",
        "layers": 10,
        "n_heads": 8,
        "dim_hidden": 64,
        "dropout": 0.0,
        "layer_norm": False,
        "batch_norm": True,
        "update_e": True,
        "attn_dropout": 0.2,
        "attn": {
            "clamp": 5.0,
            "act": "relu",
            "full_attn": True,
            "edge_enhance": True,
            "O_e": True,
            "norm_e": True,
            "fwl": False
        }
    },
    "gnn": {
        "head": "san_graph",
        "layers_pre_mp": 0,
        "layers_post_mp": 3,
        "dim_inner": 64,
        "batchnorm": True,
        "act": "relu",
        "dropout": 0.0,
        "agg": "mean",
        "normalize_adj": False
    },
    "optim": {
        "clip_grad_norm": True,
        "optimizer": "adamW",
        "weight_decay": 1e-5,
        "base_lr": 1e-3,
        "max_epoch": 2000,
        "num_warmup_epochs": 50,
        "scheduler": "cosine_with_warmup",
        "min_lr": 1e-6
    }
}

def full_edge_index(edge_index, batch=None):
    """
    Return the Full batched sparse adjacency matrices given by edge indices.
    Returns batched sparse adjacency matrices with exactly those edges that
    are not in the input `edge_index` while ignoring self-loops.
    Implementation inspired by `torch_geometric.utils.to_dense_adj`
    Args:
        edge_index: The edge indices.
        batch: Batch vector, which assigns each node to a specific example.
    Returns:
        Complementary edge index.
    """

    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch,
                        dim=0, dim_size=batch_size, reduce='add')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    negative_index_list = []
    for i in range(batch_size):
        n = num_nodes[i].item()
        size = [n, n]
        adj = torch.ones(size, dtype=torch.short,
                         device=edge_index.device)

        adj = adj.view(size)
        _edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        # _edge_index, _ = remove_self_loops(_edge_index)
        negative_index_list.append(_edge_index + cum_nodes[i])

    edge_index_full = torch.cat(negative_index_list, dim=1).contiguous()
    return edge_index_full

class MetaPathRRWPEncoder(torch.nn.Module):
    """Handles RRWP encoding for multiple meta-paths in heterogeneous graphs"""
    
    def __init__(self, meta_paths, ksteps, emb_dim):
        super().__init__()
        self.meta_paths = meta_paths  # List of meta-paths
        self.ksteps = ksteps
        self.emb_dim = emb_dim
        
        # Create separate projections for each meta-path
        self.path_projections = nn.ModuleDict({
            f"path_{i}": nn.Linear(ksteps, emb_dim)
            for i in range(len(meta_paths))
        })
        
    def compute_transition_matrix(self, edge_index, edge_type, num_nodes, meta_path):
        """Compute transition matrix for a specific meta-path"""
        adj_matrices = []
        
        # Create adjacency matrix for each edge type in meta-path
        for edge_t in meta_path:
            mask = edge_type == edge_t
            edges = edge_index[:, mask]
            adj = torch.sparse.FloatTensor(
                edges, 
                torch.ones(edges.size(1), device=edges.device),
                torch.Size([num_nodes, num_nodes])
            )
            adj_matrices.append(adj)
            
        # Multiply adjacency matrices to get meta-path transition matrix
        trans_matrix = adj_matrices[0]
        for adj in adj_matrices[1:]:
            trans_matrix = torch.sparse.mm(trans_matrix, adj)
            
        return trans_matrix

    def forward(self, batch):
        """
        Compute RRWP for each meta-path and combine them
        """
        num_nodes = batch.num_nodes
        device = batch.x.device
        
        # Store RRWP for each meta-path
        meta_path_rrwps = []
        
        for i, meta_path in enumerate(self.meta_paths):
            # Compute transition matrix for this meta-path
            trans_matrix = self.compute_transition_matrix(
                batch.edge_index,
                batch.edge_type,
                num_nodes,
                meta_path
            )
            
            # Compute powers of transition matrix up to k steps
            powers = [torch.eye(num_nodes, device=device)]
            curr_power = trans_matrix
            for _ in range(self.ksteps - 1):
                powers.append(curr_power)
                curr_power = torch.sparse.mm(curr_power, trans_matrix)
            
            # Stack powers to get RRWP for this meta-path
            path_rrwp = torch.stack(powers, dim=-1)
            
            # Project RRWP to embedding dimension
            path_rrwp = self.path_projections[f"path_{i}"](path_rrwp)
            meta_path_rrwps.append(path_rrwp)
        
        # Combine RRWPs from all meta-paths
        combined_rrwp = torch.stack(meta_path_rrwps, dim=1)
        return combined_rrwp

class HeteroRRWPLinearNodeEncoder(RRWPLinearNodeEncoder):
    """Modified node encoder for heterogeneous graphs"""
    
    def __init__(self, meta_paths, ksteps, emb_dim, **kwargs):
        super().__init__(emb_dim=emb_dim, **kwargs)
        self.meta_path_encoder = MetaPathRRWPEncoder(meta_paths, ksteps, emb_dim)
        
    def forward(self, batch):
        # Get meta-path specific RRWPs
        rrwp = self.meta_path_encoder(batch)
        
        # Combine meta-path RRWPs
        rrwp = rrwp.mean(dim=1)  # Average across meta-paths
        
        # Apply original encoding
        rrwp = self.fc(rrwp)
        
        if self.batchnorm:
            rrwp = self.bn(rrwp)
        if self.layernorm:
            rrwp = self.ln(rrwp)
            
        if "x" in batch:
            batch.x = batch.x + rrwp
        else:
            batch.x = rrwp
            
        return batch

class HeteroRRWPLinearEdgeEncoder(RRWPLinearEdgeEncoder):
    """Modified edge encoder for heterogeneous graphs"""
    
    def __init__(self, meta_paths, ksteps, emb_dim, **kwargs):
        super().__init__(emb_dim=emb_dim, **kwargs)
        self.meta_path_encoder = MetaPathRRWPEncoder(meta_paths, ksteps, emb_dim)
        
    def forward(self, batch):
        # Get meta-path specific RRWPs
        rrwp = self.meta_path_encoder(batch)
        
        # Convert node RRWPs to edge RRWPs
        edge_rrwp = []
        for i in range(len(self.meta_paths)):
            src_rrwp = rrwp[batch.edge_index[0], i]
            dst_rrwp = rrwp[batch.edge_index[1], i]
            path_edge_rrwp = torch.cat([src_rrwp, dst_rrwp], dim=-1)
            edge_rrwp.append(path_edge_rrwp)
            
        # Combine edge RRWPs from different meta-paths
        edge_rrwp = torch.stack(edge_rrwp, dim=1).mean(dim=1)
        
        # Apply original edge encoding logic
        batch.rrwp_index = batch.edge_index
        batch.rrwp_val = self.fc(edge_rrwp)
        
        return super().forward(batch)

class GraphAddPooling(torch.nn.Module):
    """Simple graph pooling that sums node features across each graph"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x, batch):
        """
        Args:
            x: Node features [num_nodes, feat_dim]
            batch: Batch vector [num_nodes] mapping each node to its graph
        Returns:
            Pooled graph features [batch_size, feat_dim]
        """
        # Default to single graph if no batch specified
        if batch is None:
            batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
            
        # Sum node features for each graph in batch
        out = scatter(x, batch, dim=0, reduce='add')
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}()'

class SimpleReLU(nn.Module):
    """Simple ReLU activation module"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """
        Args:
            x: Input tensor
        Returns:
            ReLU activated tensor
        """
        return torch.relu(x)
    
    def __repr__(self):
        return f'{self.__class__.__name__}()'

class SANGraphHead(nn.Module):
    """
    SAN prediction head for graph prediction tasks.
    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
        L (int): Number of hidden layers.
    """

    def __init__(self, dim_in, dim_out, L=2):
        super().__init__()
        self.deg_scaler = False
        self.fwl = False
        self.pooling_fun = GraphAddPooling()

        list_FC_layers = [
            nn.Linear(dim_in // 2 ** l, dim_in // 2 ** (l + 1), bias=True)
            for l in range(L)]
        list_FC_layers.append(
            nn.Linear(dim_in // 2 ** L, dim_out, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.activation = SimpleReLU()

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch):
        graph_emb = self.pooling_fun(batch.x, batch.batch)
        for l in range(self.L):
            graph_emb = self.FC_layers[l](graph_emb)
            graph_emb = self.activation(graph_emb)
        graph_emb = self.FC_layers[self.L](graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label

class TypeDictNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        num_types = cfg["dataset"]["node_encoder_num_types"]
        if num_types < 1:
            raise ValueError(f"Invalid 'node_encoder_num_types': {num_types}")

        self.encoder = torch.nn.Embedding(num_embeddings=num_types,
                                          embedding_dim=emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        # Encode just the first dimension if more exist
        batch.x = self.encoder(batch.x[:, 0])

        return batch

class TypeDictEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        num_types = cfg["dataset"]["edge_encoder_num_types"]
        if num_types < 1:
            raise ValueError(f"Invalid 'edge_encoder_num_types': {num_types}")

        self.encoder = torch.nn.Embedding(num_embeddings=num_types,
                                          embedding_dim=emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        batch.edge_attr = self.encoder(batch.edge_attr)
        return batch

class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if cfg["dataset"]["node_encoder"]:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = TypeDictNodeEncoder(cfg.gnn.dim_inner)
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg["dataset"]["node_encoder_bn"]:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg["gnn"]["dim_inner"], -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
            # Update dim_in to reflect the new dimension of the node features
            self.dim_in = cfg["gnn"]["dim_inner"]
        if cfg["dataset"]["edge_encoder"]:
            # Hard-limit max edge dim for PNA.
            if 'PNA' in cfg["gt"]["layer_type"]:
                cfg["gnn"]["dim_edge"] = min(128, cfg["gnn"]["dim_inner"])
            else:
                cfg["gnn"]["dim_edge"] = cfg["gnn"]["dim_inner"]
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = TypeDictEdgeEncoder(cfg["gnn"]["dim_edge"])
            self.edge_encoder = EdgeEncoder(cfg["gnn"]["dim_edge"])
            if cfg["dataset"]["edge_encoder_bn"]:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg["gnn"]["dim_edge"], -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch

def pyg_softmax(src, index, num_nodes=None):
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    num_nodes = maybe_num_nodes(index, num_nodes)

    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (
            scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out



class MultiHeadAttentionLayerGritSparse(nn.Module):
    """
        Proposed Attention Computation for GRIT
    """

    def __init__(self, in_dim, out_dim, num_heads, use_bias,
                 clamp=5., dropout=0., act=None,
                 edge_enhance=True,
                 sqrt_relu=False,
                 signed_sqrt=True,
                 cfg=CN(),
                 **kwargs):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.clamp = np.abs(clamp) if clamp is not None else None
        self.edge_enhance = edge_enhance

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(in_dim, out_dim * num_heads * 2, bias=True)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.E.weight)
        nn.init.xavier_normal_(self.V.weight)

        self.Aw = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, 1), requires_grad=True)
        nn.init.xavier_normal_(self.Aw)

        self.act = nn.Identity()

        if self.edge_enhance:
            self.VeRow = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, self.out_dim), requires_grad=True)
            nn.init.xavier_normal_(self.VeRow)

    def propagate_attention(self, batch):
        src = batch.K_h[batch.edge_index[0]]      # (num relative) x num_heads x out_dim
        dest = batch.Q_h[batch.edge_index[1]]     # (num relative) x num_heads x out_dim
        score = src + dest                        # element-wise multiplication

        if batch.get("E", None) is not None:
            batch.E = batch.E.view(-1, self.num_heads, self.out_dim * 2)
            E_w, E_b = batch.E[:, :, :self.out_dim], batch.E[:, :, self.out_dim:]
            # (num relative) x num_heads x out_dim
            score = score * E_w
            score = torch.sqrt(torch.relu(score)) - torch.sqrt(torch.relu(-score))
            score = score + E_b

        score = self.act(score)
        e_t = score

        # output edge
        if batch.get("E", None) is not None:
            batch.wE = score.flatten(1)

        # final attn
        score = oe.contract("ehd, dhc->ehc", score, self.Aw, backend="torch")
        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)

        raw_attn = score
        score = pyg_softmax(score, batch.edge_index[1])  # (num relative) x num_heads x 1
        score = self.dropout(score)
        batch.attn = score

        # Aggregate with Attn-Score
        msg = batch.V_h[batch.edge_index[0]] * score  # (num relative) x num_heads x out_dim
        batch.wV = torch.zeros_like(batch.V_h)  # (num nodes in batch) x num_heads x out_dim
        scatter(msg, batch.edge_index[1], dim=0, out=batch.wV, reduce='add')

        if self.edge_enhance and batch.E is not None:
            rowV = scatter(e_t * score, batch.edge_index[1], dim=0, reduce="add")
            rowV = oe.contract("nhd, dhc -> nhc", rowV, self.VeRow, backend="torch")
            batch.wV = batch.wV + rowV

    def forward(self, batch):
        Q_h = self.Q(batch.x)
        K_h = self.K(batch.x)

        V_h = self.V(batch.x)
        if batch.get("edge_attr", None) is not None:
            batch.E = self.E(batch.edge_attr)
        else:
            batch.E = None

        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)
        self.propagate_attention(batch)
        h_out = batch.wV
        e_out = batch.get('wE', None)

        return h_out, e_out

class GritTransformerLayer(nn.Module):
    """
        Proposed Transformer Layer for GRIT
    """
    def __init__(self, in_dim, out_dim, num_heads,
                 dropout=0.0,
                 attn_dropout=0.0,
                 layer_norm=False, batch_norm=True,
                 residual=True,
                 act='relu',
                 norm_e=True,
                 O_e=True,
                 cfg=dict(),
                 **kwargs):
        super().__init__()

        self.debug = False
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        # -------
        self.update_e = cfg.get("update_e", True)
        self.bn_momentum = cfg.bn_momentum
        self.bn_no_runner = cfg.bn_no_runner
        self.rezero = cfg.get("rezero", False)

        self.act = nn.Identity()
        if cfg.get("attn", None) is None:
            cfg.attn = dict()
        self.use_attn = cfg.attn.get("use", True)
        # self.sigmoid_deg = cfg.attn.get("sigmoid_deg", False)
        self.deg_scaler = cfg.attn.get("deg_scaler", True)

        self.attention = MultiHeadAttentionLayerGritSparse(
            in_dim=in_dim,
            out_dim=out_dim // num_heads,
            num_heads=num_heads,
            use_bias=cfg.attn.get("use_bias", False),
            dropout=attn_dropout,
            clamp=cfg.attn.get("clamp", 5.),
            act=cfg.attn.get("act", "relu"),
            edge_enhance=cfg.attn.get("edge_enhance", True),
            sqrt_relu=cfg.attn.get("sqrt_relu", False),
            signed_sqrt=cfg.attn.get("signed_sqrt", False),
            scaled_attn =cfg.attn.get("scaled_attn", False),
            no_qk=cfg.attn.get("no_qk", False),
        )

        self.O_h = nn.Linear(out_dim//num_heads * num_heads, out_dim)
        if O_e:
            self.O_e = nn.Linear(out_dim//num_heads * num_heads, out_dim)
        else:
            self.O_e = nn.Identity()

        # -------- Deg Scaler Option ------

        if self.deg_scaler:
            self.deg_coef = nn.Parameter(torch.zeros(1, out_dim//num_heads * num_heads, 2))
            nn.init.xavier_normal_(self.deg_coef)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim) if norm_e else nn.Identity()

        if self.batch_norm:
            # when the batch_size is really small, use smaller momentum to avoid bad mini-batch leading to extremely bad val/test loss (NaN)
            self.batch_norm1_h = nn.BatchNorm1d(out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=cfg.bn_momentum)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=cfg.bn_momentum) if norm_e else nn.Identity()

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=cfg.bn_momentum)

        if self.rezero:
            self.alpha1_h = nn.Parameter(torch.zeros(1,1))
            self.alpha2_h = nn.Parameter(torch.zeros(1,1))
            self.alpha1_e = nn.Parameter(torch.zeros(1,1))

    def forward(self, batch):
        h = batch.x
        num_nodes = batch.num_nodes
        log_deg = torch.log(0)

        h_in1 = h  # for first residual connection
        e_in1 = batch.get("edge_attr", None)
        e = None
        # multi-head attention out

        h_attn_out, e_attn_out = self.attention(batch)

        h = h_attn_out.view(num_nodes, -1)
        h = F.dropout(h, self.dropout, training=self.training)

        # degree scaler
        if self.deg_scaler:
            h = torch.stack([h, h * log_deg], dim=-1)
            h = (h * self.deg_coef).sum(dim=-1)

        h = self.O_h(h)
        if e_attn_out is not None:
            e = e_attn_out.flatten(1)
            e = F.dropout(e, self.dropout, training=self.training)
            e = self.O_e(e)

        if self.residual:
            if self.rezero: h = h * self.alpha1_h
            h = h_in1 + h  # residual connection
            if e is not None:
                if self.rezero: e = e * self.alpha1_e
                e = e + e_in1

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            if e is not None: e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            if e is not None: e = self.batch_norm1_e(e)

        # FFN for h
        h_in2 = h  # for second residual connection
        h = self.FFN_h_layer1(h)
        h = self.act(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            if self.rezero: h = h * self.alpha2_h
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)

        batch.x = h
        if self.update_e:
            batch.edge_attr = e
        else:
            batch.edge_attr = e_in1

        return batch

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})\n[{}]'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels, self.num_heads, self.residual,
            super().__repr__(),
        )

class GritTransformer(torch.nn.Module):
    '''
        The proposed GritTransformer (Graph Inductive Bias Transformer)
    '''

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        self.ablation = True
        self.ablation = False

        if cfg["posenc_RRWP"]["enable"]:
            self.rrwp_abs_encoder = HeteroRRWPLinearNodeEncoder(
                meta_paths=cfg["posenc_RRWP"]["meta_paths"],
                ksteps=cfg["posenc_RRWP"]["ksteps"],
                emb_dim=cfg["gnn"]["dim_inner"]
            )
            self.rrwp_rel_encoder = HeteroRRWPLinearEdgeEncoder(
                meta_paths=cfg["posenc_RRWP"]["meta_paths"],
                ksteps=cfg["posenc_RRWP"]["ksteps"],
                emb_dim=cfg["gnn"]["dim_edge"],
                pad_to_full_graph=cfg["gt"]["attn"]["full_attn"]
            )

        if cfg["gnn"]["layers_pre_mp"] > 0:
            self.pre_mp = GNNPreMP(
            dim_in, cfg["gnn"]["dim_inner"], cfg["gnn"]["layers_pre_mp"]
            )
            dim_in = cfg["gnn"]["dim_inner"]

        assert (
            cfg["gt"]["dim_hidden"] == cfg["gnn"]["dim_inner"] == dim_in
        ), "The inner and hidden dims must match."

        layers = []
        for l in range(cfg["gt"]["layers"]):
            layers.append(
            GritTransformerLayer(
                in_dim=cfg["gt"]["dim_hidden"],
                out_dim=cfg["gt"]["dim_hidden"],
                num_heads=cfg["gt"]["n_heads"],
                dropout=cfg["gt"]["dropout"],
                act=cfg["gnn"]["act"],
                attn_dropout=cfg["gt"]["attn_dropout"],
                layer_norm=cfg["gt"]["layer_norm"],
                batch_norm=cfg["gt"]["batch_norm"],
                residual=True,
                norm_e=cfg["gt"]["attn"]["norm_e"],
                O_e=cfg["gt"]["attn"]["O_e"],
                cfg=cfg["gt"],
            )
            )
        # layers = []

        self.layers = torch.nn.Sequential(*layers)
        self.post_mp = SANGraphHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)

        return batch


