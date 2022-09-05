import torch
import torch.nn as nn
from functools import partial
from typing import Union, List
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import GCNConv, GATv2Conv, GINConv
from torch_geometric.nn.glob import global_mean_pool, global_add_pool, global_max_pool
from torch import Tensor
from torch_sparse import SparseTensor, fill_diag
from torch_geometric.typing import Adj, OptTensor, Size
from torch_geometric.utils import add_self_loops


def get_gnnNets(input_dim, output_dim, model_config):
    conv_classes = {
        'gcn': GCNConv,
        'gat': GATv2Conv,
        'gin': lambda x,y,**z: GINConv(nn.Linear(x,y), **z),
    }
    if model_config.gnn_name.lower() in conv_classes.keys():
        gcn_model_param_names = GCNNet.__init__.__code__.co_varnames
        gcn_model_params = {param_name: getattr(model_config.param, param_name)
                            for param_name in gcn_model_param_names
                            if param_name in model_config.param.keys()}
        return GCNNet(input_dim=input_dim,
                      output_dim=output_dim,
                      conv_class=conv_classes[model_config.gnn_name.lower()],
                      ** gcn_model_params)
    else:
        raise ValueError(f"GNN name should be gcn "
                         f"and {model_config.gnn_name} is not defined.")



def identity(x: torch.Tensor, batch: torch.Tensor):
    return x


def get_readout_layers(readout):
    readout_func_dict = {
        "mean": global_mean_pool,
        "sum": global_add_pool,
        "max": global_max_pool,
        'identity': identity,
    }
    readout_func_dict = {k.lower(): v for k, v in readout_func_dict.items()}
    return readout_func_dict[readout.lower()]


# GNN_LRP takes GNNPool class as pooling layer
class GNNPool(nn.Module):
    def __init__(self, readout):
        super().__init__()
        self.readout = get_readout_layers(readout)

    def forward(self, x, batch):
        return self.readout(x, batch)


def get_nonlinear(nonlinear):
    nonlinear_func_dict = {
        "relu": F.relu,
        "leakyrelu": partial(F.leaky_relu, negative_slope=0.2),
        "sigmoid": F.sigmoid,
        "elu": F.elu
    }
    return nonlinear_func_dict[nonlinear]


class GNNBase(nn.Module):
    def __init__(self):
        super(GNNBase, self).__init__()

    def _argsparse(self, *args, **kwargs):
        r""" Parse the possible input types.
        If the x and edge_index are in args, follow the args.
        In other case, find them in kwargs.
        """
        if args:
            if len(args) == 1:
                data = args[0]
                x = data.x
                edge_index = data.edge_index
                if hasattr(data, 'batch'):
                    batch = data.batch
                else:
                    batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)

            elif len(args) == 2:
                x, edge_index = args[0], args[1]
                batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)

            elif len(args) == 3:
                x, edge_index, batch = args[0], args[1], args[2]

            else:
                raise ValueError(f"forward's args should take 1, 2 or 3 arguments but got {len(args)}")
        else:
            data: Batch = kwargs.get('data')
            if not data:
                x = kwargs.get('x')
                edge_index = kwargs.get('edge_index')
                assert x is not None, "forward's args is empty and required node features x is not in kwargs"
                assert edge_index is not None, "forward's args is empty and required edge_index is not in kwargs"
                batch = kwargs.get('batch')
                if not batch:
                    batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
            else:
                x = data.x
                edge_index = data.edge_index
                if hasattr(data, 'batch'):
                    batch = data.batch
                else:
                    batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
        return x, edge_index, batch


# GCNConv
# class GCNConv(GCNConv):
#     def __init__(self, *args, **kwargs):
#         super(GCNConv, self).__init__(*args, **kwargs)

#     # remove the sigmoid operation for edge_mask in the propagation method
#     def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
#         size = self.__check_input__(edge_index, size)

#         # Run "fused" message and aggregation (if applicable).
#         if (isinstance(edge_index, SparseTensor) and self.fuse
#                 and not self.__explain__):
#             coll_dict = self.__collect__(self.__fused_user_args__, edge_index,
#                                          size, kwargs)

#             msg_aggr_kwargs = self.inspector.distribute(
#                 'message_and_aggregate', coll_dict)
#             out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)

#             update_kwargs = self.inspector.distribute('update', coll_dict)
#             return self.update(out, **update_kwargs)

#         # Otherwise, run both functions in separation.
#         elif isinstance(edge_index, Tensor) or not self.fuse:
#             coll_dict = self.__collect__(self.__user_args__, edge_index, size,
#                                          kwargs)

#             msg_kwargs = self.inspector.distribute('message', coll_dict)
#             out = self.message(**msg_kwargs)

#             # For `GNNExplainer`, we require a separate message and aggregate
#             # procedure since this allows us to inject the `edge_mask` into the
#             # message passing computation scheme.
#             if self.__explain__:
#                 edge_mask = self.__edge_mask__
#                 # Some ops add self-loops to `edge_index`. We need to do the
#                 # same for `edge_mask` (but do not train those).
#                 if out.size(self.node_dim) != edge_mask.size(0):
#                     loop = edge_mask.new_ones(size[0])
#                     edge_mask = torch.cat([edge_mask, loop], dim=0)
#                 assert out.size(self.node_dim) == edge_mask.size(0)
#                 out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))

#             aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
#             out = self.aggregate(out, **aggr_kwargs)

#             update_kwargs = self.inspector.distribute('update', coll_dict)
#             return self.update(out, **update_kwargs)

#     # add edge_weight for normalize=False
#     def forward(self, x: Tensor, edge_index: Adj,
#                 edge_weight: OptTensor = None) -> Tensor:
#         """"""

#         if self.normalize and edge_weight is None:
#             if isinstance(edge_index, Tensor):
#                 cache = self._cached_edge_index
#                 if cache is None:
#                     edge_index, edge_weight = gcn_norm(   # yapf: disable
#                         edge_index, edge_weight, x.size(self.node_dim),
#                         self.improved, self.add_self_loops, dtype=x.dtype)
#                     if self.cached:
#                         self._cached_edge_index = (edge_index, edge_weight)
#                 else:
#                     edge_index, edge_weight = cache[0], cache[1]

#             elif isinstance(edge_index, SparseTensor):
#                 cache = self._cached_adj_t
#                 if cache is None:
#                     edge_index = gcn_norm(
#                         edge_index, edge_weight, x.size(self.node_dim),
#                         self.improved, self.add_self_loops, dtype=x.dtype)
#                     if self.cached:
#                         self._cached_adj_t = edge_index
#                 else:
#                     edge_index = cache

#         # new
#         elif not self.normalize:
#             if isinstance(edge_index, Tensor):
#                 cache = self._cached_edge_index
#                 if cache is None:
#                     if edge_weight is None:
#                         edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
#                     if self.add_self_loops:
#                         edge_index, edge_weight = add_self_loops(
#                             edge_index, edge_weight, num_nodes=x.size(self.node_dim))
#                     if self.cached:
#                         self._cached_edge_index = (edge_index, edge_weight)
#                 else:
#                     edge_index, edge_weight = cache[0], cache[1]
#             elif isinstance(edge_index, SparseTensor):
#                 cache = self._cached_adj_t
#                 if cache is None:
#                     adj_t = edge_index
#                     if not adj_t.has_value():
#                         adj_t = adj_t.fill_value(1.)
#                     if self.add_self_loops:
#                         adj_t = fill_diag(adj_t, 1.)
#                     edge_index = adj_t
#                     if self.cached:
#                         self._cached_adj_t = edge_index

#         # --- add require_grad ---
#         edge_weight.requires_grad_(True)

#         # x = torch.matmul(x, self.weight)
#         x = self.lin(x)

#         # propagate_type: (x: Tensor, edge_weight: OptTensor)
#         out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
#                              size=None)

#         if self.bias is not None:
#             out += self.bias

#         # --- My: record edge_weight ---
#         self.edge_weight = edge_weight

#         return out

import torch_geometric.nn as gnn  



class GCNNet(GNNBase):
    def __init__(self,
                 input_dim: int,
                 basis_dim: int,
                 output_dim: int,
                 num_basis_per_class: int,
                 gnn_latent_dim: Union[List[int]],
                 gnn_dropout: float = 0.0,
                 gnn_emb_normalization: bool = False,
                 gcn_adj_normalization: bool = True,
                 add_self_loop: bool = True,
                 gnn_nonlinear: str = 'relu',
                 readout: str = 'mean',
                 fc_latent_dim: Union[List[int]] = [],
                 fc_dropout: float = 0.0,
                 fc_nonlinear: str = 'relu',
                 conv_class=GCNConv
                 ):
        super(GCNNet, self).__init__()
        # first and last layer - dim_features and classes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.basis_dim = basis_dim
        self.num_basis_per_class = num_basis_per_class
        # GNN part
        self.gnn_latent_dim = gnn_latent_dim
        self.gnn_dropout = gnn_dropout
        self.num_gnn_layers = len(self.gnn_latent_dim)
        self.add_self_loop = add_self_loop
        self.gnn_emb_normalization = gnn_emb_normalization
        self.gcn_adj_normalization = gcn_adj_normalization
        self.gnn_nonlinear = get_nonlinear(gnn_nonlinear)
        # readout
        self.readout_layer = GNNPool(readout)
        # FC part
        self.fc_latent_dim = fc_latent_dim
        self.fc_dropout = fc_dropout
        self.num_mlp_layers = len(self.fc_latent_dim) + 1
        self.fc_nonlinear = get_nonlinear(fc_nonlinear)

        self.emb_dim = self.gnn_latent_dim[-1]
        # GNN layers
        self.convs = nn.ModuleList()
        args = {}
        if conv_class == GCNConv:
            args['normalize']=self.gcn_adj_normalization
            args['add_self_loops']=self.add_self_loop
        elif conv_class == GATv2Conv:
            args['add_self_loops']=self.add_self_loop
        else:
            args['train_eps'] = True
        self.convs.append(conv_class(input_dim, self.gnn_latent_dim[0], **args))
        for i in range(1, self.num_gnn_layers):
            self.convs.append(conv_class(self.gnn_latent_dim[i - 1], self.gnn_latent_dim[i], **args))
        # FC layers
        self.mlps = nn.ModuleList()
        if self.num_mlp_layers > 1:
            self.mlps.append(nn.Linear(self.emb_dim, self.fc_latent_dim[0]))

            for i in range(1, self.num_mlp_layers-1):
                self.mlps.append(nn.Linear(self.fc_latent_dim[i-1], self.fc_latent_dim[i]))
            self.mlps.append(nn.Linear(self.fc_latent_dim[-1], self.basis_dim))
        else:
            self.mlps.append(nn.Linear(self.emb_dim, self.basis_dim))

        self.basis_data = [None]*(self.num_basis_per_class*self.output_dim)
        basis_concepts = torch.Tensor(self.basis_dim, self.num_basis_per_class*self.output_dim)
        self.basis_concepts = nn.Parameter(basis_concepts)
        with torch.no_grad():
            nn.init.uniform_(self.basis_concepts)
        classifier_weights = torch.ones(num_basis_per_class*self.output_dim, self.output_dim)
        for c in range(self.output_dim):
            classifier_weights[:c*num_basis_per_class, c] = -0.5
            classifier_weights[(c+1)*num_basis_per_class:, c] = -0.5
        self.classifier_weights = nn.Parameter(classifier_weights)
        self.classifier_weights.requires_grad = False

    def device(self):
        return self.convs[0].weight.device

    def get_emb(self, *args, **kwargs):
        #  node embedding for GNN
        x, edge_index, _ = self._argsparse(*args, **kwargs)
        for i in range(self.num_gnn_layers):
            # torch.use_deterministic_algorithms(False)
            x = self.convs[i](x, edge_index)
            # torch.use_deterministic_algorithms(True)
            if self.gnn_emb_normalization:
                x = F.normalize(x, p=2, dim=-1)
            x = self.gnn_nonlinear(x)
            x = F.dropout(x, self.gnn_dropout)

        return x

    def forward(self, *args, **kwargs):
        _, _, batch = self._argsparse(*args, **kwargs)
        # node embedding for GNN
        x = self.get_emb(*args, **kwargs)

        for i in range(self.num_mlp_layers - 1):
            x = self.mlps[i](x)
            x = self.fc_nonlinear(x)
            x = F.dropout(x, p=self.fc_dropout)

        x = self.mlps[-1](x)
        
        node_embs = x
        cosine = torch.matmul(x, self.basis_concepts)
        cosine = (cosine.T/(torch.linalg.norm(x, dim=-1)+1e-8)).T
        
        x = self.readout_layer(cosine, batch)

        out = torch.matmul(x, self.classifier_weights)

        return out, node_embs, cosine
