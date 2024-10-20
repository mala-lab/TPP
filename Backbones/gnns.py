from .gnnconv import GATConv, GCNLayer, GINConv, DGLGCN
from .layers import PairNorm
from .utils import *
from dgl.base import DGLError
from dgl.nn.pytorch import edge_softmax
import dgl.function as fn
import ipdb
import torch as th
linear_choices = {'nn.Linear':nn.Linear, 'Linear_IL':Linear_IL}

class GIN(nn.Module):
    def __init__(self, args,):
        super(GIN, self).__init__()
        dims = [args.d_data] + args.GIN_args['h_dims'] + [args.n_cls]
        self.dropout = args.GIN_args['dropout']
        self.gat_layers = nn.ModuleList()
        for l in range(len(dims)-1):
            lin = torch.nn.Linear(dims[l], dims[l+1])
            self.gat_layers.append(GINConv(lin, 'sum'))


    def forward(self, g, features):
        e_list = []
        h, e = self.gat_layers[0](g, features)
        x = F.relu(h)
        logits, e = self.gat_layers[1](g, x)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        e_list = e_list + e
        return logits, e_list

    def forward_batch(self, blocks, features):
        e_list = []
        h, e = self.gat_layers[0].forward_batch(blocks[0], features)
        x = F.relu(h)
        logits, e = self.gat_layers[1].forward_batch(blocks[1], x)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        e_list = e_list + e
        return logits, e_list

    def reset_params(self):
        for layer in self.gat_layers:
            layer.reset_parameters()

class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        dims = [args.d_data] + args.GCN_args['h_dims'] + [args.GCN_args['h_dims'][-1]]
        self.dropout = args.GCN_args['dropout']
        self.gat_layers = nn.ModuleList()
        for l in range(len(dims)-1):
            self.gat_layers.append(DGLGCN(dims[l], dims[l+1]))
        args.hidden = args.GCN_args['h_dims'][-1]

    def forward(self, g, features):
        h = features
        for layer in self.gat_layers[:-1]:
            h = layer(g, h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.gat_layers[-1](g, h)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        return logits

    def forward_batch(self, blocks, features):
        e_list = []
        h = features
        for i,layer in enumerate(self.gat_layers[:-1]):
            h, e = layer.forward_batch(blocks[i], h)
            h = F.relu(h)
            e_list = e_list + e
            h = F.dropout(h, p=self.dropout, training=self.training)
        logits, e = self.gat_layers[-1].forward_batch(blocks[-1], h)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        e_list = e_list + e
        return logits, e_list

    def reset_params(self):
        for layer in self.gat_layers:
            layer.reset_parameters()

class GAT(nn.Module):
    def __init__(self, args, heads, activation):
        super(GAT, self).__init__()
        #self.g = g
        self.num_layers = args.GAT_args['num_layers']
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            args.d_data, args.GAT_args['num_hidden'], heads[0],
            args.GAT_args['feat_drop'], args.GAT_args['attn_drop'], args.GAT_args['negative_slope'], False, None))
        # self.norm_layers.append(nn.BatchNorm1d(num_hidden*heads[0]))
        self.norm_layers.append(PairNorm())
        
        # hidden layers
        for l in range(1, args.GAT_args['num_layers']):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                args.GAT_args['num_hidden'] * heads[l-1], args.GAT_args['num_hidden'], heads[l],
                args.GAT_args['feat_drop'], args.GAT_args['attn_drop'], args.GAT_args['negative_slope'], args.GAT_args['residual'], self.activation))
            # self.norm_layers.append(nn.BatchNorm1d(num_hidden*heads[l]))
            self.norm_layers.append(PairNorm())
        # output projection

        self.gat_layers.append(GATConv(
            args.GAT_args['num_hidden'] * heads[-2], args.n_cls, heads[-1],
            args.GAT_args['feat_drop'], args.GAT_args['attn_drop'], args.GAT_args['negative_slope'], args.GAT_args['residual'], None))

    def forward(self, g, inputs, save_logit_name = None):
        h = inputs
        e_list = []
        for l in range(self.num_layers):
            h, e = self.gat_layers[l](g, h)
            h = h.flatten(1)
            h = self.activation(h)
            e_list = e_list + e
        # store for ergnn
        self.second_last_h = h
        # output projection
        logits, e = self.gat_layers[-1](g, h)
        #self.second_last_h = logits if len(self.gat_layers) == 1 else h
        logits = logits.mean(1)
        e_list = e_list + e
        return logits, e_list

    def forward_batch(self, blocks, features):
        e_list = []
        h = features
        for i,layer in enumerate(self.gat_layers[:-1]):
            h, e = layer.forward_batch(blocks[i], h)
            h = h.flatten(1)
            h = self.activation(h)
            e_list = e_list + e
        logits, e = self.gat_layers[-1].forward_batch(blocks[-1], h)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        logits = logits.mean(1)
        e_list = e_list + e
        return logits, e_list


    def reset_params(self):
        for layer in self.gat_layers:
            layer.reset_parameters()

class SGC_Agg(nn.Module):
    # only the neighborhood aggregation of SGC
    def __init__(self, k=1, cached=False, norm=None, allow_zero_in_degree=False):
        super().__init__()
        self._cached = cached
        self._cached_h = None
        self._k = k
        self.norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, ')

            if self._cached_h is not None:
                feat = self._cached_h
            else:
                # ipdb.set_trace()
                # compute normalization
                degs = graph.in_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                norm = norm.to(feat.device).unsqueeze(1)
                # compute (D^-1 A^k D)^k X
                for _ in range(self._k):
                    feat = feat * norm
                    graph.ndata['h'] = feat
                    graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                    feat = graph.ndata.pop('h')
                    feat = feat * norm

                if self.norm is not None:
                    feat = self.norm(feat)

                # cache feature
                if self._cached:
                    self._cached_h = feat
            # return self.fc(feat)
            return feat

    def forward_batch(self, blocks, feat):
        if self._k != len(blocks):
            raise DGLError('The depth of the dataloader sampler is incompatible with the depth of SGC')
        for block in blocks:
            # ipdb.set_trace()
            with block.local_scope():
                if not self._allow_zero_in_degree:
                    if (block.in_degrees() == 0).any():
                        raise DGLError('There are 0-in-degree nodes in the graph, ')

                if self._cached_h is not None:
                    feat = self._cached_h
                else:
                    # compute normalization
                    degs = block.out_degrees().float().clamp(min=1)
                    norm = th.pow(degs, -0.5)
                    norm = norm.to(feat.device).unsqueeze(1)
                    # compute (D^-1 A^k D)^k X
                    feat = feat * norm
                    block.srcdata['h'] = feat
                    block.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                    feat = block.dstdata.pop('h')
                    degs = block.in_degrees().float().clamp(min=1)
                    norm = th.pow(degs, -0.5)
                    norm = norm.to(feat.device).unsqueeze(1)
                    feat = feat * norm

        with blocks[-1].local_scope():
            if self.norm is not None:
                feat = self.norm(feat)

            # cache feature
            if self._cached:
                self._cached_h = feat

        # return self.fc(feat)
        return feat

class SGC(nn.Module):
    def __init__(self, args):
        super(SGC, self).__init__()
        linear_layer = linear_choices[args.SGC_args['linear']]
        self.bn = args.SGC_args['batch_norm']
        self.dropout = args.SGC_args['dropout']
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.gpu = args.gpu
        self.neighbor_agg = SGC_Agg(k=args.SGC_args['k'])
        self.feat_trans_layers = nn.ModuleList()
        if self.bn:
            self.bns = nn.ModuleList()
        h_dims = args.SGC_args['h_dims']
        args.hidden = h_dims[-1]
        
        if len(h_dims) > 0:
            self.feat_trans_layers.append(linear_layer(args.d_data, h_dims[0], bias=args.SGC_args['linear_bias']))
            if self.bn:
                self.bns.append(nn.BatchNorm1d(h_dims[0]))
            for i in range(len(h_dims) - 1):
                self.feat_trans_layers.append(linear_layer(h_dims[i], h_dims[i + 1], bias=args.SGC_args['linear_bias']))
                if self.bn:
                    self.bns.append(nn.BatchNorm1d(h_dims[i + 1]))
            self.feat_trans_layers.append(linear_layer(h_dims[-1], h_dims[-1], bias=args.SGC_args['linear_bias']))
        elif len(h_dims) == 0:
            self.feat_trans_layers.append(linear_layer(args.d_data, h_dims[-1], bias=args.SGC_args['linear_bias']))
        else:
            raise ValueError('no valid MLP dims are given')

    def forward(self, graph, x):
        # ipdb.set_trace()
        graph = graph.local_var().to('cuda:{}'.format(self.gpu))
        x = self.neighbor_agg(graph, x)
        logits = self.feat_trans(x)
        return logits

    def forward_batch(self, blocks, x, se, twp=False, tasks=None):
        #graph = graph.local_var().to('cuda:{}'.format(self.gpu))
        e_list = []
        
        x = self.neighbor_agg.forward_batch(blocks, x)
        logits, e = self.feat_trans(blocks[0], x, twp=twp, cls=tasks)
        e_list = e_list + e
        return logits, e_list

    def feat_trans(self, x):
        for i, layer in enumerate(self.feat_trans_layers[:-1]):
            x = layer(x)
            if self.bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        beforecls = x
        x = self.feat_trans_layers[-1](x)
        return x
        # return x, beforecls
        #return x.log_softmax(dim=-1), elist
    

    def reset_params(self):
        for layer in self.feat_trans_layers:
            layer.reset_parameters()