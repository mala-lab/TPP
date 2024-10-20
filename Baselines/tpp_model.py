import torch
import copy
import ipdb
from torch import Tensor
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn.inits import glorot
from Backbones.gnns import SGC_Agg
from Baselines.grace import ModelGrace, traingrace, LogReg


class SimplePrompt(nn.Module):
    def __init__(self, in_channels: int):
        super(SimplePrompt, self).__init__()
        self.global_emb = nn.Parameter(torch.Tensor(1, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.global_emb)

    def add(self, x: Tensor):
        return x + self.global_emb
    
class GPFplusAtt(nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(GPFplusAtt, self).__init__()
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, x: Tensor):
        score = self.a(x)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)
        return x + p


class NET(torch.nn.Module):
    def __init__(self, model, task_manager, args):
        super(NET, self).__init__()
        self.task_manager = task_manager
        self.n_tasks = args.n_tasks
        self.model = model
        self.drop_edge = args.tpp_args['pe']
        self.drop_feature = args.tpp_args['pf']
        num_promt = int(args.tpp_args['prompts'])
        if num_promt < 2:
            prompt = SimplePrompt(args.d_data).cuda()
        else:
            prompt = GPFplusAtt(args.d_data, num_promt).cuda()

        cls_head = LogReg(args.hidden, args.n_cls_per_task).cuda()
        self.classifications = ModuleList([copy.deepcopy(cls_head) for _ in range(args.n_tasks)])
        self.prompts = ModuleList([copy.deepcopy(prompt) for _ in range(args.n_tasks-1)])

        self.optimizers = []
        for taskid in range(args.n_tasks):
            model_param_group = []
            if taskid == 0:
                model_param_group.append({"params":self.classifications[taskid].parameters()})
            else:
                model_param_group.append({"params": self.prompts[taskid-1].parameters()})
                model_param_group.append({"params": self.classifications[taskid].parameters()})
            self.optimizers.append(torch.optim.Adam(model_param_group, lr=args.lr, weight_decay=args.weight_decay))
        
        self.ce = torch.nn.functional.cross_entropy

    def getprototype(self, g, features, train_ids, k=3):
        g = addedges(g)
        neighbor_agg = SGC_Agg(k=k)
        features = neighbor_agg(g, features)
        degs = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5)
        norm = norm.to(features.device).unsqueeze(1)
        features = features * norm
        prototype = torch.mean(features[train_ids], dim=0)
        return prototype

    def gettaskid(self, prototypes, g, features, task, test_ids, k=3):
        g = addedges(g)
        neighbor_agg = SGC_Agg(k=k)
        features = neighbor_agg(g, features)
        degs = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5)
        norm = norm.to(features.device).unsqueeze(1)
        features = features * norm
        testprototypes = torch.mean(features[test_ids], dim=0)
        testprototypes = testprototypes.cpu()
        dist = torch.norm(prototypes[0:task] - testprototypes, dim=1)
        _, taskid = torch.min(dist, dim=0)
        return taskid.numpy()        
    
    def pretrain(self, args, g, features, batch_size = None):
        num_hidden = args.hidden
        num_proj_hidden = 2 * num_hidden
        gracemodel = ModelGrace(self.model, num_hidden, num_proj_hidden, tau=0.5).cuda()
        traingrace(gracemodel, g, features, batch_size, drop_edge_prob = self.drop_edge, drop_feature_prob = self.drop_feature)
        
    def observe_il(self, g, features, labels, t, train_ids, ids_per_cls, offset1, dataset):
        self.model.eval()
        labels = labels - offset1
        cls_head = self.classifications[t]
        cls_head.train()
        cls_head.zero_grad()
        optimizer_t = self.optimizers[t]
        if t > 0:
            prompt_t = self.prompts[t-1]
            prompt_t.train()
            prompt_t.zero_grad()
            features = prompt_t.add(features)
        output = self.model(g, features)
        output = cls_head(output)
        loss = self.ce(output[train_ids], labels[train_ids])
        loss.backward()
        optimizer_t.step() 

    def getpred(self, g, features, taskid):
        self.model.eval()
        if taskid == 0:
            output = self.model(g, features)
            cls_head = self.classifications[0]
            output = cls_head(output)
        else:
            prompt_t = self.prompts[taskid-1]
            features = prompt_t.add(features)
            output = self.model(g, features)
            cls_head = self.classifications[taskid]
            output = cls_head(output)
        return output      



def addedges(subgraph):
    subgraph = copy.deepcopy(subgraph)
    nodedegree = subgraph.in_degrees().cpu()
    isolated_nodes = torch.where(nodedegree==1)[0]
    connected_nodes = torch.where(nodedegree!=1)[0]
    isolated_nodes = isolated_nodes.numpy()
    connected_nodes = connected_nodes.numpy()
    randomnode = np.random.choice(connected_nodes, isolated_nodes.shape[0])
    srcs = np.concatenate([isolated_nodes, randomnode])
    dsts = np.concatenate([randomnode, isolated_nodes])
    subgraph.add_edges(srcs, dsts)
    return subgraph