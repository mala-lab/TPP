import torch
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import ipdb

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


def drop_feature(x, drop_prob):
    drop_mask = torch.empty((x.size(1), ), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x

def mask_edge(graph, drop_prob):
    graph = copy.deepcopy(graph)
    num_edges = graph.number_of_edges()
    edge_delete = np.random.choice(num_edges, int(drop_prob*num_edges), replace=False)
    src, dst = graph.edges()
    not_equal = src[edge_delete].cpu() != dst[edge_delete].cpu()
    edge_delete = edge_delete[not_equal]
    graph.remove_edges(edge_delete)
    return graph


class ModelGrace(nn.Module):
    def __init__(self, model, num_hidden, num_proj_hidden, tau=0.5):
        super(ModelGrace, self).__init__()
        self.model = model
        self.tau = tau
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, graph, features):
        output = self.model(graph, features)
        Z = F.elu(self.fc1(output))
        Z = self.fc2(Z)
        return Z

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1, z2, batch_size):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []
        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]
            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
            torch.cuda.empty_cache()
        return torch.cat(losses)

    def loss(self, h1, h2, batch_size):
        if batch_size is None:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)
        ret = (l1 + l2) * 0.5
        ret = ret.mean()
        return ret

def traingrace(modelgrace, graph, features, batch_size = None, drop_edge_prob = 0.2, drop_feature_prob = 0.3, epochs = 200, lr = 1e-3):
    modelgrace.train()
    optimizer = torch.optim.Adam(modelgrace.parameters(), lr=lr, weight_decay=1e-5)

    for _ in range(epochs):
        optimizer.zero_grad()
        graph_aug = mask_edge(graph, drop_edge_prob)
        features_aug = drop_feature(features, drop_feature_prob)
        Z1 = modelgrace(graph, features)
        Z2 = modelgrace(graph_aug, features_aug)
        loss = modelgrace.loss(Z1, Z2, batch_size=batch_size)
        loss.backward()
        optimizer.step()

