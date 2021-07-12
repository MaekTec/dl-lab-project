import copy
import queue

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoCoNetwork(nn.Module):

    def __init__(self, encoder, encoder_dim, queue_size=int(65536/256), momentum=0.999, softmax_temp=0.07):
        super().__init__()
        self.f_q = encoder
        self.f_k = copy.deepcopy(encoder)
        self.queue = []
        self.encoder_dim = encoder_dim
        self.queue_size = queue_size
        self.momentum = momentum
        self.softmax_temp = softmax_temp

    def forward(self, x):
        # momentum update: key network
        for f_k_param, f_q_param in zip(self.f_k.parameters(), self.f_q.parameters()):
            f_k_param.data.copy_(self.momentum * f_k_param.data + (1.0 - self.momentum) * f_q_param.data)

        x_q, x_k = x
        N = len(x_q)

        q = self.f_q(x_q)  # NxC
        k = self.f_k(x_k).detach()  # NxC

        # positive logits: Nx1
        l_pos = torch.matmul(torch.unsqueeze(q, dim=1), torch.unsqueeze(k, 2)).view(N, 1)  # (N,1,C) @ (N,C,1)

        if self.queue:  # queue not empty?
            # negative logits: NxK
            l_neg = torch.mm(q, torch.cat(self.queue, dim=0).T)  # (N, C) @ (C, K)

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)
        else:
            logits = l_pos

        # contrastive loss, Eqn.(1)
        labels = torch.zeros(N).long().cuda()  # positives are the 0-th

        # update dictionary
        self.queue.append(k)
        if len(self.queue) >= self.queue_size:
            self.queue.pop(0)

        return logits/self.softmax_temp, labels



