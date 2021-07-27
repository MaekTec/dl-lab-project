import copy
import queue

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
This implementation is very similar to the pseudo code in the the paper.
"""


class MoCoNetwork(nn.Module):
    """
    Attention: queue_size depends on dataset size, the queue should not be larger than the total dataset.
    Otherwise you have negatives which are in truth positives, because all samples in the queue are
    always used as negatives.
    """

    def __init__(self, encoder, encoder_dim, queue_size=150, momentum=0.999, softmax_temp=0.07):
        super().__init__()
        self.f_q = encoder
        self.f_k = copy.deepcopy(encoder)
        self.queue_train = []
        self.queue_val = []
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
        q = nn.functional.normalize(q, dim=1)  # L2-norm
        k = self.f_k(x_k).detach()  # NxC
        k = nn.functional.normalize(k, dim=1)  # L2-norm

        # positive logits: Nx1
        l_pos = torch.matmul(torch.unsqueeze(q, dim=1), torch.unsqueeze(k, 2)).view(N, 1)  # (N,1,C) @ (N,C,1)

        if self.training:
            queue = self.queue_train
        else:
            queue = self.queue_val

        if queue:  # queue not empty?
            # negative logits: NxK
            l_neg = torch.mm(q, torch.cat(queue, dim=0).T)  # (N, C) @ (C, K)

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)
        else:
            logits = l_pos

        # contrastive loss, Eqn.(1)
        labels = torch.zeros(N).long().cuda()  # positives are the 0-th

        # update dictionary
        queue.append(k)
        if len(queue) >= self.queue_size:
            queue.pop(0)

        return logits/self.softmax_temp, labels



