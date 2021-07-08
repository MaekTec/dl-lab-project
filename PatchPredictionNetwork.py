import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


def prob_mask_like(input, prob):
    batch_size, seq_length, _ = input.shape
    x = torch.zeros((batch_size, seq_length)).float().uniform_(0, 1)
    x = x < prob
    return x


def get_mask_token_prob(input, mask_prob):
    '''
    Returns mask_indices (Bool): Containes the Bool value of whether to apply masked token on image patch
    '''

    batch_size,seq_len , n_sequences , device = *input.shape, input.device
    n_max_masked = math.ceil(mask_prob * seq_len) # the max number of masked images in a sequences

    rand = torch.rand((batch_size, seq_len), device=device)
    _, sampled_indices = rand.topk(n_max_masked, dim=-1) # sample random indices to be masked

    mask_indices = torch.zeros((batch_size, seq_len), device=device)
    mask_indices.scatter_(1, sampled_indices, True)

    return mask_indices.bool()

class PatchPredictionNetwork(nn.Module):

    def __init__(
            self,
            transformer,
            patch_size,
            dim,
            output_channel_bits=3,
            channels=3,
            max_pixel_val=1.0,
            mask_prob=0.15,
            replace_prob=0.5,
            random_patch_prob=0.5,
            mean=None,
            std=None
    ):
        super().__init__()
        self.transformer = transformer #ViT transformer

        #output transformation
        self.to_bits = nn.Linear(dim, 2**(output_channel_bits * channels))

        # vit related dimensions
        self.patch_size = patch_size

        # mpp related probabilities
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.random_patch_prob = random_patch_prob

        # token ids
        self.mask_token = nn.Parameter(torch.randn(1, 1, channels * patch_size ** 2))

    def forward(self, input, **kwargs):

        #keep copy of original image.. used for Loss
        img = input.clone().detach()

        #reshaping of images into patches
        p = self.patch_size
        input = rearrange(input,'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',p1=p,p2=p)
        # breaks down each image into patch sequence
        # (n,3,64,64) -> (n, 16, 768)

        mask = get_mask_token_prob(input, self.mask_prob)

        # Random prob, not implemented.. todo

        # [mask] input
        replace_prob = prob_mask_like(input, self.replace_prob).to(mask.device)










        pass






