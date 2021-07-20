import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce


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
        self.to_bits = nn.Linear(dim, 2**(output_channel_bits * channels)).cuda()
        #print("self.to_bits=",self.to_bits)

        # vit related dimensions
        self.patch_size = patch_size

        # mpp related probabilities
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.random_patch_prob = random_patch_prob

        # token ids
        self.mask_token = nn.Parameter(torch.randn(1, 1, channels * patch_size ** 2))

    def forward(self, input, **kwargs):
        transformer = self.transformer
        print(transformer)
        #print(input.shape)
        #keep copy of original image.. used for Loss
        img = input.detach().clone()


        #reshaping of images into patches
        p = self.patch_size
        input = rearrange(input,'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',p1=p,p2=p)
        # breaks down each image into patch sequence
        # (n,3,64,64) -> (n, 16, 768)

        masked_input = input.detach().clone() # variable used for storing the masked img

        mask = get_mask_token_prob(input, self.mask_prob)

        # Random prob, not implemented.. todo

        # [mask] input
        replace_prob = prob_mask_like(input, self.replace_prob).to(mask.device)
        bool_mask_replace = (mask * replace_prob)==True
        masked_input[bool_mask_replace] = self.mask_token

        #convert the masked input into a linear embedding of patches
        masked_input = transformer.net.to_patch_embedding[-1](masked_input)

        #append CLS token to start of each sequence
        #CLS tokens are used to denote start of a sequence
        b,n,i = masked_input.shape
        cls_token_arr = repeat(transformer.net.cls_token, '() n d -> b n d', b=b)
        masked_input = torch.cat((cls_token_arr,masked_input), dim=1)

        #add positonal embeddings
        pos_emmbeddings_arr= transformer.net.pos_embedding[:, :(n + 1)]
        masked_input = masked_input + pos_emmbeddings_arr

        # pass masked_image to transformer
        output = transformer.net.transformer(masked_input, **kwargs)

        #print("output shape 1:", output.shape)
        #bring back to image dimension for loss calculation
        #output = self.to_bits(output)
        #print("output shape 2:", output.shape)
        #remove the channel first layer CLS
        output = output[:,1:,:]

        return output , mask

class PatchPredictionLoss(nn.Module):
    def __init__( self,patch_size,channels,output_channel_bits,max_pixel_val,mean,std):
        super(PatchPredictionLoss, self).__init__()
        self.patch_size = patch_size
        self.channels = channels
        self.output_channel_bits = output_channel_bits
        self.max_pixel_val = max_pixel_val

        self.mean = torch.tensor(mean).view(-1, 1, 1) if mean else None
        self.std = torch.tensor(std).view(-1, 1, 1) if std else None

    def forward(self,predicted, target, mask):
        bin_size = self.max_pixel_val / (2 ** self.output_channel_bits)
        device = target.device

        #target = target * self.std + self.mean # denormalise the data

        # reshape target to patches
        target = target.clamp(max = self.max_pixel_val) # clamp just in case
        #target before : 20*3*64*64
        avg_target = reduce(target, 'b c (h p1) (w p2) -> b (h w) c', 'mean',
                            p1 = self.patch_size, p2 = self.patch_size).contiguous()
        #avg target: 20*16*3

        channel_bins = torch.arange(bin_size, self.max_pixel_val, bin_size, device = device)
        discretized_target = torch.bucketize(avg_target, channel_bins)

        bin_mask = (2 ** self.output_channel_bits) ** torch.arange(0, self.channels, device = device).long()
        bin_mask = rearrange(bin_mask, 'c -> () () c')

        target_label = torch.sum(bin_mask * discretized_target, dim = -1)


        loss = F.cross_entropy(predicted[mask], target_label[mask])
        return loss








