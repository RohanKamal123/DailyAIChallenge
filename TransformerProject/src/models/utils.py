import torch

def generate_padding_mask(seq, pad_token=0):
    return (seq != pad_token).unsqueeze(1).unsqueeze(2)

def generate_lookahead_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1) == 0
    return mask.unsqueeze(0).unsqueeze(0)