import torch

def generate_padding_mask(seq, pad_token=0):
    return (seq != pad_token).unsqueeze(1).unsqueeze(2)

def generate_lookahead_mask(size):
    # Create a more efficient and gradient-friendly lookahead mask
    # Use torch.tril instead of torch.triu for better performance
    mask = torch.tril(torch.ones(size, size, dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0)