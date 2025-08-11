import torch
import torch.nn as nn
from models.transformer import Transformer
from models.utils import generate_padding_mask, generate_lookahead_mask

def train():
    model = Transformer(src_vocab_size=10000, tgt_vocab_size=10000)
    # Add dataset, loss, optimizer, and training loop later
    print("Training setup ready!")
if __name__ == "__main__":
    train()