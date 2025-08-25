import torch
from src.models.transformer import Transformer
from src.models.utils import generate_padding_mask, generate_lookahead_mask
import seaborn as sns
import matplotlib.pyplot as plt
import os
import math

# Create output directory
os.makedirs("outputs/plots", exist_ok=True)

# Smaller model for debugging
model = Transformer(
    src_vocab_size=1000,
    tgt_vocab_size=1000,
    d_model=64,
    num_heads=4,
    num_layers=1,
    d_ff=256
)

# Input tensors
src = torch.randint(0, 1000, (8, 10))  # batch=8, src_seq=10
tgt = torch.randint(0, 1000, (8, 8))   # batch=8, tgt_seq=8
src_mask = generate_padding_mask(src)
tgt_mask = generate_lookahead_mask(tgt.size(1)) & generate_padding_mask(tgt)

# Forward pass
output = model(src, tgt, src_mask, tgt_mask)
print(f"Output shape: {output.shape}")

# Generate attention heatmap
model.eval()
with torch.no_grad():
    # Use the new method to get attention weights
    attn = model.get_attention_weights(src, src_mask, layer_idx=0)
    sns.heatmap(attn[0, 0].detach().cpu().numpy())
    plt.title("Attention Heatmap (Encoder Self-Attention)")
    plt.savefig("outputs/plots/attention_heatmap.png")
    plt.close()  # Close plot to free memory