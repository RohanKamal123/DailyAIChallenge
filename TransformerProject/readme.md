# Transformer from Scratch
Built a Transformer model from scratch in PyTorch . Implements the full architecture (multi-head attention, positional encoding, encoder/decoder) and includes an attention heatmap visualization.

## Project Overview
This project demonstrates a sequence-to-sequence Transformer for tasks like translation or Q&A. It’s tested with synthetic data and visualizes attention weights to show how tokens interact in the encoder.

![Attention Heatmap](outputs/plots/attention_heatmap.png)

## Features
- **Full Implementation**: Positional Encoding, Scaled Dot-Product Attention, Multi-Head Attention, Feed-Forward Networks, Encoder/Decoder Layers.
- **Visualization**: Attention heatmap for encoder self-attention.
- **Modular Code**: Organized in `src/models/` for reusability.
- **Tested**: Forward pass outputs `torch.Size([8, 8, 1000])` on synthetic data.

## Setup
1. Clone the repo:
   ```bash
   git clone https://github.com/RohanKamal123/DailyAIChallenge.git
   cd TransformerProject