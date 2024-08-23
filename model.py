# Image Captioning Class

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torch
import math




class TransformerImageCaptioning(nn.Module):

    class SinusoidalPositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000):
            super(TransformerImageCaptioning.SinusoidalPositionalEncoding, self).__init__()
            self.encoding = self._generate_positional_encoding(d_model, max_len)
            
        def _generate_positional_encoding(self, d_model, max_len):
            # Initialize a matrix to hold the positional encodings
            positional_encodings = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
            
            # Compute sinusoidal functions
            positional_encodings[:, 0::2] = torch.sin(position * div_term)
            positional_encodings[:, 1::2] = torch.cos(position * div_term)
            
            # Add batch dimension
            positional_encodings = positional_encodings.unsqueeze(0)
            return positional_encodings
        
        def forward(self, x):
            # x is expected to be of shape (batch_size, seq_len, d_model)
            batch_size, seq_len, _ = x.size()
            
            # Return the positional encodings for the current batch size and sequence length
            return self.encoding[:, :seq_len].to(x.device)


    def __init__(self, 
                 patch_dim, 
                 d_model, 
                 num_heads, 
                 num_layers, 
                 vocab_size, 
                 max_seq_len, 
                 dropout=0.1,
                 *args,
                 **kwargs):
        super(TransformerImageCaptioning, self).__init__()
        
        # Define the positional encoding layer
        self.positional_encoding = self.SinusoidalPositionalEncoding(d_model, max_seq_len)
        
        # Define the embedding layers for image patches and decoder tokens
        self.patch_embedding = nn.Linear(patch_dim, d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )
        
        # Define the output linear layer to map to vocabulary size
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, patches, decoder_input, mask):
        # Embed patches
        patch_embeddings = self.patch_embedding(patches)          

        # Embed decoder input tokens
        decoder_embeddings = self.token_embedding(decoder_input)  

        # Create positional encodings for decoder input
        decoder_positional_encoding = self.positional_encoding(decoder_embeddings)
        decoder_embeddings = decoder_embeddings + decoder_positional_encoding
        
        # Transformer expects inputs in (batch_size, seq_len, features)
        transformer_output = self.transformer(patch_embeddings, decoder_embeddings, tgt_mask=mask)
        
        # Map output to vocabulary size
        output = self.fc_out(transformer_output)  # Shape: (batch_size, seq_len, vocab_size)
        
        return output
