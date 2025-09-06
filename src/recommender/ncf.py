# =========================================================
# File: src/recommender/ncf.py
# Description:
#   Implementation of the Neural Collaborative Filtering (NCF) model
#   using PyTorch.
#   Combines:
#     - GMF (Generalized Matrix Factorization) for linear interactions
#     - MLP (Multi-Layer Perceptron) for non-linear interactions
# =========================================================

import torch
import torch.nn as nn

# =========================================================
# 1. Neural Collaborative Filtering (NCF) Model Definition
# =========================================================

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, hidden_dims=[64, 32, 16]):
        super(NCF, self).__init__()

        # Embeddings for Generalized Matrix Factorization (GMF)
        self.user_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_gmf = nn.Embedding(num_items, embedding_dim)

        # Embeddings for Multi-Layer Perceptron (MLP)
        self.user_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_mlp = nn.Embedding(num_items, embedding_dim)

        # Build MLP layers dynamically based on hidden_dims
        layers = []
        input_dim = 2 * embedding_dim  # Concatenated user/item embeddings
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))  # Fully connected layer
            layers.append(nn.ReLU()) # Activation function
            input_dim = dim  # Update input size for next layer

        self.mlp = nn.Sequential(*layers)  # Stack layers into a sequential model

        # Final prediction layer combines GMF and MLP outputs
        self.final = nn.Linear(embedding_dim + hidden_dims[-1], 1)

        # Sigmoid activation to squash output between 0 and 1
        self.sigmoid = nn.Sigmoid()

    # ------------------------
    # Forward Pass
    # ------------------------
    def forward(self, user_ids, item_ids):
        # GMF: element-wise product of user/item embeddings
        gmf_output = self.user_gmf(user_ids) * self.item_gmf(item_ids)

        # MLP: concatenate user/item embeddings and pass through MLP
        mlp_input = torch.cat([self.user_mlp(user_ids), self.item_mlp(item_ids)], dim=-1)
        mlp_output = self.mlp(mlp_input)

        # Combine GMF and MLP outputs, then pass through final layer
        output = self.final(torch.cat([gmf_output, mlp_output], dim=-1))

        # Apply sigmoid and remove extra dimensions
        return self.sigmoid(output).squeeze()