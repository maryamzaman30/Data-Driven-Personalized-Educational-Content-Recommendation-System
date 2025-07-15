# File: src/recommender/ncf.py

import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, hidden_dims=[64, 32, 16]):
        super(NCF, self).__init__()
        self.user_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_gmf = nn.Embedding(num_items, embedding_dim)
        self.user_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_mlp = nn.Embedding(num_items, embedding_dim)

        layers = []
        input_dim = 2 * embedding_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            input_dim = dim

        self.mlp = nn.Sequential(*layers)
        self.final = nn.Linear(embedding_dim + hidden_dims[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_ids, item_ids):
        gmf_output = self.user_gmf(user_ids) * self.item_gmf(item_ids)
        mlp_input = torch.cat([self.user_mlp(user_ids), self.item_mlp(item_ids)], dim=-1)
        mlp_output = self.mlp(mlp_input)
        output = self.final(torch.cat([gmf_output, mlp_output], dim=-1))
        return self.sigmoid(output).squeeze()
