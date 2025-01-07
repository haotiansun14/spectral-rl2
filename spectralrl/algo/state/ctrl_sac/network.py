import torch
import torch.nn as nn
import torch.nn.functional as F

from spectralrl.module.net.mlp import MLP


class Phi(nn.Module):
    def __init__(
        self,
        feature_dim,
        obs_dim,
        action_dim,
        hidden_dims,
    ) -> None:
        super().__init__()
        self.mlp = MLP(obs_dim+action_dim, feature_dim, hidden_dims, activation=nn.ELU)

    def forward(self, obs, action, dt=None):
        out = self.mlp(torch.concat([obs, action], dim=-1))
        return out


class Mu(nn.Module):
    def __init__(
        self,
        feature_dim,
        obs_dim,
        hidden_dims,
    ) -> None:
        super().__init__()
        self.mlp = MLP(obs_dim, feature_dim, hidden_dims, activation=nn.ELU)

    def forward(self, obs):
        out = self.mlp(obs)
        out = F.tanh(out)
        return out


class Theta(nn.Module):
    def __init__(
        self,
        feature_dim
    ) -> None:
        super().__init__()
        self.mlp = nn.Linear(feature_dim, 1)

    def forward(self, feature):
        return self.mlp(feature)


class RFFCritic(nn.Module):
    def __init__(self,
        feature_dim: int,
        hidden_dim: int
    ):
        super().__init__()
        self.l1 = nn.Linear(feature_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        self.l4 = nn.Linear(feature_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, x, return_feature=False):
        q1_feature, q2_feature = self.forward_feature(x)
        q1 = self.l3(q1_feature)
        q2 = self.l6(q2_feature)
        return torch.stack([q1, q2], dim=0)

    def forward_feature(self, x):
        q1 = torch.sin(self.l1(x))
        q1 = torch.nn.functional.elu(self.l2(q1))

        q2 = torch.sin(self.l4(x))
        q2 = torch.nn.functional.elu(self.l5(q2))
        return q1, q2
