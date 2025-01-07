from operator import itemgetter

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectralrl.algo.state.ctrl_sac.agent import Ctrl_SAC
from spectralrl.utils.utils import convert_to_tensor, make_target, sync_target


class Speder_SAC(Ctrl_SAC):
    def __init__(
        self,
        obs_dim,
        action_dim,
        cfg,
        device
    ) -> None:
        super().__init__(obs_dim, action_dim, cfg, device)

    def train_step(self, buffer, batch_size):
        tot_metrics = {}

        for _ in range(self.feature_update_ratio):
            batch = buffer.sample(batch_size)
            obs, action, next_obs, reward, terminal = [
                convert_to_tensor(b, self.device) for b in itemgetter("obs", "action", "next_obs", "reward", "terminal")(batch)
            ]
            batch_random = buffer.sample(batch_size)
            obs_random, action_random, next_obs_random, *_ = [
                convert_to_tensor(b, self.device) for b in itemgetter("obs", "action", "next_obs")(batch_random)
            ]
            feature_loss, feature_metrics = self.feature_step(obs, action, next_obs, reward, next_obs_random)
            tot_metrics.update(feature_metrics)
            self.feature_optim.zero_grad()
            feature_loss.backward()
            self.feature_optim.step()
            sync_target(self.phi, self.phi_target, self.tau)

        critic_loss, critic_metrics = self.critic_step(obs, action, next_obs, reward, terminal)
        tot_metrics.update(critic_metrics)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss, actor_metrics = self.actor_step(obs)
        tot_metrics.update(actor_metrics)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self.auto_entropy:
            alpha_loss, alpha_metrics = self.alpha_step(obs)
            tot_metrics.update(alpha_metrics)
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

        if self._step % self.target_update_freq == 0:
            sync_target(self.critic, self.critic_target, self.tau)

        self._step += 1
        return tot_metrics

    def feature_step(self, obs, action, next_obs, reward, next_obs_random):
        z_phi = self.phi(obs, action)
        z_mu = self.mu(next_obs)
        z_mu_random = self.mu(next_obs_random)

        B, D = z_phi.shape

        model_loss_pt1 = -2 * (z_phi * z_mu).sum(-1).mean()
        model_loss_pt2 = 1 / D * (z_mu_random * z_mu_random).sum(-1).mean()
        model_loss = model_loss_pt1 + model_loss_pt2
        # TODO: no regularization term for now

        reward_loss = F.mse_loss(self.theta(z_phi), reward)
        feature_loss = model_loss + self.reward_coef * reward_loss
        return feature_loss, {
            "loss/feature_loss": feature_loss.item(),
            "loss/model_loss": model_loss.item(),
            "loss/reward_loss": reward_loss.item(),
            "loss/model_loss_pt1": model_loss_pt1.item(),
            "loss/model_loss_pt2": model_loss_pt2.item(),
        }
