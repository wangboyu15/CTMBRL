import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from Execution_MultiAsset.sde import arithmeticBrownianMotion

# from sde import arithmeticBrownianMotion

DTYPE = torch.float32
EPSILON = 1e-6
MOMENTUM = 0.99

class Dense(nn.Module):

    def __init__(
            self,
            dim_in: int,
            dim_out: int,
            batch_norm: bool = False,
            activate: bool = True
    ):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.activate = activate
        self.bn = nn.BatchNorm1d(dim_out, eps=EPSILON, momentum=MOMENTUM) if batch_norm else None
        nn.init.normal_(self.linear.weight, std=5.0 / np.sqrt(dim_in + dim_out))

    def forward(self, x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activate:
            x = F.relu(x)
        return x


class Subnetwork(nn.Module):

    def __init__(self, config, batch_norm: bool = False):
        super().__init__()
        self._config = config
        self.bn = nn.BatchNorm1d(config.dim, eps=EPSILON, momentum=MOMENTUM) if batch_norm else None

        #         self.layers = [
        #             Dense(config.num_hiddens[i-1], config.num_hiddens[i]) for i in range(1, len(config.num_hiddens)-1)
        #         ]
        #         self.layers += [Dense(config.num_hiddens[-2], config.num_hiddens[-1], activate=False)]

        self.layers = [Dense(config.num_neurons[i - 1], config.num_neurons[i], batch_norm=batch_norm, activate=True)
                       for i in range(1, len(config.num_neurons) - 1)]
        self.layers += [Dense(config.num_neurons[-2], config.num_neurons[-1], batch_norm=batch_norm, activate=False)]

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        '''
        :param x: (num_sample, 2 * d_asset + 1)
        :return:  (num_sample, d_asset)
        '''
        if self.bn is not None:
            x = self.bn(x)

        x = self.layers(x)
        return x


class learnValEnv(nn.Module):
    """The fully connected neural network model."""

    def __init__(
            self,
            config,
            S0: List[float],
            Q0: List[float],
            kappa: List[float],
            eta: List[float],
            sigma_lower: List[float],
            d_asset: int = 3,
            batch_size: int = 64,
            total_time: float = 1 / 250,
            num_time_interval: int = 100
    ):
        super().__init__()
        self.config = config
        self.sde = arithmeticBrownianMotion(
            S0=S0,
            Q0=Q0,
            kappa=kappa,
            eta=eta,
            sigma_lower=sigma_lower,
            d_asset=d_asset,
            batch_size=batch_size,
            total_time=total_time,
            num_time_interval=num_time_interval
        )    # trainable

        # make sure consistent with FBSDE equation
        self.d_asset = self.sde.d_asset
        self.num_time_interval = self.sde.num_time_interval
        self.total_time = self.sde.total_time

        self.y_init = 0.0
        self.subnetwork = Subnetwork(config)     # trainable


    '''Only Reverse engineering sigma*dW'''
    def forward(
            self,
            q_env: torch.Tensor,
            S_env: torch.Tensor,
            r_env: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        '''
        :param q_env:  (d_asset, num_time_interval + 1), from env
        :param S_env:  (num_sample, d_asset, num_time_interval + 1), from env
        :param r_env:  (num_sample, num_time_interval + 1), from env
        :return: MLoss: tensor([]), self.y_init: float
        '''

        permImpact_reverseEng = (self.sde.kappa_mat @ (q_env - q_env[:, [0]])).unsqueeze(dim=0) # (1, d_asset, num_time_interval + 1)
        cum_sigma_dw_reverseEng = S_env - permImpact_reverseEng  # (num_sample, d_asset, num_time_interval + 1)
        sigma_dw_reverseEng = torch.diff(cum_sigma_dw_reverseEng, dim=2)  # (num_sample, d_asset, num_time_interval)


        batch_size = S_env.size()[0]
        tau_tensor = torch.repeat_interleave(
            torch.FloatTensor(list(range(self.num_time_interval, -1, -1))).unsqueeze(dim=0).unsqueeze(dim=0),
            batch_size,
            dim=0
        )   # (num_sample, 1, num_time_interval+1)

        q_env_norm = q_env / q_env[:, [0]]  # normalize q
        qt_env_tensor = torch.repeat_interleave(
            q_env_norm.unsqueeze(0),
            batch_size,
            dim=0
        )  # (num_sample, d_asset, num_time_interval+1)

        reward_to_go_env = r_env + torch.sum(r_env, dim=-1, keepdim=True) - torch.cumsum(r_env, dim=-1)
        # (num_sample, num_time_interval + 1)

        y_pred = reward_to_go_env[:, -1]  # y is predicted value, (num_sample, ) terminanl payoff

        NNinput = torch.cat([tau_tensor, qt_env_tensor, S_env - S_env[:, :, [0]]], dim=1)
        # (num_sample, 2*d_asset+1, N+1)
        # subtract S0

        squaredError = torch.zeros(batch_size)  # (num_sample, )

        for i in range(self.num_time_interval - 1, -1, -1):
            '''calculate reward using model (with param theta)'''
            r_bsde = self.sde.r_t(
                mui=(q_env[:, i + 1] - q_env[:, i]) / self.sde.delta_t,  # (d_asset,)
                qi=q_env[:, i],  # (d_asset,)
                S=S_env[:, :, i]  # (num_sample, d_asset)
            )  # return: (num_sample, )
            y_pred = y_pred + r_bsde * self.sde.delta_t

            '''calculate hedge ratio using model (with param theta)'''
            z_pred = self.subnetwork(NNinput[:, :, i]) / self.d_asset  # (num_sample, d_asset)
            y_pred = y_pred - torch.sum(z_pred * sigma_dw_reverseEng[:, :, i], dim=1, keepdim=False)  # (num_sample, )

            '''calculate single step squared error between target and pred'''
            squaredError = squaredError + (reward_to_go_env[:, i] - y_pred) ** 2 * self.sde.delta_t  # (num_sample, )

        MLoss = torch.mean(squaredError)  # tensor([])
        self.y_init = torch.mean(y_pred).item()  # float

        print(f"~~~ J0_env={torch.mean(reward_to_go_env[:, 0]).item()}, J0_pred={self.y_init}")

        return MLoss, self.y_init
