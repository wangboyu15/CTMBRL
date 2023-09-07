import numpy as np
import torch
from typing import List, Tuple
from scipy.integrate import odeint
import opt_einsum as oe
contract = oe.contract


class Environment:

    def __init__(
            self,
            S0: List[float],
            Q0: List[float],
            d_asset: int = 3,
            batch_size: int = 64,
            total_time: float = 1/250,
            num_time_interval: int = 100,
            b: float = 1.0
    ):

        self.d_asset = d_asset
        self.total_time = total_time
        self.num_time_interval = num_time_interval
        self.delta_t = (self.total_time + 0.0) / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)

        assert len(S0) == d_asset
        assert len(Q0) == d_asset
        self.S0 = np.array(S0)  # (d_asset, )
        self.S0_torch = torch.FloatTensor(self.S0).reshape(1, -1, 1)  # (1, d_asset, 1)
        self.batch_size = batch_size

        self.sigma = (np.linspace(0.3, 0.5, self.d_asset) * self.S0).reshape(-1, 1)  # return vol range
        '''self.sigma.shape = (d_asset, 1)'''
        # print(f"self.sigma={self.sigma}")


        def gen_rho(d_asset: int, rho_low: float=0.05, rho_high: float=0.6):
            num_pair = int(round(d_asset * (d_asset - 1) / 2))
            for rho in np.linspace(rho_high, rho_low, num_pair):
                yield rho


        '''self.corr_mat.shape = (d_asset, d_asset)'''
        self.corr_mat = np.diag(np.ones(self.d_asset))
        rho_generator = gen_rho(self.d_asset)
        for i in range(self.d_asset):
            for j in range(i + 1, self.d_asset):
                rho = next(rho_generator)
                self.corr_mat[i, j] = rho
                self.corr_mat[j, i] = rho
        # print(f"self.corr_mat={self.corr_mat}")

        self.sigma_torch = torch.FloatTensor(self.sigma)  # (d_asset, 1)
        self.sigma_corr_Lower = np.linalg.cholesky(self.corr_mat) * self.sigma  # (d_asset, d_asset)
        self.sigma_corr_Lower_torch = torch.FloatTensor(self.sigma_corr_Lower)  # (d_asset, d_asset)

        self.Sigma_true_mat = self.sigma_corr_Lower @ self.sigma_corr_Lower.T  # var-cov matrix,  (d_asset, d_asset)
        # print(f"self.Sigma_true_mat={self.Sigma_true_mat}")


        kappa_true = np.linspace(1.5, 3.5, self.d_asset) * 1e-7
        # kappa_true = np.linspace(2.0, 3.0, self.d_asset) * 1e-7
        self.kappa_true_mat = np.diag(kappa_true)   # (d_asset, d_asset)
        if self.d_asset > 1:
            self.kappa_true_mat[0, 1] = 0.8 * kappa_true[0]
            self.kappa_true_mat[1, 0] = 0.8 * kappa_true[0]
        # print(f"self.kappa_true_mat={self.kappa_true_mat}")
        self.kappa_true_mat_torch = torch.FloatTensor(self.kappa_true_mat)   # (d_asset, d_asset)

        self.alpha_mat = 1000 * self.kappa_true_mat  # (d_asset, d_asset)
        self.alpha_mat_torch = torch.FloatTensor(self.alpha_mat)   # (d_asset, d_asset)

        eta_true = np.linspace(0.5, 1.5, self.d_asset) * 1e-8
        self.eta_true_mat = np.diag(eta_true)   # (d_asset, d_asset)
        self.eta_true_inv = np.linalg.inv(self.eta_true_mat)   # (d_asset, d_asset)

        self.eta_true_mat_torch = torch.FloatTensor(self.eta_true_mat)   # (d_asset, d_asset)
        self.eta_true_inv_torch = torch.FloatTensor(self.eta_true_inv)   # (d_asset, d_asset)

        self.Lambda_x = 3
        self.Lambda = 10 ** self.Lambda_x * 1e-8  # =1e-5
        # print(f"self.Lambda={self.Lambda}")

        '''Use shares'''
        self.q0 = np.array(Q0).reshape(-1, 1)  # (d_asset, 1), in shares
        self.q0_norm = np.mean(self.q0)
        self.qt = np.ones((self.d_asset, self.num_time_interval + 1))  # (d_asset, num_time_interval + 1)
        self.qt[:, 0] = self.q0[:, 0]

        self.mut = np.zeros((self.d_asset, self.num_time_interval + 1))  # (d_asset, num_time_interval + 1)
        ''''''

        self.Ct = self.cal_Ct()  # (num_time_interval + 1, d_asset, d_asset)
        # print(f"C0={self.Ct[0]}")
        self.b = b
        twoeta_inv = self.eta_true_inv / 2
        self.Bt_list = [self.b * twoeta_inv @ c for c in self.Ct]

        # minus_two_eta_inv = - self.eta_true_inv / 2    # Now it's correct
        for i in range(self.num_time_interval):
            # mut = minus_two_eta_inv @ self.Ct[i] @ self.qt[:, i]

            mut = -self.Bt_list[i] @ self.qt[:, i]

            self.mut[:, i] = mut
            self.qt[:, i + 1] = self.qt[:, i] + mut * self.delta_t

        # import matplotlib.pyplot as plt
        # plt.figure(dpi=300)
        # for d in range(self.d_asset):
        #     plt.plot((self.qt[d, :] / self.qt[d, 0]).tolist(), label=f"asset {d + 1}", linewidth=0.8)
        # plt.hlines(y=0.0, xmin=0, xmax=self.num_time_interval, ls='--', colors='r', linewidth=0.5)
        # plt.legend()
        # plt.savefig(f"Inventory_{self.d_asset}d_Lam{self.Lambda}_b{int(self.b)}.pdf")
        # plt.show()
        # plt.close()

        self.qt_tensor = torch.FloatTensor(self.qt)  # (d_asset, num_time_interval + 1)
        self.mut_tensor = torch.FloatTensor(self.mut)   # (d_asset, num_time_interval + 1)

        self.permImpact_tensor = torch.FloatTensor(self.kappa_true_mat @ (self.qt - self.q0)).unsqueeze(0)
        # (1, d_asset, num_time_interval + 1), self.permImpact_tensor[0, :, 0] == 0
        # print(f"self.permImpact_tensor.shape={self.permImpact_tensor.shape}")
        # print(f"self.permImpact_tensor={self.permImpact_tensor}")

        self.qtSigmaqt = np.zeros(self.num_time_interval + 1)  # (self.num_time_interval + 1, )
        for i in range(self.num_time_interval + 1):
            self.qtSigmaqt[i] = self.qt[:, i] @ self.Sigma_true_mat @ self.qt[:, i] / self.q0_norm
            # print(f"Lambda*qtSigmaqt[{i}]={self.Lambda * self.qtSigmaqt[i]}, {self.Lambda * np.sum((self.sigma_corr_Lower.T @ self.qt[:, i])**2) / self.q0_norm}")

        self.alpha_qT_torch = torch.FloatTensor(self.alpha_mat @ self.qt[:, -1]).unsqueeze(0)  # (1, d_asset)

    def matrix_ode_solver(self, C: np.array, t: float) -> np.array:
        '''
        :param C: np.array (d_asset * d_asset, )
        :param t:
        :return: np.array (d_asset * d_asset, )
        '''
        C = C.reshape(self.d_asset, self.d_asset)
        Cdot = -C.dot(self.eta_true_inv).dot(C) / 2 + 2 * self.Lambda * self.Sigma_true_mat
        return Cdot.flatten()

    def cal_Ct(self) -> np.array:
        t0 = 0.0
        y0 = 2 * self.alpha_mat - self.kappa_true_mat
        y0 = y0.flatten()  # (d_asset * d_asset, )

        # Time vector
        t = np.linspace(t0, self.total_time, num=self.num_time_interval + 1)

        # Solve the ODE using odeint
        sol = odeint(self.matrix_ode_solver, y0, t)
        sol = sol.reshape((self.num_time_interval + 1, self.d_asset, self.d_asset))
        sol = sol[::-1]     # (num_time_interval + 1, d_asset, d_asset)
        return sol

    def sample(self, num_sample: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        :param num_sample: batch saize
        :return:
        reward: (num_sample, num_time_interval + 1), last columns is terminal payoff
        S_train: (num_sample, d_asset, num_time_interval + 1)
        self.qt_tensor: (d_asset, num_time_interval + 1)
        '''
        dw_train = self.sqrt_delta_t * \
                   self.sigma_corr_Lower_torch @ torch.normal(
            mean=0.0,
            std=1.0,
            size=(num_sample, self.d_asset, self.num_time_interval)
        )

        S_train = self.S0_torch + torch.cat(
            (
                torch.zeros(num_sample, self.d_asset, 1),
                torch.cumsum(dw_train, dim=2)
            ), dim=2
        ) + self.permImpact_tensor


        reward = self.r_t(S_train) * self.delta_t  # (num_sample, num_time_interval + 1), already divided by self.q0_norm
        reward[:, -1] = self.g_T(S_train[:, :, -1]) / self.q0_norm  # (num_sample, )

        return reward, S_train, self.qt_tensor

    def r_t(self, S: torch.Tensor) -> torch.Tensor:
        '''
        :param S: (num_sample, d_asset, num_time_interval + 1)
        :return: (num_sample, num_time_interval + 1)

        self.eta_true_mat_torch: (d_asset, d_asset)
        self.mut_tensor: (d_asset, num_time_interval + 1)
        self.eta_true_mat_torch @ self.mut_tensor: (d_asset, num_time_interval + 1)
        self.qtSigmaqt: (self.num_time_interval + 1, )
        '''
        S_temp = S + (self.eta_true_mat_torch @ self.mut_tensor).unsqueeze(0)  # (num_sample, d_asset, num_time_interval + 1)
        exec_rev = contract('ijk,jk -> ik', S_temp, -self.mut_tensor)  # (num_sample, num_time_interval + 1)
        return exec_rev / self.q0_norm - (torch.FloatTensor(self.Lambda * self.qtSigmaqt)).unsqueeze(0)
        # self.qtSigmaqt already divided by self.q0_norm

    def g_T(self, S: torch.Tensor) -> torch.Tensor:
        '''
        :param S: (num_sample, d_asset)
        :return: (num_sample, )
        '''
        return (S - self.alpha_qT_torch) @ self.qt_tensor[:, -1]