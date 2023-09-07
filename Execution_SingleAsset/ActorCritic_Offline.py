import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from typing import Tuple
from scipy import sparse
from scipy.sparse.linalg import splu
from concurrent.futures import ProcessPoolExecutor

class singleExecFD:
    def __init__(
            self,
            q0: float, S0: float, T: float,
            zeta: float, Lambda: float,
            phi1: float, phi2: float,
            theta1: float, theta2: float, theta3: float,
            alpha_phi1: float, alpha_phi2: float, alpha_theta1: float, alpha_theta2: float, alpha_theta3: float,
            N: int, delta: float, dim: int, h: float,
            Svec: np.array, Smin: float, Smax: float, scheme: str = 'CN'
    ):
        '''
        :param q0: initial inventory in shares
        :param S0: initial stock price
        :param T: total time length in years
        :param zeta: temperature parameter
        :param Lambda: risk-aversion parameter
        :param phi1: policy parameter for mean of Gaussian policy
        :param phi2: policy parameter for variance of Gaussian policy
        :param theta1: env parameter for permanent impact
        :param theta2: env parameter for vol^2
        :param theta3: env parameter for temporary impact
        :param alpha_phi1: learning rate for phi1
        :param alpha_phi2: learning rate for phi2
        :param alpha_theta1: learning rate for theta1
        :param alpha_theta2: learning rate for theta2
        :param alpha_theta3: learning rate for theta3
        :param N: number of time steps
        :param delta: length of each time step
        :param dim: number of space discretization of finite difference
        :param h: length of each space discretization interval
        :param Svec: vector of stock prices on the FD grid
        :param Smin: minimum stock price
        :param Smax: maximum stock price
        :param scheme: finite diff scheme, default is 'CN'
        '''

        try:
            dim > 3
        except:
            print(f'Too small dim={dim}!')

        self.q0 = q0
        self.S0 = S0
        self.T = T
        self.zeta = zeta
        self.Lambda = Lambda
        self.phi1 = phi1
        self.phi2 = phi2
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3
        self.alpha_phi1 = alpha_phi1
        self.alpha_phi2 = alpha_phi2
        self.alpha_theta1 = alpha_theta1
        self.alpha_theta2 = alpha_theta2
        self.alpha_theta3 = alpha_theta3
        self.N = N
        self.delta = delta
        self.h = h
        self.Svec = Svec
        self.Smin = Smin
        self.Smax = Smax
        self.dim = dim
        self.scheme = scheme

        self.vVec_list = [np.zeros(self.dim) for _ in range(self.N + 1)]
        self.grad_theta1Vec_list, self.grad_theta2Vec_list, self.grad_theta3Vec_list = \
            [np.zeros(self.dim) for _ in range(self.N + 1)], \
            [np.zeros(self.dim) for _ in range(self.N + 1)], \
            [np.zeros(self.dim) for _ in range(self.N + 1)]
        self.grad_phi1Vec_list, self.grad_phi2Vec_list = \
            [np.zeros(self.dim) for _ in range(self.N + 1)], \
            [np.zeros(self.dim) for _ in range(self.N + 1)]
        self.identityMat = np.eye(self.dim)

    def nu_t(self, t: float) -> float:
        return -self.q0 * self.phi1 * np.cosh(self.phi1 * (self.T - t)) / np.sinh(self.phi1 * self.T)

    def hyperbolic_util(self, t: float) -> Tuple[float, float, float, float, float, float]:
        phi1_T = self.phi1 * self.T
        phi1_tau = self.phi1 * (self.T - t)
        sinh_phi1_T = np.sinh(phi1_T)
        cosh_phi1_T = np.cosh(phi1_T)
        sinh_phi1_tau = np.sinh(phi1_tau)
        cosh_phi1_tau = np.cosh(phi1_tau)
        return phi1_T, phi1_tau, sinh_phi1_T, cosh_phi1_T, sinh_phi1_tau, cosh_phi1_tau

    def dnut_dphi1(self, t: float) -> float:
        phi1_T, phi1_tau, sinh_phi1_T, cosh_phi1_T, sinh_phi1_tau, cosh_phi1_tau = self.hyperbolic_util(t)
        return -self.q0 * (
                cosh_phi1_tau * (sinh_phi1_T - phi1_T * cosh_phi1_T)
                + phi1_tau * sinh_phi1_T * sinh_phi1_tau
        ) / sinh_phi1_T ** 2

    def q_t(self, t: float) -> float:
        return self.q0 * np.sinh(self.phi1 * (self.T - t)) / np.sinh(self.phi1 * self.T)

    def dqt_dphi1(self, t: float) -> float:
        phi1_T, phi1_tau, sinh_phi1_T, cosh_phi1_T, sinh_phi1_tau, cosh_phi1_tau = self.hyperbolic_util(t)
        return self.q0 * (
                (self.T - t) * cosh_phi1_tau * sinh_phi1_T - self.T * cosh_phi1_T * sinh_phi1_tau
        ) / sinh_phi1_T ** 2

    '''Generate r_theta, dr_dtheta and dr_dphi'''
    def gen_r_theta(self, t: float) -> float:
        nu = self.nu_t(t)
        q = self.q_t(t)
        return -nu * (self.Svec + self.theta3 * nu) - \
               self.theta3 * self.zeta * self.phi2 - \
               self.Lambda * self.theta2 * self.S0 ** 2 * q ** 2 + \
               self.zeta * np.log(2 * np.pi * np.e * self.zeta * self.phi2) / 2

    def gen_dr_dtheta(self, t: float) -> Tuple[float, float, float]:
        nu = self.nu_t(t)
        q = self.q_t(t)
        return 0., \
               - self.Lambda * self.S0 ** 2 * q ** 2 * np.ones(self.dim), \
               -(nu ** 2 + self.zeta * self.phi2) * np.ones(self.dim)

    def gen_dr_dphi(self, t: float) -> Tuple[float, float]:
        nu = self.nu_t(t)
        dnu_dphi1 = self.dnut_dphi1(t)

        q = self.q_t(t)
        dq_dphi1 = self.dqt_dphi1(t)

        dr_dphi1 = -self.Svec * dnu_dphi1 \
                   - 2 * self.theta3 * nu * dnu_dphi1 \
                   - 2 * self.Lambda * self.theta2 * self.S0 ** 2 * q * dq_dphi1
        dr_dphi2 = self.zeta * (1 / 2 / self.phi2 - self.theta3)
        return dr_dphi1, dr_dphi2


    '''Generate D_theta, dD_dtheta, dD_dphi'''
    def gen_D_theta(self, t: float) -> np.array:
        diffusion = self.theta2 * self.S0 ** 2 / (2 * self.h ** 2)
        drift = self.theta1 * self.nu_t(t) / (2 * self.h)
        D_mat_lowerDiag = np.diag([-diffusion + drift] * (self.dim - 1), -1)
        D_mat_upperDiag = np.diag([-diffusion - drift] * (self.dim - 1), 1)
        D_mat = D_mat_lowerDiag + D_mat_upperDiag + np.diag([self.theta2 * self.S0 ** 2 / self.h ** 2] * self.dim)
        return D_mat

    def gen_dD_dtheta1(self, t: float) -> np.array:
        lowerDiag = self.nu_t(t) / (2 * self.h)
        upperDiag = -lowerDiag
        dD_dtheta1_mat_lowerDiag = np.diag([lowerDiag] * (self.dim - 1), -1)
        dD_dtheta1_mat_upperDiag = np.diag([upperDiag] * (self.dim - 1), 1)
        dD_dtheta1_mat = dD_dtheta1_mat_lowerDiag + dD_dtheta1_mat_upperDiag
        return dD_dtheta1_mat

    def gen_dD_dtheta2(self) -> np.array:
        offDiag = -self.S0 ** 2 / (2 * self.h ** 2)
        diag = -2 * offDiag
        dD_dtheta2_mat_lowerDiag = np.diag([offDiag] * (self.dim - 1), -1)
        dD_dtheta2_mat_upperDiag = np.diag([offDiag] * (self.dim - 1), 1)
        dD_dtheta2_mat = dD_dtheta2_mat_lowerDiag + np.diag([diag] * self.dim) + dD_dtheta2_mat_upperDiag
        return dD_dtheta2_mat

    def gen_dD_dtheta3(self) -> np.array:
        return np.zeros((self.dim, self.dim))

    def gen_dD_dphi1(self, t: float) -> np.array:
        dnu_dphi1 = self.dnut_dphi1(t)
        lowerDiag = self.theta1 * dnu_dphi1 / (2 * self.h)
        upperDiag = -lowerDiag
        dD_dphi1_mat_lowerDiag = np.diag([lowerDiag] * (self.dim - 1), -1)
        dD_dphi1_mat_upperDiag = np.diag([upperDiag] * (self.dim - 1), 1)
        dD_dphi1_mat = dD_dphi1_mat_lowerDiag + dD_dphi1_mat_upperDiag
        return dD_dphi1_mat

    def gen_dD_dphi2(self) -> np.array:
        return np.zeros((self.dim, self.dim))

    '''Calculate operator matrices of FD'''
    def gen_A_B_dtheta(
            self, D_theta: np.array, dD_dtheta1: np.array, dD_dtheta2: np.array, dD_dtheta3: np.array
    ) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array]:
        if self.scheme == 'Implicit':
            A_theta = sparse.csc_matrix(self.identityMat + self.delta * D_theta)  # csc format
            A_theta_LU = splu(A_theta)

            dA_dtheta1 = self.delta * dD_dtheta1
            dA_dtheta2 = self.delta * dD_dtheta2
            dA_dtheta3 = self.delta * dD_dtheta3

            B_theta = self.identityMat
            dB_dtheta1, dB_dtheta2, dB_dtheta3 = \
                np.zeros(A_theta.shape), np.zeros(A_theta.shape), np.zeros(A_theta.shape)

        elif self.scheme == 'CN':
            A_theta = sparse.csc_matrix(self.identityMat + self.delta / 2 * D_theta)  # csc format
            A_theta_LU = splu(A_theta)

            dA_dtheta1 = self.delta / 2 * dD_dtheta1
            dA_dtheta2 = self.delta / 2 * dD_dtheta2
            dA_dtheta3 = self.delta / 2 * dD_dtheta3

            B_theta = self.identityMat - self.delta / 2 * D_theta  # no need to convert to sparse matrix
            dB_dtheta1 = -dA_dtheta1
            dB_dtheta2 = -dA_dtheta2
            dB_dtheta3 = -dA_dtheta3
        else:
            print(f"{self.scheme} scheme not implemented!")
            raise NotImplementedError

        return A_theta_LU, dA_dtheta1, dA_dtheta2, dA_dtheta3, B_theta, dB_dtheta1, dB_dtheta2, dB_dtheta3

    '''Calculate value vector, dV_dtheta, and dV_dphi'''
    def cal_value_dVdtheta_vec(self):
        dD_dtheta2 = self.gen_dD_dtheta2()  # same for all t
        dD_dtheta3 = self.gen_dD_dtheta3()  # same for all t
        for i in range(self.N - 1, -1, -1):  # time
            t = i * self.delta
            r_theta = self.gen_r_theta(t=t)
            dr_dtheta1, dr_dtheta2, dr_dtheta3 = self.gen_dr_dtheta(t=t)

            D_theta = self.gen_D_theta(t=t)
            dD_dtheta1 = self.gen_dD_dtheta1(t=t)


            A_theta_LU, dA_dtheta1, dA_dtheta2, dA_dtheta3, B_theta, dB_dtheta1, dB_dtheta2, dB_dtheta3 = \
                self.gen_A_B_dtheta(
                    D_theta=D_theta,
                    dD_dtheta1=dD_dtheta1,
                    dD_dtheta2=dD_dtheta2,
                    dD_dtheta3=dD_dtheta3
                )

            self.vVec_list[i] = A_theta_LU.solve(
                r_theta * self.delta
                + B_theta @ self.vVec_list[i + 1]
            )

            self.grad_theta1Vec_list[i] = A_theta_LU.solve(
                -dA_dtheta1 @ self.vVec_list[i]
                + dr_dtheta1 * self.delta
                + dB_dtheta1 @ self.vVec_list[i + 1]
                + B_theta @ self.grad_theta1Vec_list[i + 1]
                )

            self.grad_theta2Vec_list[i] = A_theta_LU.solve(
                -dA_dtheta2 @ self.vVec_list[i]
                + dr_dtheta2 * self.delta
                + dB_dtheta2 @ self.vVec_list[i + 1]
                + B_theta @ self.grad_theta2Vec_list[i + 1]
            )

            self.grad_theta3Vec_list[i] = A_theta_LU.solve(
                -dA_dtheta3 @ self.vVec_list[i]
                + dr_dtheta3 * self.delta
                + dB_dtheta3 @ self.vVec_list[i + 1]
                + B_theta @ self.grad_theta3Vec_list[i + 1]
            )

    def cal_dGdphi_vec(self):
        dD_dphi2 = self.gen_dD_dphi2()  # same for all t
        for i in range(self.N - 1, -1, -1):  # time
            t = i * self.delta
            dr_dphi1, dr_dphi2 = self.gen_dr_dphi(t=t)
            dD_dphi1 = self.gen_dD_dphi1(t=t)

            dA_dphi1 = self.delta / 2 * dD_dphi1
            dA_dphi2 = self.delta / 2 * dD_dphi2
            dB_dphi1 = -dA_dphi1
            dB_dphi2 = -dA_dphi2

            '''Now theta should have been updated'''
            D_theta = self.gen_D_theta(t=t)
            dD_dtheta1 = self.gen_dD_dtheta1(t=t)
            dD_dtheta2 = self.gen_dD_dtheta2()
            dD_dtheta3 = self.gen_dD_dtheta3()

            A_theta_LU, dA_dtheta1, dA_dtheta2, dA_dtheta3, B_theta, dB_dtheta1, dB_dtheta2, dB_dtheta3 = \
                self.gen_A_B_dtheta(
                    D_theta=D_theta,
                    dD_dtheta1=dD_dtheta1,
                    dD_dtheta2=dD_dtheta2,
                    dD_dtheta3=dD_dtheta3
                )
            ''''''

            self.grad_phi1Vec_list[i] = A_theta_LU.solve(
                -dA_dphi1 @ self.vVec_list[i]
                + dr_dphi1 * self.delta
                + dB_dphi1 @ self.vVec_list[i + 1]
                + B_theta @ self.grad_phi1Vec_list[i + 1]
            )

            self.grad_phi2Vec_list[i] = A_theta_LU.solve(
                -dA_dphi2 @ self.vVec_list[i]
                + dr_dphi2 * self.delta
                + dB_dphi2 @ self.vVec_list[i + 1]
                + B_theta @ self.grad_phi2Vec_list[i + 1]
            )

    def find_Sidx(self, S: np.array) -> np.array:
        '''
        :param S: (PATH_NUM, N), continuous prices
        :return: (PATH_NUM, N), indexes of the continuous prices in the price vectors
        '''
        S = np.clip(S, a_min=self.Smin, a_max=self.Smax)
        Delta_S = S - self.Smin
        S_idx = Delta_S // self.h
        S_idx += (Delta_S % self.h >= self.h / 2) * 1
        S_idx = S_idx.astype(int)  # (PATH_NUM, N)
        return S_idx

    def cal_dML_dtheta(self, S: np.array, running_r_mat: np.array) -> Tuple[float, float, float, float, np.array]:
        '''
        :param S: (PATH_NUM, N)
        :param running_r_mat: (PATH_NUM, N)
        :return:
        '''
        S_idx = self.find_Sidx(S=S)

        V_theta_grid, dV_dtheta1_grid, dV_dtheta2_grid, dV_dtheta3_grid = \
            np.zeros(S.shape), np.zeros(S.shape), np.zeros(S.shape), np.zeros(S.shape)

        '''precompute and updated Vvec and dVdtheta using current theta'''
        self.cal_value_dVdtheta_vec()
        ''''''

        for i in range(self.N):
            S_idx_i = S_idx[:, i]
            V_theta_grid[:, i] = self.vVec_list[i][S_idx_i]
            dV_dtheta1_grid[:, i] = self.grad_theta1Vec_list[i][S_idx_i]
            dV_dtheta2_grid[:, i] = self.grad_theta2Vec_list[i][S_idx_i]
            dV_dtheta3_grid[:, i] = self.grad_theta3Vec_list[i][S_idx_i]

        reward_to_go = running_r_mat.sum(axis=1).reshape(-1, 1) - running_r_mat.cumsum(axis=1) + running_r_mat
        error = reward_to_go - V_theta_grid
        MLoss = np.mean(np.sum(error ** 2, axis=1)) * self.delta

        dML_dtheta1 = np.mean(np.sum(-dV_dtheta1_grid * error, axis=1)) * self.delta
        dML_dtheta2 = np.mean(np.sum(-dV_dtheta2_grid * error, axis=1)) * self.delta
        dML_dtheta3 = np.mean(np.sum(-dV_dtheta3_grid * error, axis=1)) * self.delta

        return dML_dtheta1, dML_dtheta2, dML_dtheta3, MLoss, V_theta_grid

    def PE_1step(self, S: np.array, running_r_mat: np.array) -> Tuple[float, float, float, float, np.array]:
        '''
        :param S: (PATH_NUM, N)
        :param running_r_mat: (PATH_NUM, N)
        :return:
        '''
        derror_dtheta1, derror_dtheta2, derror_dtheta3, MLoss, V_theta_grid = self.cal_dML_dtheta(S, running_r_mat)

        '''gradient descent to update env param'''
        self.theta1 -= self.alpha_theta1 * derror_dtheta1
        self.theta2 -= self.alpha_theta2 * derror_dtheta2
        self.theta3 -= self.alpha_theta3 * derror_dtheta3
        return derror_dtheta1, derror_dtheta2, derror_dtheta3, MLoss, V_theta_grid

    def PG_1step(self) -> Tuple[float, float]:
        '''
        :return:
        '''
        '''compute and updated dGdphi1 using current theta'''
        self.cal_dGdphi_vec()
        ''''''
        S0_idx = self.find_Sidx(S=self.S0)

        '''gradient ascent to update policy param'''
        self.phi1 += self.alpha_phi1 * self.grad_phi1Vec_list[0][S0_idx]
        self.phi2 += self.alpha_phi2 * self.grad_phi2Vec_list[0][S0_idx]
        return self.grad_phi1Vec_list[0][S0_idx], self.grad_phi2Vec_list[0][S0_idx]


class Training:
    def __init__(self):
        '''Setting up parameters for training'''

        '''Environment param'''
        self.T = 1 / 250  # year
        self.N = 5000
        self.Delta_t = self.T / self.N

        self.S0 = 100.0
        self.q0 = 1
        self.Q0 = 5e5

        self.kappa_true = 2.5e-7 * self.Q0  # equivalent permanent impact in pct of inventory
        self.eta_true = 2.5e-6 * self.T * self.Q0  # equivalent (annualized) temporary impact in pct of inventory
        self.sigma_true = 0.3  # annualized return vol
        ''''''

        '''Execution agent param'''
        self.Lambda_x = 3
        self.Lambda = round(10 ** self.Lambda_x * self.eta_true, 8)
        self.zeta = 5

        self.phi1_opt = np.sqrt(self.Lambda * self.sigma_true ** 2 * self.S0 ** 2 / self.eta_true)
        self.phi2_opt = 1 / (2 * self.eta_true)


        behavior_coef = 2
        self.phi1_list = [behavior_coef * self.phi1_opt]
        self.phi2_list = [behavior_coef * self.phi2_opt]
        self.dG_dphi1_list, self.dG_dphi2_list = [], []

        theta_init = 2
        self.theta1_list = [theta_init * self.kappa_true]
        self.theta2_list = [theta_init * self.sigma_true ** 2]
        self.theta3_list = [theta_init * self.eta_true]
        self.dML_dtheta1_list, self.dML_dtheta2_list, self.dML_dtheta3_list = [], [], []

        self.cum_r = []

        '''Set learning rates'''

        alpha_phi1 = 4e4
        alpha_phi2 = 1e5

        alpha_theta1 = 3.3e2
        alpha_theta2 = 8
        alpha_theta3 = 7e-3
        ''''''


        # self.PATH_NUM = 1_000
        self.PATH_NUM = 1
        # self.PATH_NUM = 64
        # self.PATH_NUM = 128
        scheme = 'CN'

        Srange_pct = 0.2
        Smax, Smin = self.S0 * (1 + Srange_pct), self.S0 * (1 - Srange_pct)
        dim = 50
        h = (Smax - Smin) / dim
        Svec = np.linspace(start=Smin, stop=Smax, num=dim, endpoint=False)  # including Smin

        self.singleExecFD_agent = singleExecFD(
            q0=self.q0,  # normalized
            S0=self.S0,
            T=self.T,
            zeta=self.zeta,
            Lambda=self.Lambda,
            phi1=self.phi1_list[-1],
            phi2=self.phi2_list[-1],
            theta1=self.theta1_list[-1],
            theta2=self.theta2_list[-1],
            theta3=self.theta3_list[-1],
            alpha_phi1=alpha_phi1,
            alpha_phi2=alpha_phi2,
            alpha_theta1=alpha_theta1,
            alpha_theta2=alpha_theta2,
            alpha_theta3=alpha_theta3,
            N=self.N,
            delta=self.Delta_t,
            dim=dim,
            h=h,
            Svec=Svec,
            Smin=Smin,
            Smax=Smax,
            scheme=scheme
        )


        # self.training_iter = 200
        self.training_iter = 150
        # self.training_iter = 20  # for quick test
        # self.training_iter = 3  # for quick test

        self.param_dict = {
            'Lambda_x': self.Lambda_x,
            'zeta': self.zeta,
            'N': self.N,
            'dim': dim,
            'Srange_pct': Srange_pct,
            'PATH_NUM': self.PATH_NUM,
            'training_iter': self.training_iter,
            'alpha_phi1': alpha_phi1,
            'alpha_phi2': alpha_phi2,
            'alpha_theta1': alpha_theta1,
            'alpha_theta2': alpha_theta2,
            'alpha_theta3': alpha_theta3
        }  # to save later
        print(f"Training param: {self.param_dict}")

    def gen_q_r(self, S: np.array, nu_normal: np.array) -> Tuple[np.array, np.array, np.array]:
        '''
        :param S: (PATH_NUM, N+1)
        :param nu_normal: (PATH_NUM, N-1)
        :return: q: (PATH_NUM, N+1), running_r: (PATH_NUM, N), S: (PATH_NUM, N+1)
        '''
        pi_var = self.zeta * self.singleExecFD_agent.phi2
        pi_std = np.sqrt(pi_var)
        q = np.zeros((self.PATH_NUM, self.N + 1))  # sample inventory trajectories
        q[:, 0] = self.q0

        nu = np.zeros((self.PATH_NUM, self.N))
        # nu_normal = np.random.normal(loc=0.0, scale=1.0, size=(self.PATH_NUM, self.N - 1))
        for i in range(1, self.N):
            mu = -q[:, i - 1] * self.singleExecFD_agent.phi1 / np.tanh(self.singleExecFD_agent.phi1 * (self.N - i) * self.Delta_t)
            nu[:, i - 1] = mu + pi_std * nu_normal[:, i - 1]
            q[:, i] = q[:, i - 1] + nu[:, i - 1] * self.Delta_t
        nu[:, -1] = (0 - q[:, -2]) / self.Delta_t  # liquidate all remaining shares at the last time step

        S += self.kappa_true * (q - self.q0)  # price after permanent impact

        Delta_xt = -nu * (S[:, :-1] + self.eta_true * nu) * self.Delta_t
        St_Delta_qt = S[:, :-1] * nu * self.Delta_t
        qt_Delta_St = q[:, :-1] * (S[:, 1:] - S[:, :-1])
        QV_penalty = -(Delta_xt + St_Delta_qt + qt_Delta_St) ** 2

        entropy = self.zeta * np.log(2 * np.pi * np.e * pi_var) / 2 * self.Delta_t if self.zeta != 0 else 0

        running_r = Delta_xt + self.Lambda * QV_penalty + entropy

        return q, running_r, S

    @staticmethod
    def cal_value_func(
        theta1_test: float, theta2_test: float, theta3_test: float, phi1: float, phi2: float,
        q, S, tau, zeta: float, Lambda: float, S0: float
    ):
        '''
        :param theta1_test:
        :param theta2_test:
        :param theta3_test:
        :param phi1:
        :param phi2:
        :param q: could be a float or np.array
        :param S: could be a float or np.array
        :param tau: could be a float or np.array
        :param zeta:
        :param Lambda:
        :param S0:
        :return: could be a float or np.array
        '''
        theta3_phi1 = theta3_test * phi1
        lambda_theta2_S02_phi1 = Lambda * theta2_test * S0 ** 2 / phi1
        V_theta = q * S - q ** 2 / 2 * (
                theta1_test +
                (theta3_phi1 + lambda_theta2_S02_phi1) / np.tanh(phi1 * tau) +
                (theta3_phi1 - lambda_theta2_S02_phi1) * phi1 * tau / np.sinh(phi1 * tau) ** 2
        )
        entropy = zeta * tau * (np.log(2 * np.pi * np.e * zeta * phi2) / 2 - theta3_test * phi2) if zeta > 0 else 0
        return V_theta + entropy

    def start_actor_critic_offline(self, seed: int = 1234):
        '''
        :param seed: random seed
        :return:
        '''
        print(f"theta1={self.singleExecFD_agent.theta1}, theta1*={self.kappa_true}, \n"
              f"theta2={self.singleExecFD_agent.theta2}, theta2*={self.sigma_true ** 2}, \n"
              f"theta3={self.singleExecFD_agent.theta3}, theta3*={self.eta_true}, \n"
              f"phi1={self.singleExecFD_agent.phi1}, phi1*={self.phi1_opt}, \n"
              f"phi2={self.singleExecFD_agent.phi2}, phi2*={self.phi2_opt} \n")
        startTime = time.perf_counter()
        np.random.seed(seed)
        for i in range(self.training_iter):
            print(f"====================================\n"
                  f"Train: {i+1}/{self.training_iter}, seed={seed}, path={self.PATH_NUM}")

            '''Generate episodes'''
            dWt = self.sigma_true * self.S0 * \
                  np.random.normal(
                      loc=0.0,
                      scale=np.sqrt(self.Delta_t),
                      size=(self.PATH_NUM, self.N)
                  )
            S0_array = self.S0 * np.ones(self.PATH_NUM).reshape(-1, 1)
            S = np.cumsum(np.concatenate([S0_array, dWt], axis=1), axis=1)
            nu_normal = np.random.normal(loc=0.0, scale=1.0, size=(self.PATH_NUM, self.N - 1))
            q, running_r, S = self.gen_q_r(S=S, nu_normal=nu_normal)

            self.cum_r.append(np.mean(running_r.sum(axis=1)))
            ''''''

            '''PE: after PE, theta has been updated'''
            derror_dtheta1, derror_dtheta2, derror_dtheta3, _, _ = \
                self.singleExecFD_agent.PE_1step(S=S[:, :-1], running_r_mat=running_r)

            self.theta1_list.append(self.singleExecFD_agent.theta1)
            self.theta2_list.append(self.singleExecFD_agent.theta2)
            self.theta3_list.append(self.singleExecFD_agent.theta3)

            self.dML_dtheta1_list.append(derror_dtheta1)
            self.dML_dtheta2_list.append(derror_dtheta2)
            self.dML_dtheta3_list.append(derror_dtheta3)
            ''''''

            '''PG: use updated theta; after PG, phi has been updated'''
            dGdphi1, dGdphi2 = self.singleExecFD_agent.PG_1step()

            self.phi1_list.append(self.singleExecFD_agent.phi1)
            self.phi2_list.append(self.singleExecFD_agent.phi2)

            self.dG_dphi1_list.append(dGdphi1)
            self.dG_dphi2_list.append(dGdphi2)
            ''''''

            print(f"theta1={self.singleExecFD_agent.theta1}, theta1*={self.kappa_true}, grad={derror_dtheta1} \n"
                  f"theta2={self.singleExecFD_agent.theta2}, theta2*={self.sigma_true**2}, grad={derror_dtheta2} \n"
                  f"theta3={self.singleExecFD_agent.theta3}, theta3*={self.eta_true}, grad={derror_dtheta3} \n"
                  f"phi1={self.singleExecFD_agent.phi1}, phi1*={self.phi1_opt}, grad={dGdphi1}\n"
                  f"phi2={self.singleExecFD_agent.phi2}, phi2*={self.phi2_opt}, grad={dGdphi2} \n"
                  f"====================================")
        print(f"Runtime={time.perf_counter() - startTime}s")

        return self.phi1_list, [0] + self.dG_dphi1_list, self.phi2_list, [0] + self.dG_dphi2_list, \
               self.theta1_list, [0] + self.dML_dtheta1_list, self.theta2_list, [0] + self.dML_dtheta2_list, \
               self.theta3_list, [0] + self.dML_dtheta3_list, [0] + self.cum_r, self.param_dict

    def train_plot(self):

        avg_window = int(self.training_iter * 0.1)
        phi1_learned = round(np.mean(self.phi1_list[-avg_window:]), 2)
        train_res = pd.DataFrame()
        train_res['phi1'] = self.phi1_list
        train_res['dG_dphi1'] = [0] + self.dG_dphi1_list
        train_res['phi2'] = self.phi2_list
        train_res['dG_dphi2'] = [0] + self.dG_dphi2_list
        train_res['theta1'] = self.theta1_list
        train_res['dML_dtheta1'] = [0] + self.dML_dtheta1_list
        train_res['theta2'] = self.theta2_list
        train_res['dML_dtheta2'] = [0] + self.dML_dtheta2_list
        train_res['theta3'] = self.theta3_list
        train_res['dML_dtheta3'] = [0] + self.dML_dtheta3_list
        train_res['realized_V'] = [0] + self.cum_r
        save_name_prefix = f'phi1_{phi1_learned}_ActorCritic_Offline'
        train_res.to_csv(f'{save_name_prefix}_learnPath.csv')

        param_df = pd.DataFrame.from_dict(self.param_dict, orient='index')
        param_df.to_csv(f'{save_name_prefix}_param.csv')

        ## Create figure
        fig = plt.figure(figsize=(12, 9), dpi=300)

        ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=1)
        ax1.plot(self.phi1_list, label=f"emp={phi1_learned}")
        ax1.hlines(y=self.phi1_opt, xmin=0, xmax=self.training_iter, linestyles='--', colors='r',
                   label=f"opt={round(self.phi1_opt, 8)}")
        ax1.title.set_text(f'phi1, ln={self.singleExecFD_agent.alpha_phi1}')
        plt.legend()


        ax2 = plt.subplot2grid((3, 2), (0, 1), colspan=1)
        ax2.plot(self.phi2_list, label=f"emp={round(np.mean(self.phi2_list[-avg_window:]), 3)}")
        ax2.hlines(y=self.phi2_opt, xmin=0, xmax=self.training_iter, linestyles='--', colors='r')
        ax2.title.set_text(f'phi2, ln={self.singleExecFD_agent.alpha_phi2}')
        plt.legend()

        ax3 = plt.subplot2grid((3, 2), (1, 0), colspan=1)
        ax3.plot(self.theta1_list, label=f"emp={round(np.mean(self.theta1_list[-avg_window:]), 8)}")
        ax3.hlines(y=self.kappa_true, xmin=0, xmax=self.training_iter, linestyles='--', colors='r',
                   label=f"opt={self.kappa_true}")
        ax3.title.set_text(f'theta1, ln={self.singleExecFD_agent.alpha_theta1}')
        plt.legend()

        ax4 = plt.subplot2grid((3, 2), (1, 1), colspan=1)
        ax4.plot(self.theta2_list, label=f"emp={round(np.mean(self.theta2_list[-avg_window:]), 8)}")
        ax4.hlines(y=self.sigma_true**2, xmin=0, xmax=self.training_iter, linestyles='--', colors='r')
        ax4.title.set_text(f'theta2, ln={self.singleExecFD_agent.alpha_theta2}')
        plt.legend()

        ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=1)
        ax5.plot(self.theta3_list, label=f"emp={round(np.mean(self.theta3_list[-avg_window:]), 8)}")
        ax5.hlines(y=self.eta_true, xmin=0, xmax=self.training_iter, linestyles='--', colors='r')
        ax5.title.set_text(f'theta3, ln={self.singleExecFD_agent.alpha_theta3}')
        plt.legend()

        ax6 = plt.subplot2grid((3, 2), (2, 1), colspan=1)
        ax6.plot(self.cum_r, label=f"emp={round(np.mean(self.cum_r[-avg_window:]), 8)}")
        J_opt = self.cal_value_func(
            theta1_test=self.kappa_true,
            theta2_test=self.sigma_true ** 2,
            theta3_test=self.eta_true,
            phi1=self.phi1_opt,
            phi2=self.phi2_opt,
            q=self.q0,
            S=self.S0,
            tau=self.T,
            zeta=self.zeta,
            Lambda=self.Lambda,
            S0=self.S0
        )
        ax6.hlines(y=J_opt, xmin=0, xmax=self.training_iter, linestyles='--', colors='r')
        ax6.title.set_text(f'V')
        plt.legend()

        fig.suptitle(f"Lam{self.Lambda_x}, N={self.N}, dim={self.singleExecFD_agent.dim}")
        plt.savefig(f'{save_name_prefix}.pdf')


    def run(self, seed=1234, plot=True):
        training_res = self.start_actor_critic_offline(seed=seed)
        if plot:
            self.train_plot()
        return training_res


def Training_once(seed=1234, plot=False):
    actorCriticTraining = Training()
    return actorCriticTraining.run(seed, plot)


def Training_multiProcess(num_repeat):
    seed_list = [1234 + i for i in range(num_repeat)]

    RunningStartTime = time.perf_counter()
    # with ProcessPoolExecutor(max_workers=10) as pool:
    with ProcessPoolExecutor() as pool:
        training_res = pool.map(Training_once, seed_list)

    res = []
    for r in training_res:
        res.append(r)

    phi1, dG_dphi1, phi2, dG_dphi2, \
    theta1, dML_dtheta1, \
    theta2, dML_dtheta2, \
    theta3, dML_dtheta3, \
    cum_r, param_dict = zip(*res)

    print(f"runtime={round(time.perf_counter() - RunningStartTime, 3)}s, saving results......")

    phi1_learned = np.mean([phi1_path[-1] for phi1_path in phi1])
    def save_res(listOfList, res_name):
        print(f"saving {res_name}")
        if res_name != 'param_dict':
            res_df = pd.DataFrame(listOfList)  # row: training path in one repeat, # col=num_repeats
            res_df.to_csv(f'phi1_{phi1_learned}_ActorCritic_Offline_learnPath_{res_name}.csv')
        else:
            param_df = pd.DataFrame.from_dict(listOfList[0], orient='index')
            param_df.to_csv(f'phi1_{phi1_learned}_ActorCritic_Offline_param.csv')

    save_res(phi1, 'phi1')
    save_res(dG_dphi1, 'dG_dphi1')
    save_res(phi2, 'phi2')
    save_res(dG_dphi2, 'dG_dphi2')
    save_res(theta1, 'theta1')
    save_res(dML_dtheta1, 'dML_dtheta1')
    save_res(theta2, 'theta2')
    save_res(dML_dtheta2, 'dML_dtheta2')
    save_res(theta3, 'theta3')
    save_res(dML_dtheta3, 'dML_dtheta3')
    save_res(cum_r, 'cum_r')
    save_res(param_dict, 'param_dict')



if __name__ == '__main__':
    '''Train once'''
    # Training_once()

    '''Repeat training for multiple times'''
    Training_multiProcess(num_repeat=25)





