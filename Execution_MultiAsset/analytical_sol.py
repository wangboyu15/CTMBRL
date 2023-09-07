import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'text.usetex': True,
    'font.family': 'Times New Roman'
})

from scipy.integrate import odeint


class multiExec:
    def __init__(
            self,
            T: float,
            N: int,
            d_asset: int,
            S0: np.array,
            Q0: np.array,
            kappa_true_mat: np.array,
            alpha_mat: np.array,
            eta_true_mat: np.array,
            eta_true_inv: np.array,
            Sigma_true_mat: np.array,
            Lambda: float,
            zeta: float,
            b: float = 1.0
    ):
        '''
        :param T:
        :param N:
        :param d_asset:
        :param S0: (d_asset, )
        :param Q0: (d_asset, )
        :param kappa_true_mat: (d_asset, d_asset)
        :param alpha_mat: (d_asset, d_asset)
        :param eta_true_mat: (d_asset, d_asset)
        :param eta_true_inv: (d_asset, d_asset)
        :param Sigma_true_mat: (d_asset, d_asset)
        :param Lambda:
        :param zeta:
        :param b:
        '''
        self.T = T
        self.N = N
        self.Delta_t = self.T / self.N
        self.d_asset = d_asset
        self.S0 = S0.reshape(-1, 1)
        self.Q0 = Q0
        self.kappa_true_mat = kappa_true_mat
        self.alpha_mat = alpha_mat
        self.eta_true_mat = eta_true_mat
        self.eta_true_inv = eta_true_inv
        self.Sigma_true_mat = Sigma_true_mat
        self.Lambda = Lambda
        self.zeta = zeta
        self.entropy = self.zeta * self.T * np.log(
            (np.pi * self.zeta) ** self.d_asset / np.linalg.det(eta_true_mat)
        ) / 2 if self.zeta > 0 else 0

        self.Ct = self.cal_Ct()

        self.b = b
        twoeta_inv = eta_true_inv / 2
        self.Bt_list = [self.b * twoeta_inv @ c for c in self.Ct]


    def cal_Cdot(self, C: np.array, t: float) -> np.array:
        '''
        :param C: np.array (d_asset * d_asset, )
        :param t:
        :return: np.array (d_asset * d_asset, )
        '''
        # print(f"cal C, t={t}, idx={int(round(t / self.Delta_t))}")
        C = C.reshape(self.d_asset, self.d_asset)
        Cdot = -C.dot(self.eta_true_inv).dot(C) / 2 + 2 * self.Lambda * self.Sigma_true_mat
        return Cdot.flatten()


    def cal_Ct(self, t0: float = 0.0) -> np.array:
        y0 = 2 * self.alpha_mat - self.kappa_true_mat
        y0 = y0.flatten()  # (d_asset * d_asset, )

        # Time vector
        t = np.linspace(t0, self.T, num=self.N + 1)

        # Solve the ODE using odeint
        sol = odeint(self.cal_Cdot, y0, t)
        sol = sol.reshape((self.N + 1, self.d_asset, self.d_asset))
        sol = sol[::-1]     # (num_time_interval + 1, d_asset, d_asset)
        return sol

    def cal_Ddot(self, D: np.array, t: float) -> np.array:
        '''
        :param D: np.array (d_asset * d_asset, )
        :param t:
        :return: Ddot: np.array (d_asset * d_asset, )
        '''

        Ct = self.cal_Ct(t0=t)[0]
        if t == self.T:
            print("T:", "true=", 2 * self.alpha_mat - self.kappa_true_mat, 'Ct=', Ct, "Dt=", D)
        twoeta_inv = self.eta_true_inv / 2
        Bt_mat = self.b * twoeta_inv @ Ct

        # B_eta_B = Bt_mat.transpose() @ self.eta_true_mat @ Bt_mat
        # C_etainv_C = Ct @ self.eta_true_inv @ Ct / 4
        # print(f"B_eta_B={B_eta_B}")
        # print(f"C_etainv_C={C_etainv_C}")
        # print(f"error=", np.sum((B_eta_B - C_etainv_C)**2))


        # Bt_mat = (self.Bt_list[min(int(round(t / self.Delta_t)), self.N)])
        Dt_transpose = D.reshape(self.d_asset, self.d_asset).transpose()
        # print(f"D-C error={np.sum((D.reshape(self.d_asset, self.d_asset) - Ct)**2)}")
        # print(f"DT-C error={np.sum((D.reshape(self.d_asset, self.d_asset) - Ct) ** 2)}")
        #
        # print(f"2DTB = {2 * Dt_transpose @ Bt_mat}")
        # print(f"C_etainv_C = {Ct @ self.eta_true_inv @ Ct}")
        # print(f"error={np.mean(( 2 * Dt_transpose @ Bt_mat - Ct @ self.eta_true_inv @ Ct )**2)}")

        Ddot = 2 * (
                Dt_transpose @ Bt_mat
                - Bt_mat.transpose() @ self.eta_true_mat @ Bt_mat
                - self.Lambda * self.Sigma_true_mat
        )


        # print(f"term1={2* self.b * Dt_transpose @ twoeta_inv @ Ct}")
        # print(f"term2={2*Bt_mat.transpose() @ self.eta_true_mat @ Bt_mat}")
        # print(f"compare={self.b ** 2 * Ct @ self.eta_true_inv @ Ct / 2}")
        # print(f"error={np.sum((self.b ** 2 * Ct @ self.eta_true_inv @ Ct / 2 - 2*Bt_mat.transpose() @ self.eta_true_mat @ Bt_mat)**2)}")
        # print(f"term3={Ct.dot(self.eta_true_inv).dot(Ct)}")
        # print(f"b={self.b}")
        Ddot0 = self.b * Dt_transpose @ self.eta_true_inv @ Ct \
                - (self.b ** 2 / 2) * Ct @ self.eta_true_inv @ Ct \
                - 2 * self.Lambda * self.Sigma_true_mat

        t1 = 2 * Dt_transpose @ Bt_mat
        t2 = self.b * Dt_transpose @ self.eta_true_inv @ Ct
        error1 = np.sum((t1 - t2) ** 2)

        print(f"t={t}, b={self.b}, Ddot0-Ddot error={np.sum((Ddot0 - Ddot)**2)}")
        print(f"error0={np.sum((2 * Bt_mat - self.b * self.eta_true_inv @ Ct) ** 2)}")
        print(f"error1={np.sum((2 * Dt_transpose @ Bt_mat - self.b * Dt_transpose @ self.eta_true_inv @ Ct) ** 2)}")
        print(f"error2={np.sum((2 * Bt_mat.transpose() @ self.eta_true_mat @ Bt_mat - (self.b ** 2 / 2) * Ct @ self.eta_true_inv @ Ct) ** 2)}")

        # Cdot = Ct.dot(self.eta_true_inv).dot(Ct) / 2 - 2 * self.Lambda * self.Sigma_true_mat
        # print(f"t={t}, error=", np.mean((Ddot - Cdot) ** 2))
        return Ddot.flatten()

    def cal_Dt(self) -> np.array:
        t0 = 0.0
        y0 = 2 * self.alpha_mat - self.kappa_true_mat
        y0 = y0.flatten()  # (d_asset * d_asset, )

        # Time vector
        t = np.linspace(self.T, t0, num=self.N + 1)
        # print(f"len(t)={len(t)}")

        # Solve the ODE using odeint
        sol = odeint(self.cal_Ddot, y0, t)
        sol = sol.reshape((self.N + 1, self.d_asset, self.d_asset))
        sol = sol[::-1]     # (num_time_interval + 1, d_asset, d_asset)
        print(f"terminal={2 * self.alpha_mat - self.kappa_true_mat}")
        print(f"D0={sol[0]}")
        print(f"D-1={sol[-1]}")
        print(f"C0={self.Ct[0]}")
        return sol

    def cal_qt(self, plot: bool = False, save_plot: bool = False) -> Tuple[np.array, np.array]:
        '''
        :param gen_opt: whether to generate optimal inventory trajectory
        :param plot:
        :param save_plot:
        :return:
        '''

        qt = np.ones((self.d_asset, self.N + 1))
        mut = np.zeros((self.d_asset, self.N + 1))
        for i in range(self.N):
            mut[:, i] = -self.Bt_list[i] @ qt[:, i]
            qt[:, i + 1] = qt[:, i] + mut[:, i] * self.Delta_t

        if plot:
            plot_x = list(range(self.N + 1))
            plt.figure(dpi=300)
            for i in range(self.d_asset):
                plt.plot(plot_x, qt[i, :].tolist(), label=f"asset {i + 1}", linewidth=0.8)

            plt.hlines(y=0.0, xmin=plot_x[0], xmax=plot_x[-1], ls='--', colors='r', linewidth=0.5)
            plt.legend()
            plt.ylabel(r'$q_t/q_0$')
            plt.xlabel(r'$N$')
            if save_plot:
                plt.savefig(f"Inventory_{self.d_asset}d_Lam{self.Lambda}.pdf")
            plt.show()
            plt.close()

        return qt, mut

    def cal_val(self) -> float:
        raise NotImplementedError



class multiExec_analytical(multiExec):

    def __init__(
            self,
            T: float,
            N: int,
            d_asset: int,
            S0: np.array,
            Q0: np.array,
            kappa_true_mat: np.array,
            alpha_mat: np.array,
            eta_true_mat: np.array,
            eta_true_inv: np.array,
            Sigma_true_mat: np.array,
            Lambda: float,
            zeta: float,
            b: float = 1.0
    ):
        super().__init__(
            T=T,
            N=N,
            d_asset=d_asset,
            S0=S0,
            Q0=Q0,
            kappa_true_mat=kappa_true_mat,
            alpha_mat=alpha_mat,
            eta_true_mat=eta_true_mat,
            eta_true_inv=eta_true_inv,
            Sigma_true_mat=Sigma_true_mat,
            Lambda=Lambda,
            zeta=zeta,
            b=b
        )

    # def cal_val(self) -> float:
    #     # q0_vec = np.ones((self.d_asset, 1))
    #     q0_vec = self.Q0.reshape(-1, 1)
    #     J0 = q0_vec.T @ self.S0 - 1 / 2 * q0_vec.T @ (self.kappa_true_mat + self.Ct[0]) @ q0_vec + self.entropy
    #     # J0 = (q0_vec.T @ self.S0 - 1 / 2 * q0_vec.T @ (self.kappa_true_mat + self.Ct[0]) @ q0_vec) / np.mean(self.Q0) \
    #     #      + self.entropy
    #     # J0.shape == (1, 1)
    #     return J0[0][0]

    def cal_val(self) -> float:
        q0_vec = self.Q0.reshape(-1, 1)
        Dt = self.cal_Dt()
        # for i in range(len(Dt)):
        #     print(f"Dt-Ct={Dt[i] - self.Ct[i]}")
        # print(f"Dt={Dt[0]}, Ct={self.Ct[0]}, "
        #       f"Bt={self.Bt_list[0]}, {self.eta_true_inv@self.Ct[0]/2}")
        J0 = (q0_vec.T @ self.S0 - 1 / 2 * q0_vec.T @ (self.kappa_true_mat + Dt[0]) @ q0_vec) / np.mean(self.Q0) \
             + self.entropy
        # J0.shape == (1, 1)
        return J0[0][0]


class multiExec_MonteCarlo(multiExec):

    def __init__(
            self,
            T: float,
            N: int,
            d_asset: int,
            S0: np.array,
            Q0: np.array,
            kappa_true_mat: np.array,
            alpha_mat: np.array,
            eta_true_mat: np.array,
            eta_true_inv: np.array,
            Sigma_true_mat: np.array,
            sigma_corr_Lower_tau: np.array,
            Lambda: float,
            zeta: float,
            b: float = 1.0
    ):
        super().__init__(
            T=T,
            N=N,
            d_asset=d_asset,
            S0=S0,
            Q0=Q0,
            kappa_true_mat=kappa_true_mat,
            alpha_mat=alpha_mat,
            eta_true_mat=eta_true_mat,
            eta_true_inv=eta_true_inv,
            Sigma_true_mat=Sigma_true_mat,
            Lambda=Lambda,
            zeta=zeta,
            b=b
        )

        self.sigma_corr_Lower_tau = sigma_corr_Lower_tau

    def cal_val(self, num_sample=10000):
        print(f"calculate qt, mut, permImpact")
        qt, mut = self.cal_qt(plot=False, save_plot=False)  # qt, mut: (d_asset, N+1)
        permImpact = np.expand_dims(self.kappa_true_mat @ (qt - 1), axis=0)  # q0=1 for all assets, (1, d_asset, N+1)




        # K_true = np.sqrt(self.Lambda * 0.3 ** 2 * 100 ** 2 / self.eta_true_mat[0,0])
        # t = self.Delta_t * np.arange(self.N+1)
        #
        # def q_t(T, t, phi1):
        #     return np.sinh(phi1 * (T - t)) / np.sinh(phi1 * T)
        #
        # qt_formula = q_t(T=self.T, t=t, phi1=K_true)
        # plt.figure()
        # plt.plot(qt[0, :].tolist(), label=f'ODE solver')
        # plt.plot(qt_formula.tolist(), label='formula')
        # plt.legend()
        # plt.show()




        print(f"calculate qtSigmaqt")
        qtSigmaqt = np.zeros(self.N + 1)  # (N+1, )
        for i in range(self.N + 1):
            qtSigmaqt[i] = qt[:, i] @ self.Sigma_true_mat @ qt[:, i]
            # print(f"qtSigmaqt[{i}]={qtSigmaqt[i]}, {np.sum(((self.sigma_corr_Lower_tau/np.sqrt(self.Delta_t)).T @ qt[:, i]) ** 2)}")

        print(f"simulate dw_train")
        np.random.seed(1234)
        dw_train = self.sigma_corr_Lower_tau @ np.random.normal(
            loc=0.0,
            scale=1.0,
            size=(num_sample, self.d_asset, self.N)
        )

        print(f"calculate S_train")
        S_train = np.expand_dims(self.S0, axis=0) + np.concatenate(
            (
                np.zeros((num_sample, self.d_asset, 1)),
                np.cumsum(dw_train, axis=2)
            ), axis=2
        ) + permImpact

        # plt.figure()
        # for i in range(50):
        #     plt.plot(S_train[i, 0, :])
        # plt.show()

        print(f"calculate cum_r")
        alpha_qT = np.expand_dims(self.alpha_mat @ qt[:, -1], axis=0)  # (1, d_asset)

        # cum_r = (S_train[:, :, -1] - alpha_qT) @ qt[:, -1]
        # for i in range(self.N - 1, -1, -1):
        #     muti = mut[:, i]  # (d_asset, )
        #     cum_r += (S_train[:, :, i] + np.expand_dims(self.eta_true_mat @ muti, axis=0)) @ (-muti) * self.Delta_t \
        #              - self.Lambda * qtSigmaqt[i] * self.Delta_t
        # cum_r = np.mean(cum_r)
        # return cum_r

        running_r = np.zeros((num_sample, self.N+1))
        running_r[:, -1] = (S_train[:, :, -1] - alpha_qT) @ qt[:, -1]
        for i in range(self.N - 1, -1, -1):
            muti = mut[:, i]  # (d_asset, )
            running_r[:, i] += (S_train[:, :, i] + np.expand_dims(self.eta_true_mat @ muti, axis=0)) @ (-muti) * self.Delta_t \
                     - self.Lambda * qtSigmaqt[i] * self.Delta_t

        cum_r = np.mean(running_r.sum(axis=1))
        return cum_r, running_r, qt, S_train


def test_3d():
    T = 1 / 250
    N = 2000
    tau = T / N

    dim = 3

    S0 = np.array([100] * dim)
    sigma = np.linspace(0.3, 0.5, dim) * S0

    Q0 = [5e5] * dim


    def gen_rho(d_asset: int, rho_low: float = 0.05, rho_high: float = 0.6):
        num_pair = int(round(d_asset * (d_asset - 1) / 2))
        for rho in np.linspace(rho_high, rho_low, num_pair):
            yield rho

    corr_mat = np.diag(np.ones(dim))
    '''self.corr_mat.shape = (d_asset, d_asset)'''
    rho_generator = gen_rho(dim)
    for i in range(dim):
        for j in range(i + 1, dim):
            rho = next(rho_generator)
            corr_mat[i, j] = rho
            corr_mat[j, i] = rho

    # print(f"corr_mat={corr_mat}")

    corr_Lower = np.linalg.cholesky(corr_mat)
    sigma_corr_Lower = corr_Lower * sigma.reshape(-1, 1)
    sigma_corr_Lower_tau = sigma_corr_Lower * np.sqrt(tau)

    kappa_true = np.linspace(1.5, 3.5, dim) * 1e-7 * np.array(Q0)
    kappa_true_mat = np.diag(kappa_true)  # (dim, dim)
    if dim > 1:
        kappa_true_mat[0, 1] = 0.8 * kappa_true[0]
        kappa_true_mat[1, 0] = 0.8 * kappa_true[0]

    print(f"perm_impact mat = {kappa_true_mat}")
    print(f"cost by perm_impact={-np.sum(kappa_true_mat)/2}")

    # alpha_mat = 10 * kappa_true_mat
    alpha_mat = 1000 * kappa_true_mat

    # eta_true = 1e-8 * 5e5
    eta_true = np.linspace(0.5, 1.5, dim) * 1e-8 * np.array(Q0)
    eta_true_mat = np.diag(eta_true)
    eta_true_inv = np.linalg.inv(eta_true_mat)

    Lambda_x = 3
    Lambda = 10 ** Lambda_x * 1e-8 * 5e5
    # Lambda = 0
    Sigma_true_mat = sigma_corr_Lower @ sigma_corr_Lower.T
    # print(f"q0Sigmaq0 = {np.sum(Sigma_true_mat)}")

    zeta = 0.0

    b = 1.0
    # b = 2.0  # MC: J0 = 270.3106608182768

    multiExec_formula = multiExec_analytical(
        T=T,
        N=N,
        d_asset=dim,
        S0=S0,
        Q0=np.ones(dim),
        kappa_true_mat=kappa_true_mat,
        alpha_mat=alpha_mat,
        eta_true_mat=eta_true_mat,
        eta_true_inv=eta_true_inv,
        Sigma_true_mat=Sigma_true_mat,
        Lambda=Lambda,
        zeta=zeta,
        b=b
    )

    print(f"analytical: J0 = {multiExec_formula.cal_val()}")
    multiExec_formula.cal_qt(plot=True, save_plot=False)

    multiExec_MC = multiExec_MonteCarlo(
        T=T,
        N=N,
        d_asset=dim,
        S0=S0,
        Q0=np.array(Q0),
        kappa_true_mat=kappa_true_mat,
        alpha_mat=alpha_mat,
        eta_true_mat=eta_true_mat,
        eta_true_inv=eta_true_inv,
        Sigma_true_mat=Sigma_true_mat,
        sigma_corr_Lower_tau=sigma_corr_Lower_tau,
        Lambda=Lambda,
        zeta=zeta,
        b=b
    )

    num_sample = 100
    J0_MC, running_r, qt, S_train = multiExec_MC.cal_val(num_sample=num_sample)
    print(f"MC: J0 = {J0_MC}")

    # print(f"calculating reward_to_go")
    # # running_r: (num_sample, N+1), qt: (d_asset, N+1), S_train: (num_sample, d_asset, N+1)
    # reward_to_go = running_r.sum(axis=1).reshape(-1, 1) - running_r.cumsum(axis=1) + running_r
    # # (num_sample, N+1)
    #
    # # squareLoss = (reward_to_go - reward_to_go.mean(axis=1).reshape(-1,1))**2 * tau
    # # MLoss = np.mean(squareLoss.sum(axis=1))
    # # print(f"MLoss={MLoss}")
    #
    # MLoss = []
    # for i in range(num_sample):
    #     if i % 10 == 0:
    #         print(f"path {i}, MLoss={np.mean(MLoss)}")
    #     MLoss_path = 0
    #     for j in range(N):
    #         S = S_train[i, :, j]
    #         q = qt[:, j]
    #         time_remain = T-j*tau
    #         multiExec_formula = multiExec_analytical(
    #             T=time_remain,
    #             N=N,
    #             d_asset=dim,
    #             S0=S, # (dim,)
    #             Q0=q,
    #             kappa_true_mat=kappa_true_mat,
    #             alpha_mat=alpha_mat,
    #             eta_true_mat=eta_true_mat,
    #             eta_true_inv=eta_true_inv,
    #             Sigma_true_mat=Sigma_true_mat,
    #             Lambda=Lambda,
    #             zeta=zeta,
    #             Bt_list=[np.eye(dim)],
    #             b=b
    #         )
    #
    #         J0_formula = multiExec_formula.cal_val()
    #         rtg = reward_to_go[i, j]
    #         MLoss_path += (rtg - J0_formula) ** 2 * tau
    #         # print(f"MLoss_path={MLoss_path}")
    #     MLoss.append(MLoss_path)
    #
    # print(f"MLoss={np.mean(MLoss)}")  # MLoss=0.0006374886248762843

def test_2d():
    T = 1 / 250
    N = 2000
    tau = T / N

    dim = 2

    S0 = np.array([100] * dim)
    sigma = np.array([0.3, 0.4]) * S0
    print(f"sigma={sigma}")

    Q0 = [5e5] * dim


    corr_mat = np.diag(np.ones(dim))
    corr_mat[0, 1] = 0.6
    corr_mat[1, 0] = 0.6
    '''self.corr_mat.shape = (d_asset, d_asset)'''
    print(f"corr_mat={corr_mat}")

    corr_Lower = np.linalg.cholesky(corr_mat)
    sigma_corr_Lower = corr_Lower * sigma.reshape(-1, 1)
    sigma_corr_Lower_tau = sigma_corr_Lower * np.sqrt(tau)
    Sigma_true_mat = sigma_corr_Lower @ sigma_corr_Lower.T

    kappa_true = np.array([1.5e-7, 2.5e-7]) * np.array(Q0)
    kappa_true_mat = np.diag(kappa_true)  # (dim, dim)
    kappa_true_mat[0, 1] = 0.8 * kappa_true[0]
    kappa_true_mat[1, 0] = 0.8 * kappa_true[0]

    print(f"perm_impact mat = {kappa_true_mat}")
    print(f"cost by perm_impact={-np.sum(kappa_true_mat)/2}")

    alpha_mat = 1000 * kappa_true_mat

    eta_true = np.array([1e-8/2, 1e-8]) * np.array(Q0)
    eta_true_mat = np.diag(eta_true)
    eta_true_inv = np.linalg.inv(eta_true_mat)

    Lambda_x = 3
    Lambda = 10 ** Lambda_x * 1e-8 * 5e5

    zeta = 0.0

    b = 1.0
    # b = 2.0  # MC: J0 = 270.3106608182768

    multiExec_formula = multiExec_analytical(
        T=T,
        N=N,
        d_asset=dim,
        S0=S0,
        Q0=np.ones(dim),
        kappa_true_mat=kappa_true_mat,
        alpha_mat=alpha_mat,
        eta_true_mat=eta_true_mat,
        eta_true_inv=eta_true_inv,
        Sigma_true_mat=Sigma_true_mat,
        Lambda=Lambda,
        zeta=zeta,
        b=b
    )

    print(f"analytical: J0 = {multiExec_formula.cal_val()}")
    multiExec_formula.cal_qt(plot=True, save_plot=False)

    multiExec_MC = multiExec_MonteCarlo(
        T=T,
        N=N,
        d_asset=dim,
        S0=S0,
        Q0=np.array(Q0),
        kappa_true_mat=kappa_true_mat,
        alpha_mat=alpha_mat,
        eta_true_mat=eta_true_mat,
        eta_true_inv=eta_true_inv,
        Sigma_true_mat=Sigma_true_mat,
        sigma_corr_Lower_tau=sigma_corr_Lower_tau,
        Lambda=Lambda,
        zeta=zeta,
        b=b
    )

    num_sample = 1000
    J0_MC, running_r, qt, S_train = multiExec_MC.cal_val(num_sample=num_sample)
    print(f"MC: J0 = {J0_MC}")

def cal_value_func(
        theta1_test, theta2_test, theta3_test, phi1, phi2,
        q, S, tau, zeta, Lambda, S0
):
    theta3_phi1 = theta3_test * phi1
    lambda_theta2_S02_phi1 = Lambda * theta2_test * S0**2 / phi1
    V_theta_mat = q * S - q ** 2 * (
            theta1_test +
            (theta3_phi1 + lambda_theta2_S02_phi1) / np.tanh(phi1 * tau) +
            (theta3_phi1 - lambda_theta2_S02_phi1) * phi1 * tau / np.sinh(phi1 * tau) ** 2
    ) / 2

    return V_theta_mat

def test_1d():
    T = 1 / 250
    N = 1000
    tau = T / N

    dim = 1

    S0 = np.array([100] * dim)
    sigma = np.linspace(0.3, 0.5, dim) * S0


    def gen_rho(d_asset: int, rho_low: float = 0.05, rho_high: float = 0.6):
        num_pair = int(round(d_asset * (d_asset - 1) / 2))
        for rho in np.linspace(rho_high, rho_low, num_pair):
            yield rho

    corr_mat = np.diag(np.ones(dim))
    '''self.corr_mat.shape = (d_asset, d_asset)'''
    rho_generator = gen_rho(dim)
    for i in range(dim):
        for j in range(i + 1, dim):
            rho = next(rho_generator)
            corr_mat[i, j] = rho
            corr_mat[j, i] = rho

    print(f"corr_mat={corr_mat}")

    corr_Lower = np.linalg.cholesky(corr_mat)
    sigma_corr_Lower = corr_Lower * sigma.reshape(-1, 1)
    sigma_corr_Lower_tau = sigma_corr_Lower * np.sqrt(tau)

    kappa_true = 2.5e-7 * 5e5
    kappa_true_mat = np.diag([kappa_true] * dim)
    if dim > 1:
        kappa_true_mat[0, 1] = 0.8 * kappa_true
        kappa_true_mat[1, 0] = 0.8 * kappa_true

    # kappa_true_mat = np.zeros((dim, dim))

    print(f"cost by perm_impact={-np.sum(kappa_true_mat)/2}")

    alpha_mat = 1000 * kappa_true_mat

    eta_true = 1e-8 * 5e5
    eta_true_mat = np.diag([eta_true] * dim)
    eta_true_inv = np.linalg.inv(eta_true_mat)

    Lambda_x = 3
    Lambda = np.power(10, Lambda_x) * eta_true
    # Lambda = 0
    Sigma_true_mat = sigma_corr_Lower @ sigma_corr_Lower.T
    print(f"q0Sigmaq0 = {np.sum(Sigma_true_mat)}")

    zeta = 0.0

    # b = 1.0
    b = 0.5

    multiExec_formula = multiExec_analytical(
        T=T,
        N=N,
        d_asset=dim,
        S0=S0,
        Q0=np.ones(dim),
        kappa_true_mat=kappa_true_mat,
        alpha_mat=alpha_mat,
        eta_true_mat=eta_true_mat,
        eta_true_inv=eta_true_inv,
        Sigma_true_mat=Sigma_true_mat,
        Lambda=Lambda,
        zeta=zeta,
        b=b
    )

    print(f"analytical: J0 = {multiExec_formula.cal_val()}")
    # multiExec_formula.cal_qt(plot=True, save_plot=False)

    K_true = np.sqrt(Lambda * (sigma[0]/S0[0]) ** 2 * S0[0] ** 2 / eta_true)
    V0_formula = cal_value_func(
        theta1_test=kappa_true, theta2_test=(sigma[0]/S0[0])**2, theta3_test=eta_true,
        phi1=b * K_true, phi2=0,
        q=1, S=S0[0], tau=T, zeta=0, Lambda=Lambda, S0=S0[0]
    )
    print(f"cloased-form: V0 = {V0_formula}")

    multiExec_MC = multiExec_MonteCarlo(
        T=T,
        N=N,
        d_asset=dim,
        S0=S0,
        Q0=np.ones(dim)*5e5,
        kappa_true_mat=kappa_true_mat,
        alpha_mat=alpha_mat,
        eta_true_mat=eta_true_mat,
        eta_true_inv=eta_true_inv,
        Sigma_true_mat=Sigma_true_mat,
        sigma_corr_Lower_tau=sigma_corr_Lower_tau,
        Lambda=Lambda,
        zeta=zeta,
        b=b
    )
    print(f"MC: J0 = {multiExec_MC.cal_val(num_sample=100000)}")






if __name__ == '__main__':
    test_3d()
    # test_2d()
    # test_1d()
