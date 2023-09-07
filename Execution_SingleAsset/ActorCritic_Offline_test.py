'''
Unit tests for ActorCritic_Offline
'''

import numpy as np
import matplotlib.pyplot as plt
import time
from Execution.ActorCritic_Offline import EMQVFD
from Execution.PE_GD_Offline import gen_q_r
from Execution.PEPG_FDvsFormula import nu_t, dnut_dphi1, q_t, dqt_dphi1, \
    gen_r_theta, gen_r_theta_grad, gen_dr_dphi, \
    gen_D_theta, gen_dD_dtheta1, gen_dD_dtheta2, gen_dD_dphi1, cal_value_grad_vec, cal_dG_dphi_formula
from scipy import sparse
from scipy.sparse.linalg import splu
import pytest
from Execution.Env_Param import env_param

class config:
    def __init__(self):
        self.T = 1 / 250  # year
        self.N = 5000
        self.Delta_t = self.T / self.N

        self.S0 = 100
        self.q0 = 1
        self.Q0 = 5e5

        self.kappa_true = 2.5e-7 * self.Q0  # =1/8
        self.eta_true = 1e-8 * self.Q0  # =1/8/(0.01 * 5e6 / 5e5)
        self.sigma_true = 0.3  # annually

        self.Lambda_x = 0
        # self.Lambda_x = 3
        # self.Lambda_x = 4
        self.Lambda = round(np.power(10, self.Lambda_x) * self.eta_true, 8)  # $^{-1} [50, 5e2, 5e3]
        self.zeta = 5
        # self.zeta = 0

        K_true = np.sqrt(self.Lambda * self.sigma_true ** 2 * self.S0 ** 2 / self.eta_true)  # [300, 670.82, 948.683, 2121.32]
        if self.Lambda_x == 0:
            phi1_behavior = 100
            behavior_coef = round(phi1_behavior / K_true, 1)

            alpha_theta1 = 46
            alpha_theta2 = 480
            alpha_theta3 = 1.9e-3

        elif self.Lambda_x == 3:
            behavior_coef = 2
            phi1_behavior = behavior_coef * K_true

            alpha_theta1 = 2.7e2
            alpha_theta2 = 7.5
            alpha_theta3 = 5.5e-3

        elif self.Lambda_x == 4:
            behavior_coef = 0.5
            phi1_behavior = behavior_coef * K_true

            alpha_theta1 = 55
            alpha_theta2 = 0.12
            alpha_theta3 = 1.5e-3

        theta_init = 2
        theta1_list = [theta_init * self.kappa_true]
        theta2_list = [theta_init * self.sigma_true ** 2]
        theta3_list = [theta_init * self.eta_true]

        phi2_behavior = behavior_coef * (1 / (2 * self.eta_true))

        PATH_NUM = 1_000
        self.scheme = 'CN'

        Srange_pct = 0.2
        self.Smax, self.Smin = self.S0 * (1 + Srange_pct), self.S0 * (1 - Srange_pct)
        self.dim = 50
        self.h = (self.Smax - self.Smin) / self.dim
        self.Svec = np.linspace(start=self.Smin, stop=self.Smax, num=self.dim, endpoint=False)  # including Smin

        self.EMQVFD_agent = EMQVFD(
            q0=self.q0,
            S0=self.S0,
            T=self.T,
            zeta=self.zeta,
            Lambda=self.Lambda,
            phi1=phi1_behavior,
            phi2=phi2_behavior,
            theta1=theta1_list[-1],
            theta2=theta2_list[-1],
            theta3=theta3_list[-1],
            alpha_phi1=0,
            alpha_phi2=0,
            alpha_theta1=alpha_theta1,
            alpha_theta2=alpha_theta2,
            alpha_theta3=alpha_theta3,
            N=self.N,
            delta=self.Delta_t,
            dim=self.dim,
            h=self.h,
            Svec=self.Svec,
            Smin=self.Smin,
            Smax=self.Smax,
            scheme=self.scheme
        )

        np.random.seed(1234)

        dWt = self.sigma_true * self.S0 * np.random.normal(loc=0.0, scale=np.sqrt(self.Delta_t), size=(PATH_NUM, self.N))
        S0_array = self.S0 * np.ones(PATH_NUM).reshape(-1, 1)
        S = np.cumsum(np.concatenate([S0_array, dWt], axis=1), axis=1)

        self.q, self.running_r, self.S = gen_q_r(
            phi1_behavior=self.EMQVFD_agent.phi1,
            phi2_behavior=self.EMQVFD_agent.phi2,
            S=S,
            q0=self.q0,
            N=self.N,
            Delta_t=self.Delta_t,
            zeta=self.EMQVFD_agent.zeta,
            kappa_true=self.kappa_true,
            eta_true=self.eta_true,
            Lambda=self.EMQVFD_agent.Lambda
        )

        self.t = np.array([i * self.Delta_t for i in range(self.N)]).reshape(1, -1)
        self.epsilon = 1e-8

Config = config()

class TestActorCritic_Offline:

    def test_nu_dnu(self):

        nu_class = Config.EMQVFD_agent.nu_t(Config.t)
        nu_fun = nu_t(q0=Config.q0, T=Config.T, t=Config.t, phi1=Config.EMQVFD_agent.phi1)
        assert np.mean(abs(nu_class - nu_fun)) < Config.epsilon

        dnut_dphi1_class = Config.EMQVFD_agent.dnut_dphi1(Config.t)
        dnut_dphi1_fun = dnut_dphi1(q0=Config.q0, T=Config.T, t=Config.t, phi1=Config.EMQVFD_agent.phi1)
        assert np.mean(abs(dnut_dphi1_class - dnut_dphi1_fun)) < Config.epsilon

    def test_q_dq(self):
        qt_class = Config.EMQVFD_agent.q_t(Config.t)
        qt_fun = q_t(q0=Config.q0, T=Config.T, t=Config.t, phi1=Config.EMQVFD_agent.phi1)
        assert np.mean(abs(qt_class - qt_fun)) < Config.epsilon

        dqt_dphi1_class = Config.EMQVFD_agent.dqt_dphi1(Config.t)
        dqt_dphi1_fun = dqt_dphi1(q0=Config.q0, T=Config.T, t=Config.t, phi1=Config.EMQVFD_agent.phi1)
        assert np.mean(abs(dqt_dphi1_class - dqt_dphi1_fun)) < Config.epsilon

    def test_r_drdtheta_drdphi(self):
        for t in Config.t[0, :]:
            gen_r_theta_class = Config.EMQVFD_agent.gen_r_theta(t)
            gen_r_theta_func = gen_r_theta(
                q0=Config.q0,
                T=Config.T,
                t=t,
                phi1=Config.EMQVFD_agent.phi1,
                Lambda=Config.EMQVFD_agent.Lambda,
                S0=Config.S0,
                Svec=Config.Svec,
                theta2=Config.EMQVFD_agent.theta2,
                theta3=Config.EMQVFD_agent.theta3
            ) + (-Config.EMQVFD_agent.theta3 * Config.EMQVFD_agent.zeta * Config.EMQVFD_agent.phi2)
            assert np.mean(abs(gen_r_theta_class - gen_r_theta_func)) < Config.epsilon

            gen_dr_dtheta1_class, gen_dr_dtheta2_class, gen_dr_dtheta3_class = \
                Config.EMQVFD_agent.gen_dr_dtheta(t)
            gen_dr_dtheta1_func, gen_dr_dtheta2_func, gen_dr_dtheta3_func = gen_r_theta_grad(
                q0=Config.q0,
                T=Config.T,
                t=t,
                phi1=Config.EMQVFD_agent.phi1,
                Lambda=Config.EMQVFD_agent.Lambda,
                S0=Config.S0,
                dim=Config.dim
            )
            gen_dr_dtheta3_func += (-Config.EMQVFD_agent.zeta * Config.EMQVFD_agent.phi2)
            assert np.mean(abs(gen_dr_dtheta1_class - gen_dr_dtheta1_func)) < Config.epsilon
            assert np.mean(abs(gen_dr_dtheta2_class - gen_dr_dtheta2_func)) < Config.epsilon
            assert np.mean(abs(gen_dr_dtheta3_class - gen_dr_dtheta3_func)) < Config.epsilon

            gen_dr_dphi1_class, gen_dr_dphi2_class = Config.EMQVFD_agent.gen_dr_dphi(t)
            gen_dr_dphi1_func, gen_dr_dphi2_func = gen_dr_dphi(
                q0=Config.q0,
                T=Config.T,
                t=t,
                phi1=Config.EMQVFD_agent.phi1,
                Lambda=Config.EMQVFD_agent.Lambda,
                zeta=Config.EMQVFD_agent.zeta,
                S0=Config.S0,
                Svec=Config.Svec,
                theta2=Config.EMQVFD_agent.theta2,
                theta3=Config.EMQVFD_agent.theta3
            )
            assert np.mean(abs(gen_dr_dphi1_class - gen_dr_dphi1_func)) < Config.epsilon
            assert np.mean(abs(gen_dr_dphi2_class - gen_dr_dphi2_func)) < Config.epsilon
            # assert np.mean(abs(gen_dr_dphi2_class - (Config.zeta * (Config.T - t) * ))) < Config.epsilon

    def test_D_dDdtheta_dDdphi(self):
        for t in Config.t[0, :]:
            gen_D_theta_class = Config.EMQVFD_agent.gen_D_theta(t)
            gen_D_theta_func = gen_D_theta(
                dim=Config.dim,
                q0=Config.q0,
                T=Config.T,
                t=t,
                phi1=Config.EMQVFD_agent.phi1,
                h=Config.h,
                S0=Config.S0,
                theta1=Config.EMQVFD_agent.theta1,
                theta2=Config.EMQVFD_agent.theta2
            )
            assert np.mean(abs(gen_D_theta_class - gen_D_theta_func)) < Config.epsilon

            gen_dD_dtheta1_class = Config.EMQVFD_agent.gen_dD_dtheta1(t)
            gen_dD_dtheta1_func = gen_dD_dtheta1(
                dim=Config.dim,
                q0=Config.q0,
                T=Config.T,
                t=t,
                phi1=Config.EMQVFD_agent.phi1,
                h=Config.h
            )
            assert np.mean(abs(gen_dD_dtheta1_class - gen_dD_dtheta1_func)) < Config.epsilon

            gen_dD_dtheta2_class = Config.EMQVFD_agent.gen_dD_dtheta2()
            gen_dD_dtheta2_func = gen_dD_dtheta2(
                dim=Config.dim,
                S0=Config.S0,
                h=Config.h
            )
            assert np.mean(abs(gen_dD_dtheta2_class - gen_dD_dtheta2_func)) < Config.epsilon

            gen_dD_dphi1_class = Config.EMQVFD_agent.gen_dD_dphi1(t)
            gen_dD_dphi1_func = gen_dD_dphi1(
                theta1=Config.EMQVFD_agent.theta1,
                phi1=Config.EMQVFD_agent.phi1,
                h=Config.h,
                q0=Config.q0,
                T=Config.T,
                t=t,
                dim=Config.dim
            )
            assert np.mean(abs(gen_dD_dphi1_class - gen_dD_dphi1_func)) < Config.epsilon

    def test_gen_A_B_dtheta(self):
        identityMat = np.eye(Config.dim)
        for t in Config.t[0, :]:
            D_theta = Config.EMQVFD_agent.gen_D_theta(t=t)
            dD_dtheta1 = Config.EMQVFD_agent.gen_dD_dtheta1(t=t)
            dD_dtheta2 = Config.EMQVFD_agent.gen_dD_dtheta2()
            dD_dtheta3 = Config.EMQVFD_agent.gen_dD_dtheta3()
            A_theta_LU, dA_dtheta1, dA_dtheta2, dA_dtheta3, B_theta, dB_dtheta1, dB_dtheta2, dB_dtheta3 = \
                Config.EMQVFD_agent.gen_A_B_dtheta(
                    D_theta=D_theta,
                    dD_dtheta1=dD_dtheta1,
                    dD_dtheta2=dD_dtheta2,
                    dD_dtheta3=dD_dtheta3
                )

            D_theta_func = gen_D_theta(
                dim=Config.dim,
                q0=Config.q0,
                T=Config.T,
                t=t,
                phi1=Config.EMQVFD_agent.phi1,
                h=Config.h,
                S0=Config.S0,
                theta1=Config.EMQVFD_agent.theta1,
                theta2=Config.EMQVFD_agent.theta2
            )
            dD_dtheta1_func = gen_dD_dtheta1(
                dim=Config.dim,
                q0=Config.q0,
                T=Config.T,
                t=t,
                phi1=Config.EMQVFD_agent.phi1,
                h=Config.h
            )
            dD_dtheta2_func = gen_dD_dtheta2(
                dim=Config.dim,
                S0=Config.S0,
                h=Config.h
            )
            dD_dtheta3_func = np.zeros(D_theta.shape)
            A_theta_func = sparse.csc_matrix(identityMat + Config.Delta_t / 2 * D_theta_func)  # csc format
            A_theta_LU_func = splu(A_theta_func)

            dA_dtheta1_func = Config.Delta_t / 2 * dD_dtheta1_func
            dA_dtheta2_func = Config.Delta_t / 2 * dD_dtheta2_func
            dA_dtheta3_func = Config.Delta_t / 2 * dD_dtheta3_func

            B_theta_func = identityMat - Config.Delta_t / 2 * D_theta_func  # no need to convert to sparse matrix

            dB_dtheta1_func = -dA_dtheta1_func
            dB_dtheta2_func = -dA_dtheta2_func
            dB_dtheta3_func = -dA_dtheta3_func

            # assert np.mean(abs(A_theta_LU - A_theta_LU_func)) < Config.epsilon
            assert np.mean(abs(dA_dtheta1 - dA_dtheta1_func)) < Config.epsilon
            assert np.mean(abs(dA_dtheta2 - dA_dtheta2_func)) < Config.epsilon
            assert np.mean(abs(dA_dtheta3 - dA_dtheta3_func)) < Config.epsilon
            assert np.mean(abs(B_theta - B_theta_func)) < Config.epsilon
            assert np.mean(abs(dB_dtheta1 - dB_dtheta1_func)) < Config.epsilon
            assert np.mean(abs(dB_dtheta2 - dB_dtheta2_func)) < Config.epsilon
            assert np.mean(abs(dB_dtheta3 - dB_dtheta3_func)) < Config.epsilon

    def test_V_dVdtheta_dVdphi(self):
        Config.EMQVFD_agent.cal_value_dVdtheta_vec()
        vVec_list_class = Config.EMQVFD_agent.vVec_list
        grad_theta1Vec_list_class = Config.EMQVFD_agent.grad_theta1Vec_list
        grad_theta2Vec_list_class = Config.EMQVFD_agent.grad_theta2Vec_list
        grad_theta3Vec_list_class = Config.EMQVFD_agent.grad_theta3Vec_list

        Config.EMQVFD_agent.cal_dGdphi_vec()
        grad_phi1Vec_list_class = Config.EMQVFD_agent.grad_phi1Vec_list
        grad_phi2Vec_list_class = Config.EMQVFD_agent.grad_phi2Vec_list

        vVec_list_func, \
        grad_theta1Vec_list_func, grad_theta2Vec_list_func, grad_theta3Vec_list_func, \
        grad_phi1Vec_list_func, grad_phi2Vec_list_func = cal_value_grad_vec(
            scheme=Config.scheme,
            q0=Config.q0,
            T=Config.T,
            N=Config.N,
            phi1=Config.EMQVFD_agent.phi1,
            Svec=Config.Svec,
            dim=Config.dim,
            h=Config.h,
            delta=Config.Delta_t,
            S0=Config.S0,
            Lambda=Config.Lambda,
            theta1=Config.EMQVFD_agent.theta1,
            theta2=Config.EMQVFD_agent.theta2,
            theta3=Config.EMQVFD_agent.theta3,
            zeta=Config.EMQVFD_agent.zeta
        )
        entropy = Config.zeta * Config.T * (
                np.log(2 * np.pi * np.e * Config.zeta * Config.EMQVFD_agent.phi2) / 2
                - Config.EMQVFD_agent.theta3 * Config.EMQVFD_agent.phi2
        )

        theta3_phi1 = Config.EMQVFD_agent.theta3 * Config.EMQVFD_agent.phi1
        lambda_theta2_S02_phi1 = Config.EMQVFD_agent.Lambda * Config.EMQVFD_agent.theta2 * Config.EMQVFD_agent.S0 ** 2 / Config.EMQVFD_agent.phi1
        phi1_T = Config.EMQVFD_agent.phi1 * Config.T
        V_vec_formula = Config.EMQVFD_agent.q0 * Config.EMQVFD_agent.Svec - Config.EMQVFD_agent.q0 ** 2 / 2 * (
                Config.EMQVFD_agent.theta1 +
                (theta3_phi1 + lambda_theta2_S02_phi1) / np.tanh(phi1_T) +
                (theta3_phi1 - lambda_theta2_S02_phi1) * phi1_T / np.sinh(phi1_T) ** 2
        ) + entropy

        Smin, Smax = 95, 105
        Slist = Config.EMQVFD_agent.Svec.tolist()
        fig = plt.figure(figsize=(12, 12), dpi=300)
        ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=1)
        # ax1.plot(Slist, vVec_list_class[0].tolist(), label=f"V, class")
        # ax1.plot(Slist, vVec_list_func[0].tolist(), label=f"V, func")
        # ax1.plot(Slist, V_vec_formula.tolist(), c='red', label=f"Formula")

        # ax1.plot(Slist, (vVec_list_func[0] + entropy - vVec_list_class[0]).tolist(), label=f"V, class")

        plt.legend()

        ax2 = plt.subplot2grid((3, 2), (0, 1), colspan=1)
        # ax2.plot(Slist, grad_theta2Vec_list_class[0].tolist(), label=f"dV_dtheta1, class")
        # ax2.plot(Slist, grad_theta2Vec_list_func[0].tolist(), label=f"dV_dtheta1, func")
        # ax2.hlines(y=-Config.q0 ** 2 / 2, xmin=Slist[0], xmax=Slist[-1], colors='red', label=f"Formula")
        ax2.plot(Slist, (grad_theta2Vec_list_class[0] - grad_theta2Vec_list_func[0]).tolist(), label=f"dV_dtheta1, class")
        plt.legend()

        ax3 = plt.subplot2grid((3, 2), (1, 0), colspan=1)
        # ax3.plot(Slist, grad_theta2Vec_list_class[0].tolist(), label=f"dV_dtheta2, class")
        # ax3.plot(Slist, grad_theta2Vec_list_func[0].tolist(), label=f"dV_dtheta2, func")
        # dV_dtheta2_0 = - Config.q0 ** 2 * Config.Lambda * Config.S0 ** 2 / (2 * Config.EMQVFD_agent.phi1) * (
        #         1 / np.tanh(phi1_T) - phi1_T / np.sinh(phi1_T) ** 2
        # )
        # ax3.hlines(y=dV_dtheta2_0, xmin=Slist[0], xmax=Slist[-1], colors='red', label=f"Formula")
        ax3.plot(Slist, (grad_theta2Vec_list_class[0] - grad_theta2Vec_list_func[0]).tolist(), label=f"dV_dtheta2, class")
        plt.legend()

        ax4 = plt.subplot2grid((3, 2), (1, 1), colspan=1)
        # ax4.plot(Slist, grad_theta3Vec_list_class[0].tolist(), label=f"dV_dtheta3, class")
        # ax4.plot(Slist, (grad_theta3Vec_list_func[0] - Config.zeta * Config.EMQVFD_agent.phi2 * Config.T).tolist(), label=f"dV_dtheta3, func")
        # dV_dtheta3_0 = - Config.q0 ** 2 * Config.EMQVFD_agent.phi1 * (
        #         1 / np.tanh(phi1_T) + phi1_T / np.sinh(phi1_T) ** 2
        # ) / 2 - Config.zeta * Config.EMQVFD_agent.phi2 * Config.T
        # ax4.hlines(y=dV_dtheta3_0, xmin=Slist[0], xmax=Slist[-1], colors='red', label=f"Formula")
        ax4.plot(Slist, (grad_theta3Vec_list_func[0] - Config.zeta * Config.EMQVFD_agent.phi2 * Config.T - grad_theta3Vec_list_class[0]).tolist(), label=f"dV_dtheta3, class")
        plt.legend()

        ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=1)
        ax5.plot(Slist, grad_phi1Vec_list_class[0].tolist(), label=f"dG_dphi1, class")
        ax5.plot(Slist, grad_phi1Vec_list_func[0].tolist(), label=f"dG_dphi1, func")
        dG_dphi1_formula, dG_dphi2_formula = cal_dG_dphi_formula(
            phi1_behavior=Config.EMQVFD_agent.phi1,
            phi2_behavior=Config.EMQVFD_agent.phi2,
            q0=Config.q0,
            S0=Config.S0,
            eta_true=Config.eta_true,
            sigma_true=Config.sigma_true,
            Lambda=Config.Lambda,
            T=Config.T,
            zeta=Config.zeta
        )
        ax5.hlines(y=dG_dphi1_formula, xmin=Slist[0], xmax=Slist[-1], colors='red', label=f"Formula")
        plt.legend()

        ax6 = plt.subplot2grid((3, 2), (2, 1), colspan=1)
        ax6.plot(Slist, grad_phi2Vec_list_class[0].tolist(), label=f"dG_dphi2, class")
        ax6.plot(Slist, grad_phi2Vec_list_func[0].tolist(), label=f"dG_dphi2, func")
        ax6.hlines(y=dG_dphi2_formula, xmin=Slist[0], xmax=Slist[-1], colors='red', label=f"Formula")
        plt.legend()

        # fig.suptitle(f"Lam{Lambda_x}, phi1={phi1 / K_true:.1f}K, N={N}, dim={dim}, scheme={scheme}, {run_time:.3f}s")
        # plt.savefig(f'PE_Exec_FDvsFormula_Lam{Lambda_x}_{scheme}.pdf')
        plt.show()


        for i in range(Config.N):
            t = i * Config.Delta_t
            entropy = Config.zeta * (Config.T - t) * (
                    np.log(2 * np.pi * np.e * Config.zeta * Config.EMQVFD_agent.phi2) / 2
                    - Config.EMQVFD_agent.theta3 * Config.EMQVFD_agent.phi2
            )
            # assert np.mean(abs(vVec_list_class[i] - (vVec_list_func[i] + entropy)) / abs(vVec_list_func[i] + entropy)) < Config.epsilon
            # assert np.mean(abs(grad_theta1Vec_list_class[i] - grad_theta1Vec_list_func[i]) / abs(grad_theta1Vec_list_func[i])) < Config.epsilon
            # assert np.mean(abs(grad_theta2Vec_list_class[i] - grad_theta2Vec_list_func[i]) / abs(grad_theta2Vec_list_func[i])) < Config.epsilon
            # assert np.mean(abs(grad_theta3Vec_list_class[i] - (grad_theta3Vec_list_func[i] - Config.zeta * Config.EMQVFD_agent.phi2 * (Config.T-t))) / abs((grad_theta3Vec_list_func[i]- Config.zeta * Config.EMQVFD_agent.phi2 * (Config.T-t)))) < Config.epsilon
            # assert np.mean(abs(grad_phi1Vec_list_class[i] - grad_phi1Vec_list_func[i]) / abs(grad_phi1Vec_list_func[i])) < Config.epsilon
            # assert np.mean(abs(grad_phi2Vec_list_class[i] - grad_phi2Vec_list_func[i]) / abs(grad_phi2Vec_list_func[i])) < Config.epsilon
            print(f"-------------------i={i}-------------------")
            print(np.mean(abs(vVec_list_class[i] - (vVec_list_func[i] + entropy)) / abs(
                vVec_list_func[i] + entropy)))
            print(np.mean(abs(grad_theta1Vec_list_class[i] - grad_theta1Vec_list_func[i]) / abs(
                grad_theta1Vec_list_func[i])))
            print(np.mean(abs(grad_theta2Vec_list_class[i] - grad_theta2Vec_list_func[i]) / abs(
                grad_theta2Vec_list_func[i])))
            print(np.mean(abs(grad_theta3Vec_list_class[i] - (
                        grad_theta3Vec_list_func[i] - Config.zeta * Config.EMQVFD_agent.phi2 * (Config.T - t))) / abs((
                                                                                                                                  grad_theta3Vec_list_func[
                                                                                                                                      i] - Config.zeta * Config.EMQVFD_agent.phi2 * (
                                                                                                                                              Config.T - t)))))
            print(np.mean(abs(grad_phi1Vec_list_class[i] - grad_phi1Vec_list_func[i]) / abs(
                grad_phi1Vec_list_func[i])))
            print(np.mean(abs(grad_phi2Vec_list_class[i] - grad_phi2Vec_list_func[i]) / abs(
                grad_phi2Vec_list_func[i])))
            print(f"-------------------------------------------")



    def test_PE_GD_Offline(self, Lambda_x):
        RunningStartTime = time.perf_counter()

        T = 1 / 250  # year
        N = 5000
        Delta_t = T / N

        S0 = 100
        q0 = 1
        Q0 = 5e5

        kappa_true = 2.5e-7 * Q0  # =1/8
        eta_true = 1e-8 * Q0  # =1/8/(0.01 * 5e6 / 5e5)
        sigma_true = 0.3  # annually

        Lambda = round(np.power(10, Lambda_x) * eta_true, 8)  # $^{-1} [50, 5e2, 5e3]
        zeta = 5

        K_true = np.sqrt(Lambda * sigma_true ** 2 * S0 ** 2 / eta_true)  # [300, 670.82, 948.683, 2121.32]
        if Lambda_x == 0:
            # phi1_behavior = 100
            phi1_behavior = 31.9 * K_true
            behavior_coef = round(phi1_behavior / K_true, 1)

            alpha_theta1 = 46
            alpha_theta2 = 480
            alpha_theta3 = 1.9e-3

            # alpha_theta1 = 10
            # alpha_theta2 = 100
            # alpha_theta3 = 1.9e-4

        elif Lambda_x == 3:
            behavior_coef = 2
            phi1_behavior = behavior_coef * K_true

            # alpha_theta1 = 2.7e2
            # alpha_theta2 = 7.5
            # alpha_theta3 = 5.5e-3

            alpha_theta1 = 2.63e2
            alpha_theta2 = 7.1
            alpha_theta3 = 5.5e-3

        elif Lambda_x == 4:
            behavior_coef = 0.5
            phi1_behavior = behavior_coef * K_true

            alpha_theta1 = 55
            alpha_theta2 = 0.12
            alpha_theta3 = 1.5e-3

        theta_init = 2
        theta1_list = [theta_init * kappa_true]
        theta2_list = [theta_init * sigma_true ** 2]
        theta3_list = [theta_init * eta_true]

        phi2_behavior = behavior_coef * (1 / (2 * eta_true))
        print(f"K_true={K_true}, phi1_behavior={phi1_behavior}, phi2_behavior={phi2_behavior}")

        PATH_NUM = 1_000
        scheme = 'CN'

        Srange_pct = 0.2
        Smax, Smin = S0 * (1 + Srange_pct), S0 * (1 - Srange_pct)
        dim = 50
        h = (Smax - Smin) / dim
        Svec = np.linspace(start=Smin, stop=Smax, num=dim, endpoint=False)  # including Smin

        EMQVFD_agent = EMQVFD(
            q0=q0,
            S0=S0,
            T=T,
            zeta=zeta,
            Lambda=Lambda,
            phi1=phi1_behavior,
            phi2=phi2_behavior,
            theta1=theta1_list[-1],
            theta2=theta2_list[-1],
            theta3=theta3_list[-1],
            alpha_phi1=0,
            alpha_phi2=0,
            alpha_theta1=alpha_theta1,
            alpha_theta2=alpha_theta2,
            alpha_theta3=alpha_theta3,
            N=N,
            delta=Delta_t,
            dim=dim,
            h=h,
            Svec=Svec,
            Smin=Smin,
            Smax=Smax,
            scheme=scheme
        )

        np.random.seed(1234)

        dWt = sigma_true * S0 * np.random.normal(loc=0.0, scale=np.sqrt(Delta_t), size=(PATH_NUM, N))
        S0_array = S0 * np.ones(PATH_NUM).reshape(-1, 1)
        S = np.cumsum(np.concatenate([S0_array, dWt], axis=1), axis=1)

        q, running_r, S = gen_q_r(
            phi1_behavior=EMQVFD_agent.phi1,
            phi2_behavior=EMQVFD_agent.phi2,
            S=S,
            q0=q0,
            N=N,
            Delta_t=Delta_t,
            zeta=EMQVFD_agent.zeta,
            kappa_true=kappa_true,
            eta_true=eta_true,
            Lambda=EMQVFD_agent.Lambda
        )

        t = np.array([i * Delta_t for i in range(N)]).reshape(1, -1)
        tau = T - t

        EPOCH_NUM = 50

        derror_dtheta1_list = []
        derror_dtheta2_list = []
        derror_dtheta3_list = []
        MLoss_list = []

        print(f"Lam{Lambda_x}, "
              f"alpha_theta1={alpha_theta1}, "
              f"alpha_theta2={alpha_theta2}, "
              f"alpha_theta3={alpha_theta3}")

        check_error_RMSE_list, check_error_MAE_list = [], []
        time_cal_value_grad_vec_avg, time_cal_value_grad_grid_avg, time_ML_grad_avg = 0, 0, 0

        for epoch in range(1, EPOCH_NUM + 1):
            derror_dtheta1, derror_dtheta2, derror_dtheta3, MLoss, V_theta_grid = EMQVFD_agent.PE_1step(
                S=S[:, :-1],
                running_r_mat=running_r
            )

            derror_dtheta1_list.append(derror_dtheta1)
            derror_dtheta2_list.append(derror_dtheta2)
            derror_dtheta3_list.append(derror_dtheta3)
            MLoss_list.append(MLoss)

            theta1_list.append(EMQVFD_agent.theta1)
            theta2_list.append(EMQVFD_agent.theta2)
            theta3_list.append(EMQVFD_agent.theta3)

            print(f"{scheme}, {epoch}, "
                  f"theta1={theta1_list[-1]}, theta1*={kappa_true},"
                  f"theta2={theta2_list[-1]}, theta2*={sigma_true ** 2},"
                  f"theta3={theta3_list[-1]}, theta3*={eta_true}")

        RunningTime = round(time.perf_counter() - RunningStartTime, 3)
        print(f"Running time: {RunningTime} sec for {PATH_NUM} paths and {EPOCH_NUM} epochs.")


        ## Create figure
        fig = plt.figure(figsize=(12, 12), dpi=300)

        avg_window = max(1, int(len(theta2_list) * 0.1))
        theta1_learned = np.mean(theta1_list[-avg_window:]) / Q0
        theta2_learned = np.mean(theta2_list[-avg_window:])
        theta3_learned = np.mean(theta3_list[-avg_window:]) / Q0

        ax1 = plt.subplot(421)
        theta1_list_plot = [theta1 / Q0 for theta1 in theta1_list]
        ax1.plot(theta1_list_plot, label=f'{theta1_learned}')
        ax1.hlines(y=kappa_true / Q0, xmin=0, xmax=EPOCH_NUM, linestyles='--', colors='r', label=f'{kappa_true / Q0}')
        ax1.title.set_text('theta1')
        plt.legend()

        ax2 = plt.subplot(422)
        ax2.plot(derror_dtheta1_list, label=f'{derror_dtheta1_list[-1]}')
        ax2.hlines(y=0, xmin=0, xmax=EPOCH_NUM, linestyles='--', colors='r', label=f'ln_r={alpha_theta1}')
        ax2.title.set_text('derror_dtheta1')
        plt.legend()

        ax3 = plt.subplot(423)
        ax3.plot(theta2_list, label=f'{theta2_learned}')
        ax3.hlines(y=sigma_true ** 2, xmin=0, xmax=EPOCH_NUM, linestyles='--', colors='r', label=f'{sigma_true ** 2}')
        ax3.title.set_text('theta2')
        plt.legend()

        ax4 = plt.subplot(424)
        ax4.plot(derror_dtheta2_list, label=f'{derror_dtheta2_list[-1]}')
        ax4.hlines(y=0, xmin=0, xmax=EPOCH_NUM, linestyles='--', colors='r', label=f'ln_r={alpha_theta2}')
        ax4.title.set_text('derror_dtheta2')
        plt.legend()

        ax5 = plt.subplot(425)
        theta3_list_plot = [theta3 / Q0 for theta3 in theta3_list]
        ax5.plot(theta3_list_plot, label=f'{theta3_learned}')
        ax5.hlines(y=eta_true / Q0, xmin=0, xmax=EPOCH_NUM, linestyles='--', colors='r', label=f'{eta_true / Q0}')
        ax5.title.set_text('theta3')
        plt.legend()

        ax6 = plt.subplot(426)
        ax6.plot(derror_dtheta3_list, label=f'{derror_dtheta3_list[-1]}')
        ax6.hlines(y=0, xmin=0, xmax=EPOCH_NUM, linestyles='--', colors='r', label=f'ln_r={alpha_theta3}')
        ax6.title.set_text('derror_dtheta3')
        plt.legend()

        ax7 = plt.subplot(427)
        ax7.plot(MLoss_list)
        ax7.title.set_text('Martingale Loss')

        ax8 = plt.subplot(428)
        ax8.plot(check_error_RMSE_list, label='RMSE')
        ax8.plot(check_error_MAE_list, label='MAE')
        ax8.title.set_text('FD Error (Value Function)')
        plt.legend()

        fig.suptitle(
            f"Lam{Lambda_x}, N={N}, dim={dim}, {scheme}, {behavior_coef} policy, {theta_init} init_theta, {PATH_NUM} paths, {RunningTime} sec")
        # plt.savefig(f"PE_Exec_GD_Offline_FD_Lam{Lambda_x}_{scheme}.pdf")
        plt.show()
        ###

    def test_PG_GD_Offline(self, Lambda_x):
        T = env_param['T']  # year
        S0 = env_param['S0']
        q0 = env_param['q0']

        kappa_true = env_param['kappa_true']
        eta_true = env_param['eta_true']
        sigma_true = env_param['sigma_true']


        Lambda = round(np.power(10, Lambda_x) * eta_true, 8)
        K_true = np.sqrt(Lambda * sigma_true ** 2 * S0 ** 2 / eta_true)

        N = 5000 if Lambda_x != 4 else 10000
        Delta_t = T / N  # year

        if Lambda_x == 0:
            phi_init = 31.9
        elif Lambda_x == 3:
            phi_init = 2
        elif Lambda_x == 4:
            phi_init = 0.5
        phi1 = K_true * phi_init
        phi1_list = [phi1]

        zeta = 5
        # zeta = 0
        phi2 = 1 / (2 * eta_true) * 2
        phi2_list = [phi2]

        t = [i * Delta_t for i in range(N)]

        Srange_pct = 0.2
        Smax, Smin = S0 * (1 + Srange_pct), S0 * (1 - Srange_pct)
        dim = 50
        h = (Smax - Smin) / dim
        Svec = np.linspace(start=Smin, stop=Smax, num=dim, endpoint=False)  # including Smin

        start = time.perf_counter()

        '''Learned by PE'''
        Q0 = 5e5
        if Lambda_x == 0:
            theta1 = 2.585502269681398e-07 * Q0
            theta2 = 0.0901465309048738
            theta3 = 1.0198212109604483e-08 * Q0
        elif Lambda_x == 3:
            # theta1 = 2.4990916883131735e-07 * Q0
            # theta2 = 0.09104381226281572
            # theta3 = 9.883170962697038e-09 * Q0
            theta1 = kappa_true
            theta2 = sigma_true**2
            theta3 = eta_true
        elif Lambda_x == 4:
            theta1 = 2.5439369682498553e-07 * Q0
            theta2 = 0.09074150809067658
            theta3 = 9.957696856625211e-09 * Q0

        if Lambda_x == 0:
            # alpha_phi1 = 5e4
            # alpha_phi1 = 8e4
            # alpha_phi1 = 3e5
            # alpha_phi1 = 4e5
            # alpha_phi1 = 3.5e5
            # alpha_phi1 = 3.8e5
            alpha_phi1 = 3.9e5
            # alpha_phi1 = 3.2e5
            alpha_phi2 = 3e4
        elif Lambda_x == 3:
            # alpha_phi1 = 1e4
            alpha_phi1 = 2e4
            # alpha_phi1 = 1.8e4
            # alpha_phi1 = 1.5e4
            # alpha_phi1 = 1.2e4
            # alpha_phi1 = 1.1e4
            # alpha_phi1 = 1.05e4
            # alpha_phi1 = 8e3
            # alpha_phi1 = 9e3
            # alpha_phi2 = 3e3
            alpha_phi2 = 3e4
        elif Lambda_x == 4:
            # alpha_phi1 = 2e4
            # alpha_phi1 = 3e4
            # alpha_phi1 = 2.95e4
            # alpha_phi1 = 2.8e4
            # alpha_phi1 = 2.6e4
            # alpha_phi1 = 2.3e4
            # alpha_phi1 = 2.2e4
            alpha_phi1 = 1.6e4
            alpha_phi2 = 3e4

        scheme = 'CN'
        # EPOCH = 100
        EPOCH = 300
        Delta_S0 = S0 - Smin
        S0_idx = Delta_S0 // h
        S0_idx += (Delta_S0 % h >= h / 2) * 1
        S0_idx = int(round(S0_idx))

        EMQVFD_agent = EMQVFD(
            q0=q0,
            S0=S0,
            T=T,
            zeta=zeta,
            Lambda=Lambda,
            phi1=phi1_list[-1],
            phi2=phi2_list[-1],
            theta1=theta1,
            theta2=theta2,
            theta3=theta3,
            alpha_phi1=alpha_phi1,
            alpha_phi2=alpha_phi2,
            alpha_theta1=0,
            alpha_theta2=0,
            alpha_theta3=0,
            N=N,
            delta=Delta_t,
            dim=dim,
            h=h,
            Svec=Svec,
            Smin=Smin,
            Smax=Smax,
            scheme=scheme
        )

        start = time.perf_counter()
        dG_dphi1_FD_list, dG_dphi1_formula_list = [], []
        dG_dphi2_FD_list, dG_dphi2_formula_list = [], []
        for tring_epoch in range(EPOCH):
            print(f"{tring_epoch}, Lam{Lambda_x}, "
                  f"phi1_curr={phi1_list[-1]}, K={K_true}, "
                  f"phi2_curr={phi2_list[-1]}, phi2^*={1/2/eta_true}")
            dG_dphi1_FD, dG_dphi2_FD = EMQVFD_agent.PG_1step()
            phi1_list.append(EMQVFD_agent.phi1)
            phi2_list.append(EMQVFD_agent.phi2)
            dG_dphi1_FD_list.append(dG_dphi1_FD)
            dG_dphi2_FD_list.append(dG_dphi2_FD)

            dG_dphi1_formula, dG_dphi2_formula = cal_dG_dphi_formula(
                phi1_behavior=phi1_list[-1],
                phi2_behavior=phi2_list[-1],
                q0=q0,
                S0=S0,
                eta_true=eta_true,
                sigma_true=sigma_true,
                Lambda=Lambda,
                T=T,
                zeta=zeta
            )
            print(f"dG_dphi1_FD={dG_dphi1_FD}, dG_dphi1_formula={dG_dphi1_formula}")
            print(f"dG_dphi2_FD={dG_dphi2_FD}, dG_dphi2_formula={dG_dphi2_formula}")
            dG_dphi1_formula_list.append(dG_dphi1_formula)
            dG_dphi2_formula_list.append(dG_dphi2_formula)

        run_time = time.perf_counter() - start

        fig = plt.figure(figsize=(8, 6), dpi=300)
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1)
        phi1_learned = np.mean(phi1_list[-int(0.1 * EPOCH):])
        ax1.plot(phi1_list, label=f"phi1_learned={phi1_learned}")
        ax1.hlines(y=K_true, xmin=0, xmax=EPOCH, colors='r', ls='--')
        ax1.title.set_text(f'ln={alpha_phi1}')
        plt.legend()

        ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1)
        ax2.plot(dG_dphi1_FD_list, label=f"dG_dphi1_FD")
        ax2.plot(dG_dphi1_formula_list, label=f"dG_dphi1_formula")
        plt.legend()

        ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
        phi2_learned = np.mean(phi2_list[-int(0.1 * EPOCH):])
        ax3.plot(phi2_list, label=f"phi2_learned={phi2_learned}")
        ax3.hlines(y=1/(2*eta_true), xmin=0, xmax=EPOCH, colors='r', ls='--')
        ax3.title.set_text(f'ln={alpha_phi2}')
        plt.legend()

        ax4 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
        ax4.plot(dG_dphi2_FD_list, label=f"dG_dphi2_FD")
        ax4.plot(dG_dphi2_formula_list, label=f"dG_dphi2_formula")
        plt.legend()

        fig.suptitle(f"Lam{Lambda_x}, N={N}, dim={dim}, scheme={scheme}, {run_time:.3f}s")
        plt.savefig(f"PG_Exec_Offline_FD_PEdone_theta*_Lam{Lambda_x}.pdf")
        plt.show()

if __name__ == '__main__':
    # pytest.main()
    tt = TestActorCritic_Offline()
    tt.test_PG_GD_Offline(Lambda_x=3)
    # tt.test_PE_GD_Offline(Lambda_x=3)
    # tt.test_V_dVdtheta_dVdphi()
    # tt.test_gen_A_B_dtheta()
    # tt.test_r_drdtheta_drdphi()