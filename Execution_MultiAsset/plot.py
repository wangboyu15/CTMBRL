import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'text.usetex': True,
    'font.family': 'Times New Roman'
})
import pandas as pd
import numpy as np
from Execution_MultiAsset.config import get_config
from Execution_MultiAsset.environment import Environment
from Execution_MultiAsset.analytical_sol import multiExec_analytical





def plot_singleTrain():
    print(f"plot_singleTrain")

    config = get_config(
        'MultiAssetExec',
        d_asset=3,
        total_time=1 / 250,
        num_time_interval=1000,
        num_iterations=500,
        num_hidden_layers=3,
        num_neuron=20,
        batch_size=64,
        valid_size=256,
        logging_frequency=100,
        # logging_frequency=10,
        verbose=True
    )

    S0 = [100.] * config.d_asset
    Q0 = [5e5] * config.d_asset

    # b = 1.0
    b = 2.0

    env = Environment(
        S0=S0,
        Q0=Q0,
        d_asset=config.d_asset,
        batch_size=config.batch_size,
        total_time=config.total_time,
        num_time_interval=config.num_time_interval,
        b=b
    )

    if env.b == 1.0:
        multiExec_formula = multiExec_analytical(
            T=config.total_time,
            N=config.num_time_interval,
            d_asset=config.d_asset,
            S0=env.S0,
            Q0=env.q0[:, 0],
            kappa_true_mat=env.kappa_true_mat,
            alpha_mat=env.alpha_mat,
            eta_true_mat=env.eta_true_mat,
            eta_true_inv=env.eta_true_inv,
            Sigma_true_mat=env.Sigma_true_mat,
            Lambda=env.Lambda,
            # zeta=5.0
            zeta=0.0
        )
        J0 = multiExec_formula.cal_val()
        print(f"J0_analytical={J0}")
    else:
        num_sample = 50000
        reward, S_train, qt_tensor = env.sample(num_sample=num_sample)
        import torch
        torch.manual_seed(seed=1234)
        J0 = torch.mean(torch.sum(reward, dim=-1, keepdim=False)).detach().item()
        print(f"J0_MC = {J0}")


    # read_name = f"true_env_param_learn_paths"
    # read_name = f"b{int(b)}_learn_env_param_learn_paths"
    read_name = f"Online_b2_learn_env_param_seed4"
    df = pd.read_csv(f'{read_name}.csv', index_col=0)

    '''Plot value and ML'''
    J0_DL = df['J0'].values.tolist()
    # MLoss = df['ML']
    TDloss = df['TDloss']
    log10_relative_error = [np.log10(abs(y - J0) / J0 + 1e-20) for y in J0_DL]
    learned_relaive_error = abs(J0_DL[-1] - J0) / J0
    plot_x = list(range(len(J0_DL)))
    plt.figure(figsize=(12, 4), dpi=300)
    plt.subplot(131)
    # plt.subplot(121)
    plt.plot(plot_x, J0_DL, label=f"learned={round(J0_DL[-1], 3)}", linewidth=0.8)
    plt.hlines(y=J0, xmin=0, xmax=len(J0_DL), colors='r', ls='--',
               label=f"true={round(J0, 3)}")
    plt.ylabel(f"value function")
    plt.legend()

    plt.subplot(132)
    # plt.subplot(122)
    plt.plot(plot_x, log10_relative_error, label=r"$|y-y^*|/y^*$" + f"={round(learned_relaive_error * 100, 3)}\%",
             linewidth=0.8)
    plt.ylabel(f"log10(relative error)")
    plt.legend()

    plt.subplot(133)
    # plot_x = list(range(len(MLoss)))
    # plt.plot(plot_x, [np.log10(l) for l in MLoss], linewidth=0.8)
    # plt.ylabel(f"log10(MLoss)")

    plot_x = list(range(len(TDloss)))
    plt.plot(plot_x, [np.log10(l) for l in TDloss], linewidth=0.8)
    plt.ylabel(f"log10(TDloss)")

    plt.savefig(f'{read_name}.pdf')
    plt.close()
    ''''''





    '''Plot kappa'''
    print(f"Ploting kappa ......")
    kappa_true = [1.5e-7, 2.5e-7, 3.5e-7, 1.2e-7]

    normalizing_const = 1e-6
    # kappa_1 = (df['kappa_1'] * normalizing_const).values.tolist()[1:]
    # kappa_2 = (df['kappa_2'] * normalizing_const).values.tolist()[1:]
    # kappa_3 = (df['kappa_3'] * normalizing_const).values.tolist()[1:]
    # kappa_4 = (df['kappa_4'] * normalizing_const).values.tolist()[1:]

    kappa_1 = df['kappa_1']
    kappa_1[1:] *= normalizing_const
    kappa_1 = kappa_1.values.tolist()

    kappa_2 = df['kappa_2']
    kappa_2[1:] *= normalizing_const
    kappa_2 = kappa_2.values.tolist()

    kappa_3 = df['kappa_3']
    kappa_3[1:] *= normalizing_const
    kappa_3 = kappa_3.values.tolist()

    kappa_4 = df['kappa_4']
    kappa_4[1:] *= normalizing_const
    kappa_4 = kappa_4.values.tolist()



    plt.figure(figsize=(9, 6), dpi=300)
    plt.subplot(221)
    plot_x = list(range(len(kappa_1)))
    plt.plot(plot_x, kappa_1, label=f"learned={np.mean(kappa_1[-10:]): .8E}", linewidth=0.8)
    plt.hlines(y=kappa_true[0], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={kappa_true[0]}")
    plt.ylabel(r"$\kappa_{11}$")
    plt.legend()

    plt.subplot(222)
    plot_x = list(range(len(kappa_2)))
    plt.plot(plot_x, kappa_2, label=f"learned={np.mean(kappa_2[-10:]): .8E}", linewidth=0.8)
    plt.hlines(y=kappa_true[1], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={kappa_true[1]}")
    plt.ylabel(r"$\kappa_{22}$")
    plt.legend()

    plt.subplot(223)
    plot_x = list(range(len(kappa_3)))
    plt.plot(plot_x, kappa_3, label=f"learned={np.mean(kappa_3[-10:]): .8E}", linewidth=0.8)
    plt.hlines(y=kappa_true[2], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={kappa_true[2]}")
    plt.ylabel(r"$\kappa_{33}$")
    plt.legend()

    plt.subplot(224)
    plot_x = list(range(len(kappa_4)))
    plt.plot(plot_x, kappa_4, label=f"learned={np.mean(kappa_4[-10:]): .8E}", linewidth=0.8)
    plt.hlines(y=kappa_true[3], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={kappa_true[3]}")
    plt.ylabel(r"$\kappa_{12}$")
    plt.legend()
    plt.savefig(f"{read_name}_kappa.pdf")
    # plt.show()
    plt.close()
    ''''''

    '''Plot eta'''
    print(f"Ploting eta ......")
    eta_true = [0.5e-8, 1e-8, 1.5e-8]
    normalizing_const = 1e-7
    # eta_1 = (df['eta_1'] * normalizing_const).values.tolist()[1:]
    # eta_2 = (df['eta_2'] * normalizing_const).values.tolist()[1:]
    # eta_3 = (df['eta_3'] * normalizing_const).values.tolist()[1:]

    eta_1 = df['eta_1']
    eta_1[1:] *= normalizing_const
    eta_1 = eta_1.values.tolist()

    eta_2 = df['eta_2']
    eta_2[1:] *= normalizing_const
    eta_2 = eta_2.values.tolist()

    eta_3 = df['eta_3']
    eta_3[1:] *= normalizing_const
    eta_3 = eta_3.values.tolist()


    plt.figure(figsize=(12, 4), dpi=300)
    plt.subplot(131)
    plot_x = list(range(len(eta_1)))
    plt.plot(plot_x, eta_1, label=f"learned={np.mean(eta_1[-10:]): .8E}", linewidth=0.8)
    plt.hlines(y=eta_true[0], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={eta_true[0]}")
    plt.ylabel(r"$\eta_{1}$")
    plt.legend()

    plt.subplot(132)
    plot_x = list(range(len(eta_2)))
    plt.plot(plot_x, eta_2, label=f"learned={np.mean(eta_2[-10:]): .8E}", linewidth=0.8)
    plt.hlines(y=eta_true[1], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={eta_true[1]}")
    plt.ylabel(r"$\eta_{2}$")
    plt.legend()

    plt.subplot(133)
    plot_x = list(range(len(eta_3)))
    plt.plot(plot_x, eta_3, label=f"learned={np.mean(eta_3[-10:]): .8E}", linewidth=0.8)
    plt.hlines(y=eta_true[2], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={eta_true[2]}")
    plt.ylabel(r"$\eta_{3}$")
    plt.legend()

    plt.savefig(f"{read_name}_eta.pdf")
    # plt.show()
    plt.close()
    ''''''

    '''Plot sigma_low'''
    print(f"Ploting sigma_low ......")
    sigma_lower_true = [1 * 30, 0.6 * 40, 0.8 * 40, 0.325 * 50, -0.18125 * 50, 0.92818 * 50]
    normalizing_const = 100
    # sigma_low_1 = (df['sigma_lower_1'] * normalizing_const).values.tolist()[1:]
    # sigma_low_2 = (df['sigma_lower_2'] * normalizing_const).values.tolist()[1:]
    # sigma_low_3 = (df['sigma_lower_3'] * normalizing_const).values.tolist()[1:]
    # sigma_low_4 = (df['sigma_lower_4'] * normalizing_const).values.tolist()[1:]
    # sigma_low_5 = (df['sigma_lower_5'] * normalizing_const).values.tolist()[1:]
    # sigma_low_6 = (df['sigma_lower_6'] * normalizing_const).values.tolist()[1:]

    sigma_low_1 = df['sigma_lower_1']
    sigma_low_1[1:] *= normalizing_const
    sigma_low_1 = sigma_low_1.values.tolist()

    sigma_low_2 = df['sigma_lower_2']
    sigma_low_2[1:] *= normalizing_const
    sigma_low_2 = sigma_low_2.values.tolist()

    sigma_low_3 = df['sigma_lower_3']
    sigma_low_3[1:] *= normalizing_const
    sigma_low_3 = sigma_low_3.values.tolist()

    sigma_low_4 = df['sigma_lower_4']
    sigma_low_4[1:] *= normalizing_const
    sigma_low_4 = sigma_low_4.values.tolist()

    sigma_low_5 = df['sigma_lower_5']
    sigma_low_5[1:] *= normalizing_const
    sigma_low_5 = sigma_low_5.values.tolist()

    sigma_low_6 = df['sigma_lower_6']
    sigma_low_6[1:] *= normalizing_const
    sigma_low_6 = sigma_low_6.values.tolist()



    plt.figure(figsize=(12, 8), dpi=300)
    plt.subplot(231)
    plot_x = list(range(len(sigma_low_1)))
    plt.plot(plot_x, sigma_low_1, label=f"learned={np.mean(sigma_low_1[-10:]): .8f}", linewidth=0.8)
    plt.hlines(y=sigma_lower_true[0], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={sigma_lower_true[0]}")
    plt.ylabel(r"$\sigma_{11}$")
    plt.legend()

    plt.subplot(232)
    plot_x = list(range(len(sigma_low_2)))
    plt.plot(plot_x, sigma_low_2, label=f"learned={np.mean(sigma_low_2[-10:]): .8f}", linewidth=0.8)
    plt.hlines(y=sigma_lower_true[1], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={sigma_lower_true[1]}")
    plt.ylabel(r"$\sigma_{21}$")
    plt.legend()

    plt.subplot(233)
    plot_x = list(range(len(sigma_low_3)))
    plt.plot(plot_x, sigma_low_3, label=f"learned={np.mean(sigma_low_3[-10:]): .8f}", linewidth=0.8)
    plt.hlines(y=sigma_lower_true[2], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={sigma_lower_true[2]}")
    plt.ylabel(r"$\sigma_{22}$")
    plt.legend()


    plt.subplot(234)
    plot_x = list(range(len(sigma_low_4)))
    plt.plot(plot_x, sigma_low_4, label=f"learned={np.mean(sigma_low_4[-10:]): .8f}", linewidth=0.8)
    plt.hlines(y=sigma_lower_true[3], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={sigma_lower_true[3]}")
    plt.ylabel(r"$\sigma_{31}$")
    plt.legend()

    plt.subplot(235)
    plot_x = list(range(len(sigma_low_5)))
    plt.plot(plot_x, sigma_low_5, label=f"learned={np.mean(sigma_low_5[-10:]): .8f}", linewidth=0.8)
    plt.hlines(y=sigma_lower_true[4], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={sigma_lower_true[4]}")
    plt.ylabel(r"$\sigma_{32}$")
    plt.legend()

    plt.subplot(236)
    plot_x = list(range(len(sigma_low_6)))
    plt.plot(plot_x, sigma_low_6, label=f"learned={np.mean(sigma_low_6[-10:]): .8f}", linewidth=0.8)
    plt.hlines(y=sigma_lower_true[5], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={sigma_lower_true[5]}")
    plt.ylabel(r"$\sigma_{33}$")
    plt.legend()

    plt.savefig(f"{read_name}_sigma_low.pdf")
    # plt.show()
    plt.close()
    ''''''



def plot_multiTrain():
    print(f"plot_multiTrain")

    config = get_config(
        'MultiAssetExec',
        d_asset=3,
        total_time=1 / 250,
        num_time_interval=1000,
        num_iterations=500,
        num_hidden_layers=3,
        num_neuron=20,
        batch_size=64,
        valid_size=256,
        logging_frequency=100,
        # logging_frequency=10,
        verbose=True
    )

    S0 = [100.] * config.d_asset
    Q0 = [5e5] * config.d_asset

    # b = 1.0
    b = 2.0

    env = Environment(
        S0=S0,
        Q0=Q0,
        d_asset=config.d_asset,
        batch_size=config.batch_size,
        total_time=config.total_time,
        num_time_interval=config.num_time_interval,
        b=b
    )

    if env.b == 1.0:
        multiExec_formula = multiExec_analytical(
            T=config.total_time,
            N=config.num_time_interval,
            d_asset=config.d_asset,
            S0=env.S0,
            Q0=env.q0[:, 0],
            kappa_true_mat=env.kappa_true_mat,
            alpha_mat=env.alpha_mat,
            eta_true_mat=env.eta_true_mat,
            eta_true_inv=env.eta_true_inv,
            Sigma_true_mat=env.Sigma_true_mat,
            Lambda=env.Lambda,
            # zeta=5.0
            zeta=0.0
        )
        J0 = multiExec_formula.cal_val()
        print(f"J0_analytical={J0}")
    else:
        num_sample = 50000
        reward, S_train, qt_tensor = env.sample(num_sample=num_sample)
        import torch
        torch.manual_seed(seed=1234)
        J0 = torch.mean(torch.sum(reward, dim=-1, keepdim=False)).detach().item()
        print(f"J0_MC = {J0}")

    # num_repeat = 5
    # num_repeat = 8
    num_repeat = 10
    # num_repeat = 20
    seed_list = [i for i in range(1, num_repeat + 1)]

    J0_DL_all, MLoss_all = [], []
    kappa_1_all, kappa_2_all, kappa_3_all, kappa_4_all = [], [], [], []
    eta_1_all, eta_2_all, eta_3_all = [], [], []
    sigma_low_1_all, sigma_low_2_all, sigma_low_3_all, sigma_low_4_all, sigma_low_5_all, sigma_low_6_all = \
        [], [], [], [], [], []

    for random_seed in seed_list:
        read_name = f"b{int(b)}_learn_env_param_seed{random_seed}"
        print(f"reading {read_name}")

        df = pd.read_csv(f'{read_name}.csv', index_col=0)

        J0_DL_all.append(df['J0'].values.tolist())
        MLoss_all.append(df['ML'].values.tolist())

        '''record kappa'''
        normalizing_const = 1e-6
        kappa_1 = df['kappa_1']
        kappa_1[1:] *= normalizing_const
        kappa_1_all.append(kappa_1.values.tolist())

        kappa_2 = df['kappa_2']
        kappa_2[1:] *= normalizing_const
        kappa_2_all.append(kappa_2.values.tolist())

        kappa_3 = df['kappa_3']
        kappa_3[1:] *= normalizing_const
        kappa_3_all.append(kappa_3.values.tolist())

        kappa_4 = df['kappa_4']
        kappa_4[1:] *= normalizing_const
        kappa_4_all.append(kappa_4.values.tolist())
        ''''''


        '''record eta'''
        normalizing_const = 1e-7

        eta_1 = df['eta_1']
        eta_1[1:] *= normalizing_const
        eta_1_all.append(eta_1.values.tolist())

        eta_2 = df['eta_2']
        eta_2[1:] *= normalizing_const
        eta_2_all.append(eta_2.values.tolist())

        eta_3 = df['eta_3']
        eta_3[1:] *= normalizing_const
        eta_3_all.append(eta_3.values.tolist())
        ''''''


        '''record sigma_low'''
        normalizing_const = 100

        sigma_low_1 = df['sigma_lower_1']
        sigma_low_1[1:] *= normalizing_const
        sigma_low_1_all.append(sigma_low_1.values.tolist())

        sigma_low_2 = df['sigma_lower_2']
        sigma_low_2[1:] *= normalizing_const
        sigma_low_2_all.append(sigma_low_2.values.tolist())

        sigma_low_3 = df['sigma_lower_3']
        sigma_low_3[1:] *= normalizing_const
        sigma_low_3_all.append(sigma_low_3.values.tolist())

        sigma_low_4 = df['sigma_lower_4']
        sigma_low_4[1:] *= normalizing_const
        sigma_low_4_all.append(sigma_low_4.values.tolist())

        sigma_low_5 = df['sigma_lower_5']
        sigma_low_5[1:] *= normalizing_const
        sigma_low_5_all.append(sigma_low_5.values.tolist())

        sigma_low_6 = df['sigma_lower_6']
        sigma_low_6[1:] *= normalizing_const
        sigma_low_6_all.append(sigma_low_6.values.tolist())
        ''''''


    '''shape=(len(seed_list), N+1)'''
    J0_DL_all = np.array(J0_DL_all)
    MLoss_all = np.array(MLoss_all)
    kappa_1_all = np.array(kappa_1_all)
    kappa_2_all = np.array(kappa_2_all)
    kappa_3_all = np.array(kappa_3_all)
    kappa_4_all = np.array(kappa_4_all)
    eta_1_all = np.array(eta_1_all)
    eta_2_all = np.array(eta_2_all)
    eta_3_all = np.array(eta_3_all)
    sigma_low_1_all = np.array(sigma_low_1_all)
    sigma_low_2_all = np.array(sigma_low_2_all)
    sigma_low_3_all = np.array(sigma_low_3_all)
    sigma_low_4_all = np.array(sigma_low_4_all)
    sigma_low_5_all = np.array(sigma_low_5_all)
    sigma_low_6_all = np.array(sigma_low_6_all)
    num_time_steps = J0_DL_all.shape[1]
    read_name = f"b{int(b)}_learn_env_param"


    '''Plot value and ML'''
    print(f"Plot value and ML")
    J0_DL_mean = J0_DL_all.mean(axis=0)
    J0_DL_std = J0_DL_all.std(axis=0)

    plot_x = list(range(num_time_steps))
    plt.figure(figsize=(12, 4), dpi=300)
    plt.subplot(131)
    plt.plot(plot_x, J0_DL_mean.tolist(), label=f"learned={round(J0_DL_mean[-1], 3)}", linewidth=0.8)
    plt.fill_between(
        x=plot_x,
        y1=(J0_DL_mean + J0_DL_std).tolist(),
        y2=(J0_DL_mean - J0_DL_std).tolist(),
        facecolor='orange',
        alpha=0.3
    )
    plt.hlines(y=J0, xmin=0, xmax=len(J0_DL_mean), colors='r', ls='--',
               label=f"true={round(J0, 3)}")
    plt.ylabel(f"value function")
    plt.legend()


    log10_relative_error = np.log10(abs(J0_DL_all - J0) / J0 + 1e-20)
    log10_relative_error_mean = log10_relative_error.mean(axis=0)
    log10_relative_error_std = log10_relative_error.std(axis=0)

    learned_relaive_error = abs(np.mean(J0_DL_all[:, -1]) - J0) / J0
    plt.subplot(132)
    plt.plot(plot_x, log10_relative_error_mean.tolist(), label=r"$|y-y^*|/y^*$" + f"={round(learned_relaive_error * 100, 3)}\%",
             linewidth=0.8)
    plt.fill_between(
        x=plot_x,
        y1=(log10_relative_error_mean + log10_relative_error_std).tolist(),
        y2=(log10_relative_error_mean - log10_relative_error_std).tolist(),
        facecolor='orange',
        alpha=0.3
    )
    plt.ylabel(f"log10(relative error)")
    plt.legend()

    plt.subplot(133)
    MLoss_log10 = np.log10(MLoss_all)
    MLoss_mean = MLoss_log10.mean(axis=0)
    MLoss_std = MLoss_log10.std(axis=0)
    plot_x = list(range(MLoss_all.shape[1]))
    plt.plot(plot_x, MLoss_mean.tolist(), linewidth=0.8)
    plt.fill_between(
        x=plot_x,
        y1=(MLoss_mean + MLoss_std).tolist(),
        y2=(MLoss_mean - MLoss_std).tolist(),
        facecolor='orange',
        alpha=0.3
    )
    plt.ylabel(f"log10(MLoss)")
    plt.savefig(f'{read_name}.pdf')
    plt.show()
    ''''''





    '''Plot kappa'''
    print(f"Ploting kappa ......")
    kappa_true = [1.5e-7, 2.5e-7, 3.5e-7, 1.2e-7]
    # kappa_true = [2.0e-7, 2.5e-7, 3.0e-7, 1.6e-7]



    plt.figure(figsize=(9, 6), dpi=300)
    plt.subplot(221)
    plot_x = list(range(kappa_1_all.shape[1]))

    kappa_1_mean = kappa_1_all.mean(axis=0)
    kappa_1_std = kappa_1_all.std(axis=0)

    plt.plot(plot_x, kappa_1_mean.tolist(), label=f"learned={kappa_1_mean[-1]: .6E}, Std={kappa_1_std[-1]: .2E}", linewidth=0.8)
    # plt.fill_between(
    #     x=plot_x,
    #     y1=(kappa_1_mean + kappa_1_std).tolist(),
    #     y2=(kappa_1_mean - kappa_1_std).tolist(),
    #     facecolor='orange',
    #     alpha=0.3
    # )
    plt.hlines(y=kappa_true[0], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={kappa_true[0]}")
    plt.ylabel(r"$\kappa_{1}$")
    plt.legend()

    plt.subplot(222)
    kappa_2_mean = kappa_2_all.mean(axis=0)
    kappa_2_std = kappa_2_all.std(axis=0)
    plt.plot(plot_x, kappa_2_mean.tolist(), label=f"learned={kappa_2_mean[-1]: .6E}, Std={kappa_2_std[-1]: .2E}", linewidth=0.8)
    # plt.fill_between(
    #     x=plot_x,
    #     y1=(kappa_2_mean + kappa_2_std).tolist(),
    #     y2=(kappa_2_mean - kappa_2_std).tolist(),
    #     facecolor='orange',
    #     alpha=0.3
    # )
    plt.hlines(y=kappa_true[1], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={kappa_true[1]}")
    plt.ylabel(r"$\kappa_{2}$")
    plt.legend()

    plt.subplot(223)
    kappa_3_mean = kappa_3_all.mean(axis=0)
    kappa_3_std = kappa_3_all.std(axis=0)
    plt.plot(plot_x, kappa_3_mean.tolist(), label=f"learned={kappa_3_mean[-1]: .6E}, Std={kappa_3_std[-1]: .2E}", linewidth=0.8)
    plt.hlines(y=kappa_true[2], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={kappa_true[2]}")
    # plt.fill_between(
    #     x=plot_x,
    #     y1=(kappa_3_mean + kappa_3_std).tolist(),
    #     y2=(kappa_3_mean - kappa_3_std).tolist(),
    #     facecolor='orange',
    #     alpha=0.3
    # )
    plt.ylabel(r"$\kappa_{3}$")
    plt.legend()

    plt.subplot(224)
    kappa_4_mean = kappa_4_all.mean(axis=0)
    kappa_4_std = kappa_4_all.std(axis=0)
    plt.plot(plot_x, kappa_4_mean.tolist(), label=f"learned={kappa_4_mean[-1]: .6E}, Std={kappa_4_std[-1]: .2E}", linewidth=0.8)
    plt.hlines(y=kappa_true[3], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={kappa_true[3]}")
    # plt.fill_between(
    #     x=plot_x,
    #     y1=(kappa_4_mean + kappa_4_std).tolist(),
    #     y2=(kappa_4_mean - kappa_4_std).tolist(),
    #     facecolor='orange',
    #     alpha=0.3
    # )
    plt.ylabel(r"$\kappa_{12}$")
    plt.legend()
    plt.savefig(f"{read_name}_kappa.pdf")
    plt.show()
    plt.close()
    ''''''

    '''Plot eta'''
    print(f"Ploting eta ......")
    eta_true = [0.5e-8, 1e-8, 1.5e-8]


    plt.figure(figsize=(12, 4), dpi=300)
    plt.subplot(131)
    plot_x = list(range(eta_1_all.shape[1]))

    eta_1_mean = eta_1_all.mean(axis=0)
    eta_1_std = eta_1_all.std(axis=0)

    plt.plot(plot_x, eta_1_mean.tolist(), label=f"learned={eta_1_mean[-1]: .6E}, Std={eta_1_std[-1]: .2E}", linewidth=0.8)
    plt.hlines(y=eta_true[0], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={eta_true[0]}")
    # plt.fill_between(
    #     x=plot_x,
    #     y1=(eta_1_mean + eta_1_std).tolist(),
    #     y2=(eta_1_mean - eta_1_std).tolist(),
    #     facecolor='orange',
    #     alpha=0.3
    # )

    plt.ylabel(r"$\eta_{1}$")
    plt.legend()

    plt.subplot(132)
    eta_2_mean = eta_2_all.mean(axis=0)
    eta_2_std = eta_2_all.std(axis=0)
    plt.plot(plot_x, eta_2_mean.tolist(), label=f"learned={eta_2_mean[-1]: .6E}, Std={eta_2_std[-1]: .2E}", linewidth=0.8)
    plt.hlines(y=eta_true[1], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={eta_true[1]}")
    # plt.fill_between(
    #     x=plot_x,
    #     y1=(eta_2_mean + eta_2_std).tolist(),
    #     y2=(eta_2_mean - eta_2_std).tolist(),
    #     facecolor='orange',
    #     alpha=0.3
    # )
    plt.ylabel(r"$\eta_{2}$")
    plt.legend()

    plt.subplot(133)
    eta_3_mean = eta_3_all.mean(axis=0)
    eta_3_std = eta_3_all.std(axis=0)
    plt.plot(plot_x, eta_3_mean.tolist(), label=f"learned={eta_3_mean[-1]: .6E}, Std={eta_3_std[-1]: .2E}", linewidth=0.8)
    plt.hlines(y=eta_true[2], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={eta_true[2]}")
    # plt.fill_between(
    #     x=plot_x,
    #     y1=(eta_3_mean + eta_3_std).tolist(),
    #     y2=(eta_3_mean - eta_3_std).tolist(),
    #     facecolor='orange',
    #     alpha=0.3
    # )
    plt.ylabel(r"$\eta_{3}$")
    plt.legend()

    plt.savefig(f"{read_name}_eta.pdf")
    plt.show()
    plt.close()
    ''''''

    '''Plot sigma_low'''
    print(f"Ploting sigma_low ......")
    sigma_lower_true = [1 * 30, 0.6 * 40, 0.8 * 40, 0.325 * 50, -0.18125 * 50, 0.92818 * 50]


    plt.figure(figsize=(12, 8), dpi=300)
    plt.subplot(231)
    plot_x = list(range(sigma_low_1_all.shape[1]))

    sigma_low_1_mean = sigma_low_1_all.mean(axis=0)
    sigma_low_1_std = sigma_low_1_all.std(axis=0)

    plt.plot(plot_x, sigma_low_1_mean.tolist(), label=f"learned={sigma_low_1_mean[-1]: .6f}, Std={sigma_low_1_std[-1]: .2f}", linewidth=0.8)
    plt.hlines(y=sigma_lower_true[0], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={sigma_lower_true[0]}")
    # plt.fill_between(
    #     x=plot_x,
    #     y1=(sigma_low_1_mean + sigma_low_1_std).tolist(),
    #     y2=(sigma_low_1_mean - sigma_low_1_std).tolist(),
    #     facecolor='orange',
    #     alpha=0.3
    # )
    plt.ylabel(r"$\sigma_{11}$")
    plt.legend()

    plt.subplot(232)
    sigma_low_2_mean = sigma_low_2_all.mean(axis=0)
    sigma_low_2_std = sigma_low_2_all.std(axis=0)

    plt.plot(plot_x, sigma_low_2_mean.tolist(), label=f"learned={sigma_low_2_mean[-1]: .6f}, Std={sigma_low_2_std[-1]: .2f}", linewidth=0.8)
    plt.hlines(y=sigma_lower_true[1], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={sigma_lower_true[1]}")
    # plt.fill_between(
    #     x=plot_x,
    #     y1=(sigma_low_2_mean + sigma_low_2_std).tolist(),
    #     y2=(sigma_low_2_mean - sigma_low_2_std).tolist(),
    #     facecolor='orange',
    #     alpha=0.3
    # )
    plt.ylabel(r"$\sigma_{21}$")
    plt.legend()

    plt.subplot(233)
    sigma_low_3_mean = sigma_low_3_all.mean(axis=0)
    sigma_low_3_std = sigma_low_3_all.std(axis=0)

    plt.plot(plot_x, sigma_low_3_mean.tolist(), label=f"learned={sigma_low_3_mean[-1]: .6f}, Std={sigma_low_3_std[-1]: .2f}", linewidth=0.8)
    plt.hlines(y=sigma_lower_true[2], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={sigma_lower_true[2]}")
    # plt.fill_between(
    #     x=plot_x,
    #     y1=(sigma_low_3_mean + sigma_low_3_std).tolist(),
    #     y2=(sigma_low_3_mean - sigma_low_3_std).tolist(),
    #     facecolor='orange',
    #     alpha=0.3
    # )
    plt.ylabel(r"$\sigma_{22}$")
    plt.legend()


    plt.subplot(234)
    sigma_low_4_mean = sigma_low_4_all.mean(axis=0)
    sigma_low_4_std = sigma_low_4_all.std(axis=0)

    plt.plot(plot_x, sigma_low_4_mean.tolist(), label=f"learned={sigma_low_4_mean[-1]: .6f}, Std={sigma_low_4_std[-1]: .2f}", linewidth=0.8)
    plt.hlines(y=sigma_lower_true[3], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={sigma_lower_true[3]}")
    # plt.fill_between(
    #     x=plot_x,
    #     y1=(sigma_low_4_mean + sigma_low_4_std).tolist(),
    #     y2=(sigma_low_4_mean - sigma_low_4_std).tolist(),
    #     facecolor='orange',
    #     alpha=0.3
    # )
    plt.ylabel(r"$\sigma_{31}$")
    plt.legend()

    plt.subplot(235)
    sigma_low_5_mean = sigma_low_5_all.mean(axis=0)
    sigma_low_5_std = sigma_low_5_all.std(axis=0)
    plt.plot(plot_x, sigma_low_5_mean.tolist(), label=f"learned={sigma_low_5_mean[-1]: .6f}, Std={sigma_low_5_std[-1]: .2f}", linewidth=0.8)
    plt.hlines(y=sigma_lower_true[4], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={sigma_lower_true[4]}")
    # plt.fill_between(
    #     x=plot_x,
    #     y1=(sigma_low_5_mean + sigma_low_5_std).tolist(),
    #     y2=(sigma_low_5_mean - sigma_low_5_std).tolist(),
    #     facecolor='orange',
    #     alpha=0.3
    # )
    plt.ylabel(r"$\sigma_{32}$")
    plt.legend()

    plt.subplot(236)
    sigma_low_6_mean = sigma_low_6_all.mean(axis=0)
    sigma_low_6_std = sigma_low_6_all.std(axis=0)

    plt.plot(plot_x, sigma_low_6_mean.tolist(), label=f"learned={sigma_low_6_mean[-1]: .6f}, Std={sigma_low_6_std[-1]: .2f}", linewidth=0.8)
    plt.hlines(y=sigma_lower_true[5], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={sigma_lower_true[5]}")
    # plt.fill_between(
    #     x=plot_x,
    #     y1=(sigma_low_6_mean + sigma_low_6_std).tolist(),
    #     y2=(sigma_low_6_mean - sigma_low_6_std).tolist(),
    #     facecolor='orange',
    #     alpha=0.3
    # )
    plt.ylabel(r"$\sigma_{33}$")
    plt.legend()

    plt.savefig(f"{read_name}_sigma_low.pdf")
    plt.show()
    plt.close()
    ''''''



def plot_multiTrain_online():
    print(f"plot_multiTrain_online")

    config = get_config(
        'MultiAssetExec',
        d_asset=3,
        total_time=1 / 250,
        num_time_interval=1000,
        num_iterations=500,
        num_hidden_layers=3,
        num_neuron=20,
        batch_size=64,
        valid_size=256,
        logging_frequency=100,
        # logging_frequency=10,
        verbose=True
    )

    S0 = [100.] * config.d_asset
    Q0 = [5e5] * config.d_asset

    # b = 1.0
    b = 2.0

    env = Environment(
        S0=S0,
        Q0=Q0,
        d_asset=config.d_asset,
        batch_size=config.batch_size,
        total_time=config.total_time,
        num_time_interval=config.num_time_interval,
        b=b
    )

    if env.b == 1.0:
        pass
    else:
        num_sample = 50000
        reward, S_train, qt_tensor = env.sample(num_sample=num_sample)
        import torch
        torch.manual_seed(seed=1234)
        J0 = torch.mean(torch.sum(reward, dim=-1, keepdim=False)).detach().item()
        print(f"J0_MC = {J0}")

    # # num_repeat = 5
    # # num_repeat = 8
    # num_repeat = 10
    # # num_repeat = 20
    # seed_list = [i for i in range(1, num_repeat + 1)]

    # J0_DL_all, TDlss_all = [], []
    # kappa_1_all, kappa_2_all, kappa_3_all, kappa_4_all = [], [], [], []
    # eta_1_all, eta_2_all, eta_3_all = [], [], []
    # sigma_low_1_all, sigma_low_2_all, sigma_low_3_all, sigma_low_4_all, sigma_low_5_all, sigma_low_6_all = \
    #     [], [], [], [], [], []

    read_name_prefix = f'Online_b2_270.71263275146487'

    J0_DL_all = pd.read_csv(f"{read_name_prefix}_y_pred.csv", index_col=0)
    TDlss_all = pd.read_csv(f"{read_name_prefix}_TDloss.csv", index_col=0)

    normalizing_const_kappa = 1e-6
    kappa1_all = pd.read_csv(f"{read_name_prefix}_kappa1.csv", index_col=0) * normalizing_const_kappa
    kappa2_all = pd.read_csv(f"{read_name_prefix}_kappa2.csv", index_col=0) * normalizing_const_kappa
    kappa3_all = pd.read_csv(f"{read_name_prefix}_kappa3.csv", index_col=0) * normalizing_const_kappa
    kappa4_all = pd.read_csv(f"{read_name_prefix}_kappa4.csv", index_col=0) * normalizing_const_kappa

    normalizing_const_eta = 1e-7
    eta1_all = pd.read_csv(f"{read_name_prefix}_eta1.csv", index_col=0) * normalizing_const_eta
    eta2_all = pd.read_csv(f"{read_name_prefix}_eta2.csv", index_col=0) * normalizing_const_eta
    eta3_all = pd.read_csv(f"{read_name_prefix}_eta3.csv", index_col=0) * normalizing_const_eta

    normalizing_const_sigma_lower = 100
    sigma_lower1_all = pd.read_csv(f"{read_name_prefix}_sigma_lower1.csv", index_col=0) * normalizing_const_sigma_lower
    sigma_lower2_all = pd.read_csv(f"{read_name_prefix}_sigma_lower2.csv", index_col=0) * normalizing_const_sigma_lower
    sigma_lower3_all = pd.read_csv(f"{read_name_prefix}_sigma_lower3.csv", index_col=0) * normalizing_const_sigma_lower
    sigma_lower4_all = pd.read_csv(f"{read_name_prefix}_sigma_lower4.csv", index_col=0) * normalizing_const_sigma_lower
    sigma_lower5_all = pd.read_csv(f"{read_name_prefix}_sigma_lower5.csv", index_col=0) * normalizing_const_sigma_lower
    sigma_lower6_all = pd.read_csv(f"{read_name_prefix}_sigma_lower6.csv", index_col=0) * normalizing_const_sigma_lower

    num_time_steps = J0_DL_all.shape[1]

    '''Plot value'''
    print(f"Plot value")
    J0_DL_mean = J0_DL_all.mean(axis=0)
    J0_DL_std = J0_DL_all.std(axis=0)

    plot_x = list(range(num_time_steps))
    plt.figure(figsize=(12, 4), dpi=300)
    plt.subplot(121)
    plt.plot(plot_x, J0_DL_mean.tolist(), label=f"learned={J0_DL_mean[-1]: .2f}, Std={J0_DL_std[-1]: .2f}", linewidth=0.8)
    plt.fill_between(
        x=plot_x,
        y1=(J0_DL_mean + J0_DL_std).tolist(),
        y2=(J0_DL_mean - J0_DL_std).tolist(),
        facecolor='orange',
        alpha=0.3
    )
    plt.hlines(y=J0, xmin=0, xmax=len(J0_DL_mean), colors='r', ls='--',
               label=f"true={round(J0, 3)}")
    plt.ylabel(f"value function")
    plt.legend()


    log10_relative_error = np.log10(abs(J0_DL_all - J0) / J0 + 1e-20)
    log10_relative_error_mean = log10_relative_error.mean(axis=0)
    log10_relative_error_std = log10_relative_error.std(axis=0)

    learned_relaive_error = abs(np.mean(J0_DL_all.iloc[:, -1]) - J0) / J0
    plt.subplot(122)
    plt.plot(plot_x, log10_relative_error_mean.tolist(), label=r"$|y-y^*|/y^*$" + f"={learned_relaive_error * 100: .2f}\%",
             linewidth=0.8)
    plt.fill_between(
        x=plot_x,
        y1=(log10_relative_error_mean + log10_relative_error_std).tolist(),
        y2=(log10_relative_error_mean - log10_relative_error_std).tolist(),
        facecolor='orange',
        alpha=0.3
    )
    plt.ylabel(f"log10(relative error)")
    plt.legend()

    plt.savefig(f'{read_name_prefix}_y_pred.pdf')
    plt.close()
    ''''''

    '''Plot TDloss'''
    print(f"Plot TDloss")
    plt.figure(dpi=300)
    TSloss_log10 = np.log10(TDlss_all)
    TDloss_mean = TSloss_log10.mean(axis=0)
    TDloss_std = TSloss_log10.std(axis=0)
    plot_x = list(range(TDlss_all.shape[1]))
    plt.plot(plot_x, TDloss_mean.tolist(), linewidth=0.8)
    plt.fill_between(
        x=plot_x,
        y1=(TDloss_mean + TDloss_std).tolist(),
        y2=(TDloss_mean - TDloss_std).tolist(),
        facecolor='orange',
        alpha=0.3
    )
    plt.ylabel(f"log10(TDloss)")
    plt.savefig(f'{read_name_prefix}_TDloss.pdf')
    plt.close()
    ''''''



    plot_band_model_param = False

    '''Plot kappa'''
    print(f"Ploting kappa ......")
    kappa_true = [1.5e-7, 2.5e-7, 3.5e-7, 1.2e-7]


    plt.figure(figsize=(9, 6), dpi=300)
    plt.subplot(221)
    plot_x = list(range(kappa1_all.shape[1]))

    kappa_1_mean = kappa1_all.mean(axis=0)
    kappa_1_std = kappa1_all.std(axis=0)

    plt.plot(plot_x, kappa_1_mean.tolist(), label=f"learned={kappa_1_mean[-1]: .4E}, Std={kappa_1_std[-1]: .2E}", linewidth=0.8)
    if plot_band_model_param:
        plt.fill_between(
            x=plot_x,
            y1=(kappa_1_mean + kappa_1_std).tolist(),
            y2=(kappa_1_mean - kappa_1_std).tolist(),
            facecolor='orange',
            alpha=0.3
        )
    plt.hlines(y=kappa_true[0], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={kappa_true[0]: .4E}")
    plt.ylabel(r"$\kappa_{1}$")
    plt.legend()

    plt.subplot(222)
    kappa_2_mean = kappa2_all.mean(axis=0)
    kappa_2_std = kappa2_all.std(axis=0)
    plt.plot(plot_x, kappa_2_mean.tolist(), label=f"learned={kappa_2_mean[-1]: .4E}, Std={kappa_2_std[-1]: .2E}", linewidth=0.8)
    if plot_band_model_param:
        plt.fill_between(
            x=plot_x,
            y1=(kappa_2_mean + kappa_2_std).tolist(),
            y2=(kappa_2_mean - kappa_2_std).tolist(),
            facecolor='orange',
            alpha=0.3
        )
    plt.hlines(y=kappa_true[1], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={kappa_true[1]: .4E}")
    plt.ylabel(r"$\kappa_{2}$")
    plt.legend()

    plt.subplot(223)
    kappa_3_mean = kappa3_all.mean(axis=0)
    kappa_3_std = kappa3_all.std(axis=0)
    plt.plot(plot_x, kappa_3_mean.tolist(), label=f"learned={kappa_3_mean[-1]: .4E}, Std={kappa_3_std[-1]: .2E}", linewidth=0.8)
    plt.hlines(y=kappa_true[2], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={kappa_true[2]: .4E}")
    if plot_band_model_param:
        plt.fill_between(
            x=plot_x,
            y1=(kappa_3_mean + kappa_3_std).tolist(),
            y2=(kappa_3_mean - kappa_3_std).tolist(),
            facecolor='orange',
            alpha=0.3
        )
    plt.ylabel(r"$\kappa_{3}$")
    plt.legend()

    plt.subplot(224)
    kappa_4_mean = kappa4_all.mean(axis=0)
    kappa_4_std = kappa4_all.std(axis=0)
    plt.plot(plot_x, kappa_4_mean.tolist(), label=f"learned={kappa_4_mean[-1]: .4E}, Std={kappa_4_std[-1]: .2E}", linewidth=0.8)
    plt.hlines(y=kappa_true[3], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={kappa_true[3]: .4E}")
    if plot_band_model_param:
        plt.fill_between(
            x=plot_x,
            y1=(kappa_4_mean + kappa_4_std).tolist(),
            y2=(kappa_4_mean - kappa_4_std).tolist(),
            facecolor='orange',
            alpha=0.3
        )
    plt.ylabel(r"$\kappa_{12}$")
    plt.legend()
    plt.savefig(f"{read_name_prefix}_kappa.pdf")
    plt.close()
    ''''''

    '''Plot eta'''
    print(f"Ploting eta ......")
    eta_true = [0.5e-8, 1e-8, 1.5e-8]


    plt.figure(figsize=(12, 4), dpi=300)
    plt.subplot(131)
    plot_x = list(range(eta1_all.shape[1]))

    eta_1_mean = eta1_all.mean(axis=0)
    eta_1_std = eta1_all.std(axis=0)

    plt.plot(plot_x, eta_1_mean.tolist(), label=f"learned={eta_1_mean[-1]: .2E}, Std={eta_1_std[-1]: .2E}", linewidth=0.8)
    plt.hlines(y=eta_true[0], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={eta_true[0]: .2E}")
    if plot_band_model_param:
        plt.fill_between(
            x=plot_x,
            y1=(eta_1_mean + eta_1_std).tolist(),
            y2=(eta_1_mean - eta_1_std).tolist(),
            facecolor='orange',
            alpha=0.3
        )

    plt.ylabel(r"$\eta_{1}$")
    plt.legend()

    plt.subplot(132)
    eta_2_mean = eta2_all.mean(axis=0)
    eta_2_std = eta2_all.std(axis=0)
    plt.plot(plot_x, eta_2_mean.tolist(), label=f"learned={eta_2_mean[-1]: .2E}, Std={eta_2_std[-1]: .2E}", linewidth=0.8)
    plt.hlines(y=eta_true[1], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={eta_true[1]: .2E}")

    if plot_band_model_param:
        plt.fill_between(
            x=plot_x,
            y1=(eta_2_mean + eta_2_std).tolist(),
            y2=(eta_2_mean - eta_2_std).tolist(),
            facecolor='orange',
            alpha=0.3
        )
    plt.ylabel(r"$\eta_{2}$")
    plt.legend()

    plt.subplot(133)
    eta_3_mean = eta3_all.mean(axis=0)
    eta_3_std = eta3_all.std(axis=0)
    plt.plot(plot_x, eta_3_mean.tolist(), label=f"learned={eta_3_mean[-1]: .2E}, Std={eta_3_std[-1]: .2E}", linewidth=0.8)
    plt.hlines(y=eta_true[2], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={eta_true[2]: .2E}")

    if plot_band_model_param:
        plt.fill_between(
            x=plot_x,
            y1=(eta_3_mean + eta_3_std).tolist(),
            y2=(eta_3_mean - eta_3_std).tolist(),
            facecolor='orange',
            alpha=0.3
        )
    plt.ylabel(r"$\eta_{3}$")
    plt.legend()

    plt.savefig(f"{read_name_prefix}_eta.pdf")
    # plt.show()
    plt.close()
    ''''''

    '''Plot sigma_low'''
    print(f"Ploting sigma_lower ......")
    sigma_lower_true = [1 * 30, 0.6 * 40, 0.8 * 40, 0.325 * 50, -0.18125 * 50, 0.92818 * 50]


    plt.figure(figsize=(12, 8), dpi=300)
    plt.subplot(231)
    plot_x = list(range(sigma_lower1_all.shape[1]))

    sigma_low_1_mean = sigma_lower1_all.mean(axis=0)
    sigma_low_1_std = sigma_lower1_all.std(axis=0)

    plt.plot(plot_x, sigma_low_1_mean.tolist(), label=f"learned={sigma_low_1_mean[-1]: .2f}, Std={sigma_low_1_std[-1]: .2f}", linewidth=0.8)
    plt.hlines(y=sigma_lower_true[0], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={sigma_lower_true[0]: .2f}")
    if plot_band_model_param:
        plt.fill_between(
            x=plot_x,
            y1=(sigma_low_1_mean + sigma_low_1_std).tolist(),
            y2=(sigma_low_1_mean - sigma_low_1_std).tolist(),
            facecolor='orange',
            alpha=0.3
        )
    plt.ylabel(r"$\sigma_{11}$")
    plt.legend()

    plt.subplot(232)
    sigma_low_2_mean = sigma_lower2_all.mean(axis=0)
    sigma_low_2_std = sigma_lower2_all.std(axis=0)

    plt.plot(plot_x, sigma_low_2_mean.tolist(), label=f"learned={sigma_low_2_mean[-1]: .2f}, Std={sigma_low_2_std[-1]: .2f}", linewidth=0.8)
    plt.hlines(y=sigma_lower_true[1], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={sigma_lower_true[1]: .2f}")
    if plot_band_model_param:
        plt.fill_between(
            x=plot_x,
            y1=(sigma_low_2_mean + sigma_low_2_std).tolist(),
            y2=(sigma_low_2_mean - sigma_low_2_std).tolist(),
            facecolor='orange',
            alpha=0.3
        )
    plt.ylabel(r"$\sigma_{21}$")
    plt.legend()

    plt.subplot(233)
    sigma_low_3_mean = sigma_lower3_all.mean(axis=0)
    sigma_low_3_std = sigma_lower3_all.std(axis=0)

    plt.plot(plot_x, sigma_low_3_mean.tolist(), label=f"learned={sigma_low_3_mean[-1]: .2f}, Std={sigma_low_3_std[-1]: .2f}", linewidth=0.8)
    plt.hlines(y=sigma_lower_true[2], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={sigma_lower_true[2]: .2f}")
    if plot_band_model_param:
        plt.fill_between(
            x=plot_x,
            y1=(sigma_low_3_mean + sigma_low_3_std).tolist(),
            y2=(sigma_low_3_mean - sigma_low_3_std).tolist(),
            facecolor='orange',
            alpha=0.3
        )
    plt.ylabel(r"$\sigma_{22}$")
    plt.legend()


    plt.subplot(234)
    sigma_low_4_mean = sigma_lower4_all.mean(axis=0)
    sigma_low_4_std = sigma_lower4_all.std(axis=0)

    plt.plot(plot_x, sigma_low_4_mean.tolist(), label=f"learned={sigma_low_4_mean[-1]: .2f}, Std={sigma_low_4_std[-1]: .2f}", linewidth=0.8)
    plt.hlines(y=sigma_lower_true[3], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={sigma_lower_true[3]: .2f}")
    if plot_band_model_param:
        plt.fill_between(
            x=plot_x,
            y1=(sigma_low_4_mean + sigma_low_4_std).tolist(),
            y2=(sigma_low_4_mean - sigma_low_4_std).tolist(),
            facecolor='orange',
            alpha=0.3
        )
    plt.ylabel(r"$\sigma_{31}$")
    plt.legend()

    plt.subplot(235)
    sigma_low_5_mean = sigma_lower5_all.mean(axis=0)
    sigma_low_5_std = sigma_lower5_all.std(axis=0)
    plt.plot(plot_x, sigma_low_5_mean.tolist(), label=f"learned={sigma_low_5_mean[-1]: .2f}, Std={sigma_low_5_std[-1]: .2f}", linewidth=0.8)
    plt.hlines(y=sigma_lower_true[4], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={sigma_lower_true[4]: .2f}")
    if plot_band_model_param:
        plt.fill_between(
            x=plot_x,
            y1=(sigma_low_5_mean + sigma_low_5_std).tolist(),
            y2=(sigma_low_5_mean - sigma_low_5_std).tolist(),
            facecolor='orange',
            alpha=0.3
        )
    plt.ylabel(r"$\sigma_{32}$")
    plt.legend()

    plt.subplot(236)
    sigma_low_6_mean = sigma_lower6_all.mean(axis=0)
    sigma_low_6_std = sigma_lower6_all.std(axis=0)

    plt.plot(plot_x, sigma_low_6_mean.tolist(), label=f"learned={sigma_low_6_mean[-1]: .2f}, Std={sigma_low_6_std[-1]: .2f}", linewidth=0.8)
    plt.hlines(y=sigma_lower_true[5], xmin=0, xmax=plot_x[-1], colors='r', ls='--',
               label=f"true={sigma_lower_true[5]: .2f}")
    if plot_band_model_param:
        plt.fill_between(
            x=plot_x,
            y1=(sigma_low_6_mean + sigma_low_6_std).tolist(),
            y2=(sigma_low_6_mean - sigma_low_6_std).tolist(),
            facecolor='orange',
            alpha=0.3
        )
    plt.ylabel(r"$\sigma_{33}$")
    plt.legend()

    plt.savefig(f"{read_name_prefix}_sigma_low.pdf")
    plt.close()
    ''''''


if __name__ == '__main__':
    # plot_singleTrain()
    # plot_multiTrain()
    plot_multiTrain_online()


