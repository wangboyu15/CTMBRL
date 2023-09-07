import time
import os
path = os.getcwd()
if 'Execution_MultiAsset' not in path:
    path += '/Execution_MultiAsset'
    os.chdir(path)
print(f"{os.getcwd()}")

# from Execution_MultiAsset.config import get_config
# from Execution_MultiAsset.bsde_solver import learnValEnv
# from Execution_MultiAsset.environment import Environment

from config import get_config
from bsde_solver import learnValEnv
from environment import Environment

# from Execution_MultiAsset.analytical_sol import multiExec_analytical, multiExec_MonteCarlo
import logging
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

use_cuda = torch.cuda.is_available()
print(f'GPU available: {use_cuda}')
device = torch.device("cuda" if use_cuda else "cpu")


# def train(config, random_seed: int):
def train(random_seed: int):
    config = get_config(
        'MultiAssetExec',
        d_asset=3,
        total_time=1 / 250,
        # num_time_interval=100,
        num_time_interval=1000,
        # num_iterations=4000,
        # num_iterations=500,
        num_iterations=200,
        # num_iterations=150,
        # num_iterations=100,
        # num_iterations=50,
        num_hidden_layers=3,
        num_neuron=32,
        batch_size=32,
        # batch_size=64,
        # batch_size=128,
        valid_size=256,
        logging_frequency=100,
        verbose=True
    )

    torch.manual_seed(random_seed)

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)-6s %(message)s'
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

    '''Use true env param to do some init testing'''
    kappa_true = [1.5e-7, 2.5e-7, 3.5e-7, 1.2e-7]
    eta_true = [0.5e-8, 1e-8, 1.5e-8]
    sigma_lower_true = [1 * 30, 0.6 * 40, 0.8 * 40, 0.325 * 50, -0.18125 * 50, 0.92818 * 50]

    # env_val_learner = learnValEnv(
    #     config=config,
    #     S0=S0,
    #     Q0=Q0,
    #     kappa=kappa_true,
    #     eta=eta_true,
    #     sigma_lower=sigma_lower_true,
    #     d_asset=config.d_asset,
    #     batch_size=config.batch_size,
    #     total_time=config.total_time,
    #     num_time_interval=config.num_time_interval
    # )
    ''''''

    '''Also train env model param'''
    kappa_error = 0.000794320000000015
    kappa_init = [k * (1 + kappa_error) for k in kappa_true]
    kappa_init[0] = kappa_true[0] * (1 - kappa_error)


    eta_error = -0.024374399999999907
    eta_init = [e * (1 + eta_error) for e in eta_true]

    sigma_error = -0.006646477777777826
    sigma_lower_init = [s * (1 + np.sign(s) * sigma_error) for s in sigma_lower_true]  # true=[30, 24.0, 32.0, 16.25, -9.0625, 46.409]  # true

    env_val_learner = learnValEnv(
        config=config,
        S0=S0,
        Q0=Q0,
        kappa=kappa_init,  # init kappa
        eta=eta_init,  # init eta
        sigma_lower=sigma_lower_init,  # init sigma_lower
        d_asset=config.d_asset,
        batch_size=config.batch_size,
        total_time=config.total_time,
        num_time_interval=config.num_time_interval
    )
    ''''''

    print(f"BSDE trainable param: {env_val_learner.sde._parameters}")

    '''For fixed theta=theta*'''
    # optimizer = optim.SGD(env_val_learner.parameters(), 1e-6)
    ''''''

    '''For learning env model with b = 2.0'''
    '''NEAR initial points'''

    optimizer = optim.Adam([
        {'params': env_val_learner.subnetwork.parameters()},
        {'params': env_val_learner.sde.kappa1_to_learn, 'lr': 4.3e-5},  # batch 32
        # {'params': env_val_learner.sde.kappa1_to_learn, 'lr': 0.0},
        {'params': env_val_learner.sde.kappa2_to_learn, 'lr': 4.5e-5},  # batch 32
        # {'params': env_val_learner.sde.kappa2_to_learn, 'lr': 0.0},
        {'params': env_val_learner.sde.kappa3_to_learn, 'lr': 1.55e-4},  # batch 32
        # {'params': env_val_learner.sde.kappa3_to_learn, 'lr': 0.0},
        {'params': env_val_learner.sde.kappa4_to_learn, 'lr': 1.1e-4},  # batch 32
        # {'params': env_val_learner.sde.kappa4_to_learn, 'lr': 0.0},
        {'params': env_val_learner.sde.eta1_to_learn, 'lr': 6.6e-4},  # batch 32
        # {'params': env_val_learner.sde.eta1_to_learn, 'lr': 0.0},
        {'params': env_val_learner.sde.eta2_to_learn, 'lr': 1.25e-3},  # batch 32
        # {'params': env_val_learner.sde.eta2_to_learn, 'lr': 0.0},
        {'params': env_val_learner.sde.eta3_to_learn, 'lr': 1.85e-3},  # batch 32
        # {'params': env_val_learner.sde.eta3_to_learn, 'lr': 0.0},
        {'params': env_val_learner.sde.sigma_lower1_to_learn, 'lr': 1.1e-3},  # batch 32
        # {'params': env_val_learner.sde.sigma_lower1_to_learn, 'lr': 0.0},
        {'params': env_val_learner.sde.sigma_lower2_to_learn, 'lr': 8e-4},  # batch 32
        # {'params': env_val_learner.sde.sigma_lower2_to_learn, 'lr': 0.0},
        {'params': env_val_learner.sde.sigma_lower3_to_learn, 'lr': 1.1e-3},  # batch 32
        # {'params': env_val_learner.sde.sigma_lower3_to_learn, 'lr': 0.0},
        {'params': env_val_learner.sde.sigma_lower4_to_learn, 'lr': 5.5e-4},  # batch 32
        # {'params': env_val_learner.sde.sigma_lower4_to_learn, 'lr': 0.0},
        {'params': env_val_learner.sde.sigma_lower5_to_learn, 'lr': 3e-4},  # batch 32
        # {'params': env_val_learner.sde.sigma_lower5_to_learn, 'lr': 0.0},
        {'params': env_val_learner.sde.sigma_lower6_to_learn, 'lr': 1.6e-3}  # batch 32
        # {'params': env_val_learner.sde.sigma_lower6_to_learn, 'lr': 0.0}
    ],
        lr=0.01
    )

    ''''''

    y_init_list = []
    ML_loss_list = []
    kappa_list = [kappa_init]
    eta_list = [eta_init]
    sigma_lower_list = [sigma_lower_init]

    # # for validation
    # r_valid, S_valid, q_valid = env.sample(num_sample=config.valid_size)
    # '''
    # r_valid: (valid_size, num_time_interval + 1), last columns is terminal payoff
    # S_valid: (valid_size, d_asset, num_time_interval + 1)
    # q_valid: (d_asset, num_time_interval + 1)
    # '''

    # begin sgd iteration
    for step in range(config.num_iterations + 1):
        print(f"========================================================")
        print(f"Training: seed={random_seed}, iter={step + 1}/{config.num_iterations}")

        r_train, S_train, q_train = env.sample(num_sample=config.batch_size)
        '''
        r_train: (batch_size, num_time_interval + 1), last columns is terminal payoff
        S_train: (batch_size, d_asset, num_time_interval + 1)
        q_train: (d_asset, num_time_interval + 1)
        '''

        # print(
        #     f"-- lr_theta: {[optimizer_theta.param_groups[i]['lr'] for i in range(len(optimizer_theta.param_groups))]}")
        # print(f"-- lr_NN: {optimizer_NN.param_groups[0]['lr']}")
        # print(f"-- lr_theta: {optimizer_theta.param_groups[0]['lr']}")


        optimizer.zero_grad()

        env_val_learner.train()
        loss, y_init = env_val_learner(
            q_env=q_train,
            S_env=S_train,
            r_env=r_train
        )  # loss: tensor([]), y_init: float
        # loss.backward()
        loss.backward(retain_graph=True)

        optimizer.step()

        # print('-- grad: kappa_to_learn.grad=', env_val_learner.sde.kappa_to_learn.grad)
        print('-- grad: kappa1_to_learn.grad=', env_val_learner.sde.kappa1_to_learn.grad)
        print('-- grad: kappa2_to_learn.grad=', env_val_learner.sde.kappa2_to_learn.grad)
        print('-- grad: kappa3_to_learn.grad=', env_val_learner.sde.kappa3_to_learn.grad)
        print('-- grad: kappa4_to_learn.grad=', env_val_learner.sde.kappa4_to_learn.grad)

        # print('-- grad: eta_to_learn.grad=', env_val_learner.sde.eta_to_learn.grad)
        print('-- grad: eta1_to_learn.grad=', env_val_learner.sde.eta1_to_learn.grad.item())
        print('-- grad: eta2_to_learn.grad=', env_val_learner.sde.eta2_to_learn.grad.item())
        print('-- grad: eta3_to_learn.grad=', env_val_learner.sde.eta3_to_learn.grad.item())


        # print('-- grad: sigma_lower_to_learn.grad=', env_val_learner.sde.sigma_lower_to_learn.grad)
        print('-- grad: sigma_lower1_to_learn.grad=', env_val_learner.sde.sigma_lower1_to_learn.grad.item())
        print('-- grad: sigma_lower2_to_learn.grad=', env_val_learner.sde.sigma_lower2_to_learn.grad.item())
        print('-- grad: sigma_lower3_to_learn.grad=', env_val_learner.sde.sigma_lower3_to_learn.grad.item())
        print('-- grad: sigma_lower4_to_learn.grad=', env_val_learner.sde.sigma_lower4_to_learn.grad.item())
        print('-- grad: sigma_lower5_to_learn.grad=', env_val_learner.sde.sigma_lower5_to_learn.grad.item())
        print('-- grad: sigma_lower6_to_learn.grad=', env_val_learner.sde.sigma_lower6_to_learn.grad.item())

        y_init_list.append(y_init)
        ML_loss_list.append(loss.item())
        # kappa_list.append(env_val_learner.sde.kappa_to_learn.detach().numpy().tolist())
        kappa_list.append([
            env_val_learner.sde.kappa1_to_learn.detach().item(),
            env_val_learner.sde.kappa2_to_learn.detach().item(),
            env_val_learner.sde.kappa3_to_learn.detach().item(),
            env_val_learner.sde.kappa4_to_learn.detach().item()
        ])

        # eta_list.append(env_val_learner.sde.eta_to_learn.detach().numpy().tolist())
        eta_list.append([
            env_val_learner.sde.eta1_to_learn.detach().item(),
            env_val_learner.sde.eta2_to_learn.detach().item(),
            env_val_learner.sde.eta3_to_learn.detach().item(),
        ])

        # sigma_lower_list.append(env_val_learner.sde.sigma_lower_to_learn.detach().numpy().tolist())
        sigma_lower_list.append([
            env_val_learner.sde.sigma_lower1_to_learn.detach().item(),
            env_val_learner.sde.sigma_lower2_to_learn.detach().item(),
            env_val_learner.sde.sigma_lower3_to_learn.detach().item(),
            env_val_learner.sde.sigma_lower4_to_learn.detach().item(),
            env_val_learner.sde.sigma_lower5_to_learn.detach().item(),
            env_val_learner.sde.sigma_lower6_to_learn.detach().item()
        ])

        print(f"*** ML={ML_loss_list[-1]}, J0={y_init_list[-1]}")
        for name, param in env_val_learner.sde.named_parameters():
            if 'kappa_' in name:
                print(f"::: kappa_true={[k * 1e6 for k in kappa_true]}, init={[k * 1e6 for k in kappa_list[0]]}")
            elif 'kappa1' in name:
                print(f"::: kappa1_true={kappa_true[0] * 1e6}, init={kappa_list[0][0] * 1e6}")
            elif 'kappa2' in name:
                print(f"::: kappa2_true={kappa_true[1] * 1e6}, init={kappa_list[0][1] * 1e6}")
            elif 'kappa3' in name:
                print(f"::: kappa3_true={kappa_true[2] * 1e6}, init={kappa_list[0][2] * 1e6}")
            elif 'kappa4' in name:
                print(f"::: kappa4_true={kappa_true[3] * 1e6}, init={kappa_list[0][3] * 1e6}")

            elif 'eta_' in name:
                print(f"::: eta_true={[k * 1e7 for k in eta_true]}, init={[k * 1e7 for k in eta_list[0]]}")
            elif 'eta1' in name:
                print(f"::: eta1_true={eta_true[0] * 1e7}, init={eta_list[0][0] * 1e7}")
            elif 'eta2' in name:
                print(f"::: eta2_true={eta_true[1] * 1e7}, init={eta_list[0][1] * 1e7}")
            elif 'eta3' in name:
                print(f"::: eta3_true={eta_true[2] * 1e7}, init={eta_list[0][2] * 1e7}")

            elif 'sigma_lower_' in name:
                print(f"::: sigma_lower_true={[k * 1e2 for k in sigma_lower_true]}, init={[k * 1e2 for k in sigma_lower_list[0]]}")
            elif 'sigma_lower1_' in name:
                print(f"::: sigma1_lower_true={sigma_lower_true[0] / 1e2}, init={sigma_lower_list[0][0] / 1e2}")
            elif 'sigma_lower2_' in name:
                print(f"::: sigma2_lower_true={sigma_lower_true[1] / 1e2}, init={sigma_lower_list[0][1] / 1e2}")
            elif 'sigma_lower3_' in name:
                print(f"::: sigma3_lower_true={sigma_lower_true[2] / 1e2}, init={sigma_lower_list[0][2] / 1e2}")
            elif 'sigma_lower4_' in name:
                print(f"::: sigma4_lower_true={sigma_lower_true[3] / 1e2}, init={sigma_lower_list[0][3] / 1e2}")
            elif 'sigma_lower5_' in name:
                print(f"::: sigma5_lower_true={sigma_lower_true[4] / 1e2}, init={sigma_lower_list[0][4] / 1e2}")
            elif 'sigma_lower6_' in name:
                print(f"::: sigma6_lower_true={sigma_lower_true[5] / 1e2}, init={sigma_lower_list[0][5] / 1e2}")

            print(f"::: {name}, {param.detach().numpy()}")

        print(f"========================================================")

    '''Final validation'''
    find_valid_size = 10000
    print(f"final valid: {find_valid_size} episodes...")
    r_valid, S_valid, q_valid = env.sample(num_sample=find_valid_size)
    env_val_learner.eval()
    loss, y_init = env_val_learner(
        q_env=q_valid,
        S_env=S_valid,
        r_env=r_valid
    )


    y_init_list.append(y_init)
    ML_loss_list.append(loss.item())

    save_df = pd.concat(
        [
            pd.DataFrame(y_init_list),
            pd.DataFrame(ML_loss_list),
            pd.DataFrame(kappa_list),
            pd.DataFrame(eta_list),
            pd.DataFrame(sigma_lower_list),
        ], axis=1
    )
    save_df.columns = ['J0', 'ML'] + \
                      [f'kappa_{i + 1}' for i in range(len(kappa_list[0]))] + \
                      [f'eta_{i + 1}' for i in range(len(eta_list[0]))] + \
                      [f'sigma_lower_{i + 1}' for i in range(len(sigma_lower_list[0]))]

    print(f"Saving training results...")
    save_name = f"b{int(env.b)}_learn_env_param_seed{random_seed}"
    save_df.to_csv(f"{save_name}.csv")
    torch.save(env_val_learner.state_dict(), f'{save_name}.pt')  ## in Mac

    # return y_init_list, ML_loss_list, kappa_list, eta_list, sigma_lower_list, env_val_learner, env, save_name
    '''In multiprocessing, return env_val_learner raises an error, not sure the reason why'''
    return y_init_list, ML_loss_list, kappa_list, eta_list, sigma_lower_list, env, save_name


if __name__ == '__main__':
    '''Multi process'''
    # num_repeat = 10
    num_repeat = 20
    # num_repeat = 100
    seed_list = [i for i in range(1, num_repeat+1)]

    RunningStartTime = time.perf_counter()
    # with ProcessPoolExecutor(max_workers=10) as pool:
    with ProcessPoolExecutor() as pool:
        training_res = pool.map(train, seed_list)





    '''Single process'''
    # cfg = get_config(
    #     'MultiAssetExec',
    #     d_asset=3,
    #     total_time=1/250,
    #     # num_time_interval=100,
    #     num_time_interval=1000,
    #     # num_iterations=4000,
    #     # num_iterations=500,
    #     # num_iterations=200,
    #     num_iterations=100,
    #     num_hidden_layers=3,
    #     num_neuron=32,
    #     batch_size=64,
    #     valid_size=256,
    #     logging_frequency=100,
    #     verbose=True
    # )


    '''Run training !'''
    # y_init_list, ML_loss_list, kappa_list, eta_list, sigma_lower_list, env_val_learner, env, save_name = train(
    #     config=cfg,
    #     # random_seed=1
    #     # random_seed=2
    #     # random_seed=3
    #     random_seed=4
    # )

    # y_init_list, ML_loss_list, kappa_list, eta_list, sigma_lower_list, env, save_name = train(
    #     # random_seed=1
    #     # random_seed=2
    #     # random_seed=3
    #     random_seed=4
    # )
    # ''''''

    # if env.b == 1.0:
    #     multiExec_formula = multiExec_analytical(
    #         T=env.total_time,
    #         N=env.num_time_interval,
    #         d_asset=env.d_asset,
    #         S0=env.S0,
    #         Q0=env.q0[:, 0],
    #         kappa_true_mat=env.kappa_true_mat,
    #         alpha_mat=env.alpha_mat,
    #         eta_true_mat=env.eta_true_mat,
    #         eta_true_inv=env.eta_true_inv,
    #         Sigma_true_mat=env.Sigma_true_mat,
    #         Lambda=env.Lambda,
    #         zeta=0.0,
    #         b=env.b
    #     )
    #     J0 = multiExec_formula.cal_val()
    #     print(f"J0_analytical={J0}")
    # else:
    #     num_sample = 10000
    #     reward, S_train, qt_tensor = env.sample(num_sample=num_sample)
    #     J0 = torch.mean(torch.sum(reward, dim=-1, keepdim=False)).detach().item()
    #     print(f"J0_MC = {J0}")
    #
    #
    # log10_relative_error = [np.log10(abs(y - J0) / J0 + 1e-20) for y in y_init_list]
    # learned_relaive_error = abs(y_init_list[-1] - J0) / J0
    # plot_x = list(range(len(y_init_list)))
    # plt.figure(figsize=(12, 4), dpi=300)
    # plt.subplot(131)
    # plt.plot(plot_x, y_init_list, label=f"learned={round(y_init_list[-1], 3)}", linewidth=0.8)
    # plt.hlines(y=J0, xmin=0, xmax=len(y_init_list), colors='r', ls='--', label=f"optimal={round(J0, 3)}")
    # plt.ylabel(f"value function")
    # plt.legend()
    #
    # plt.subplot(132)
    # plt.plot(plot_x, log10_relative_error, label=r"$|y-y^*|/y^*$" + f"={round(learned_relaive_error * 100, 3)}\%", linewidth=0.8)
    # plt.ylabel(f"log10(relative error)")
    # plt.legend()
    #
    # plt.subplot(133)
    # plot_x = list(range(len(ML_loss_list)))
    # plt.plot(plot_x, [np.log10(l) for l in ML_loss_list], linewidth=0.8)
    # plt.ylabel(f"log10(MLoss)")
    # plt.savefig(f'{save_name}.pdf')
    # plt.show()
