import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib
matplotlib.rcParams.update({
    # 'text.usetex': False,
    'text.usetex': True,
    'font.family': 'Times New Roman'
})


S0 = 100
q0 = 1
Q0 = 5e5
kappa_true = 2.5e-7 * Q0
eta_true = 1e-8 * Q0
sigma_true = 0.3

T = 1 / 250
N = 5000
Delta_t = T / N

zeta = 5

Lambda_x = 3
Lambda = round(np.power(10, Lambda_x) * eta_true, 8)
K_true = np.sqrt(Lambda * sigma_true ** 2 * S0 ** 2 / eta_true)

color_Wall_iris_blue = (65 / 235, 138 / 235, 180 / 235)
color_Star_blue = (154 / 235, 180 / 235, 205 / 235)
color_Capri_blue = (26 / 235, 85 / 235, 153 / 235)
color_Prussian_blue = (0 / 235, 49 / 235, 83 / 235)
color_red = (194 / 235, 81 / 235, 96 / 235)
color_qing = (108 / 235, 168 / 235, 175 / 235)

legend_font_size = 18
label_font_size = 20


def plot_singleTrain(file_name_EMQV):

    Online_or_Offline = 'Online' if 'Online' in file_name_EMQV else 'Offline'

    training_res = pd.read_csv(file_name_EMQV + '.csv', index_col=0)
    save_name_prefix = f"ActorCritic_{Online_or_Offline}_plot"

    phi1_np = training_res['phi1'].values
    phi1_list = phi1_np.tolist()
    '''Plot phi1'''
    plt.figure(dpi=300)
    plt.plot(phi1_list, color=color_Wall_iris_blue, label=r'$\phi_1$')
    plt.hlines(y=K_true, xmin=0, xmax=len(phi1_list), linestyles='--', colors=color_red, label=r'$\phi_1^*$')

    # plt.title(f'Learning path of ' + r'$\phi_1$')
    plt.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
    if Lambda_x == 4:
        plt.legend(loc='lower right', fontsize=legend_font_size)
    else:
        plt.legend(fontsize=legend_font_size)

    plt.xlabel(f'iteration', usetex=False, fontsize=label_font_size)
    plt.ylabel(r'$\phi_1$', fontsize=label_font_size)
    # plt.title(r'Learning path of $\phi_1$')
    plt.savefig(f'{save_name_prefix}_phi1.pdf', dpi=300)
    # plt.show()
    ''''''


    theta1_list = (training_res['theta1'].values / Q0).tolist()
    '''Plot theta1'''
    plt.figure(dpi=300)
    plt.plot(theta1_list, color=color_Wall_iris_blue, label=r'$\theta_1$')
    plt.hlines(y=kappa_true / Q0, xmin=0, xmax=len(theta1_list), linestyles='--', colors=color_red,
               label=r'$\theta_1^*$')

    # plt.title(f'Learning path of ' + r'$\theta_1$')
    plt.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
    plt.legend(fontsize=legend_font_size)

    plt.xlabel(f'iteration', usetex=False, fontsize=label_font_size)
    plt.ylabel(r'$\theta_1$', fontsize=label_font_size)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))  # convert to scientific notation
    # plt.title(r'Learning path of $\theta_1$')
    plt.savefig(f'{save_name_prefix}_theta1.pdf', dpi=300)
    # plt.show()
    ''''''

    theta2_list = (training_res['theta2'].values).tolist()
    '''Plot theta2'''
    plt.figure(dpi=300)
    plt.plot(theta2_list, color=color_Wall_iris_blue, label=r'$\theta_2$')
    plt.hlines(y=sigma_true**2, xmin=0, xmax=len(theta2_list), linestyles='--', colors=color_red,
               label=r'$\theta_2^*$')

    # plt.title(f'Learning path of ' + r'$\theta_2$')
    plt.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
    plt.legend(fontsize=legend_font_size)

    plt.xlabel(f'iteration', usetex=False, fontsize=label_font_size)
    plt.ylabel(r'$\theta_2$', fontsize=label_font_size)
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # plt.title(r'Learning path of $\theta_2$')
    plt.savefig(f'{save_name_prefix}_theta2.pdf', dpi=300)
    # plt.show()
    ''''''

    theta3_list = (training_res['theta3'].values / Q0).tolist()
    '''Plot theta3'''
    plt.figure(dpi=300)
    plt.plot(theta3_list, color=color_Wall_iris_blue, label=r'$\theta_3$')
    plt.hlines(y=eta_true / Q0, xmin=0, xmax=len(theta3_list), linestyles='--', colors=color_red,
               label=r'$\theta_3^*$')

    # plt.title(f'Learning path of ' + r'$\theta_3$')
    plt.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
    plt.legend(fontsize=legend_font_size)

    plt.xlabel(f'iteration', usetex=False, fontsize=label_font_size)
    plt.ylabel(r'$\theta_3$', fontsize=label_font_size)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # plt.title(r'Learning path of $\theta_3$')
    plt.savefig(f'{save_name_prefix}_theta3.pdf', dpi=300)
    # plt.show()
    ''''''


def plot_multiTrain(file_name_EMQV):

    def plot(name_latex, name_str, training_res, opt_val, plot_band=False):
        mean = training_res.mean(axis=0)  # each row is a learning path
        std = training_res.std(axis=0)  # each row is a learning path

        latex_name = f"${name_latex}$"
        plt.figure(dpi=300)
        # if name_str in ['phi1', 'theta2']:
        if name_str in ['phi1']:
            # plt.plot(mean.tolist(), color=color_Wall_iris_blue, label=latex_name + f", Std={std[-1]:.3f}")
            plt.plot(mean.tolist(), color=color_Wall_iris_blue, label=latex_name + f", Std={std[-1]:.2f}")
        else:
            # plt.plot(mean.tolist(), color=color_Wall_iris_blue, label=latex_name + f", Std={std[-1]:.3E}")
            plt.plot(mean.tolist(), color=color_Wall_iris_blue, label=latex_name + f", Std={std[-1]:.2E}")
        if plot_band:
            plt.fill_between(
                x=list(range(len(mean))),
                y1=mean + std,
                y2=mean - std,
                color='orange',
                alpha=0.3
            )

        latex_name_opt = f"${name_latex}^*$"
        plt.hlines(y=opt_val, xmin=0, xmax=len(mean), linestyles='--', colors=color_red, label=latex_name_opt)


        plt.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
        plt.legend(fontsize=legend_font_size)

        plt.xlabel(f'iteration', usetex=False, fontsize=label_font_size)
        plt.ylabel(latex_name, fontsize=label_font_size)
        plt.savefig(f'{file_name_EMQV}_{name_str}.pdf', dpi=300)

    '''Plot phi1'''
    print(f"Ploting phi1 ......")
    training_res = pd.read_csv(f"{file_name_EMQV}_phi1.csv", index_col=0).values
    plot(name_latex='\phi_1', name_str='phi1', training_res=training_res, opt_val=K_true,
         plot_band=True)
    ''''''

    '''Plot phi2'''
    print(f"Ploting phi2 ......")
    training_res = pd.read_csv(f"{file_name_EMQV}_phi2.csv", index_col=0).values
    plot(name_latex='\phi_2', name_str='phi2', training_res=training_res*Q0, opt_val=1/(2*eta_true)*Q0,
         plot_band=True)
    ''''''

    # '''Plot theta1'''
    print(f"Ploting theta1 ......")
    training_res = pd.read_csv(f"{file_name_EMQV}_theta1.csv", index_col=0).values
    plot(name_latex='\\theta_1', name_str='theta1', training_res=training_res / Q0, opt_val=kappa_true / Q0,
         plot_band=True)
    # ''''''

    '''Plot theta2'''
    print(f"Ploting theta2 ......")
    training_res = pd.read_csv(f"{file_name_EMQV}_theta2.csv", index_col=0).values
    plot(name_latex='\\theta_2', name_str='theta2', training_res=training_res, opt_val=sigma_true**2,
         plot_band=True)
    ''''''

    '''Plot theta3'''
    print(f"Ploting theta3 ......")
    training_res = pd.read_csv(f"{file_name_EMQV}_theta3.csv", index_col=0).values
    plot(name_latex='\\theta_3', name_str='theta3', training_res=training_res / Q0, opt_val=eta_true / Q0,
         plot_band=True)
    ''''''



if __name__ == '__main__':
    # plot_multiTrain(file_name_EMQV='phi1_1232.2839238807894_ActorCritic_Offline_learnPath')
    # plot_multiTrain(file_name_EMQV='phi1_947.1540201818153_ActorCritic_Offline_learnPath')
    plot_multiTrain(file_name_EMQV='ActorCritic_Online_phi1_937.4085710920881_learnPath')
    # plot_singleTrain(file_name_EMQV='ActorCritic_Online_phi1_950.13_learnPath')



