import math
import os
from argparse import ArgumentParser

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from .Data_preparation import read_runs, encode_and_normalize
from .ablation import do_weighted_ablation_mo
from .fanova import do_weighted_fanova_mo
from .weighting import get_weightings, is_pareto_efficient


def plot(df_fanova, df_ablated, objectives_normed, out_path, df, hps, default):
    """
    Plots the Pareto front, the weighted importance for fANOVA and the ablation path analysis per hyperparameter.
    :param df_fanova: fANOVA dataframe to plot
    :param df_ablated: ablation dataframe to plot
    :param objectives_normed: the names of the objectives
    :param out_path: output path
    :param df: encoded configuration data
    :param hps: list of hyperparameters in the configuration space
    """
    os.makedirs(out_path, exist_ok=True)

    # create color mapping
    hps = hps + ['Default', 'initial_lr']
    hps = sorted(hps)
    if not any('energy' in obj for obj in objectives_normed):
        hps.remove('learning_rate_init')
    colors = {label: color for label, color in zip(hps, sns.color_palette('colorblind', n_colors=len(hps)))}

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # Pareto front
    df['pareto'] = is_pareto_efficient(df[objectives_normed].to_numpy())
    print(df[df['pareto'] == True])
    to_plot = df[df['pareto']].sort_values(by=objectives_normed[0])
    ax1 = axes[0]
    ax1.step(to_plot[objectives_normed[0].split('_normed')[0]], to_plot[objectives_normed[1].split('_normed')[0]],
             marker='o', linestyle='-', markersize=5, where='post')
    ax1.plot(default[0], default[1], 'o', markersize=5, color='red', label='default')
    ax1.set_xlabel('Error')
    ax1.set_ylabel(objectives_normed[1].split('_normed')[0])
    ax1.set_title('Pareto Front')

    # fanova
    df_fanova['hp_name'] = df_fanova['hp_name'].str.replace('learning_rate_init', 'initial_lr')
    weight = [col for col in df_fanova.columns if col.startswith('weight_')][0]
    df_fanova.to_csv(out_path + '/df_fanova.csv', index=False)
    df_fanova = df_fanova.reindex(sorted(df_fanova.columns), axis=1)

    ax2 = axes[1]
    for group_id, group_data in df_fanova.groupby('hp_name'):
        group_data.sort_values(by=weight).plot(x=weight, y='fanova', label=group_id, ax=ax2,
                                               color=colors[group_id])
        to_plot = group_data.sort_values(by=weight)
        ax2.fill_between(to_plot[weight], to_plot['fanova'] - to_plot['variance'],
                         to_plot['fanova'] + to_plot['variance'], alpha=0.2, color=colors[group_id])
    ax2.set_xlabel('Weight for Error')
    ax2.set_ylabel('Importance')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, df_fanova['fanova'].max())
    ax2.set_title('MO-fANOVA')

    #no pareto x axis
    ax2.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0],
                   [objectives_normed[1].split('_normed')[0], '0.2', '0.4', '0.6', '0.8', 'Error'])

    # pareto x axis
    # x_ticks = sorted(list(df_fanova[weight].apply(lambda x: round(x*100)/100).unique()))
    # ax2.set_xticks(x_ticks,
    #                [objectives_normed[1].split('_normed')[0]] + x_ticks[1:-1] + ['Error'], rotation=90)
    ax2.get_legend().remove()

    # ablation
    df_ablated['hp_name'] = df_ablated['hp_name'].str.replace('learning_rate_init', 'initial_lr')
    weight = [col for col in df_ablated.columns if col.startswith('weight_')][0]
    df_ablated.to_csv(out_path + '/df_ablation.csv', index=False)
    df_ablated = df_ablated.reindex(sorted(df_ablated.columns), axis=1)

    ax3 = axes[2]
    df_ablated['accuracy'] = np.where(df_ablated['hp_name'] == 'Default', 1 - df_ablated['new_performance'],
                                      df_ablated['ablation'])
    grouped_df = df_ablated.groupby([weight, 'hp_name'])
    grouped_df = grouped_df['accuracy'].sum().unstack(fill_value=0)
    grouped_df.plot(kind='area', stacked=True, color=[v for k, v in colors.items() if k in grouped_df.columns], ax=ax3)
    ax3.set_xlabel('Weight for Error')
    ax3.set_ylabel('Sum of Weighted\nNormalized Objectives')
    ax3.set_ylim(
        math.floor(10 * (1 - (df_ablated[df_ablated['hp_name'] == 'Default']['new_performance'].max() + 0.01))) / 10, 1)
    ax3.set_title('MO-Ablation Path Analysis')

    #no pareto x axis
    ax3.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0],
                   [objectives_normed[1].split('_normed')[0], '0.2', '0.4', '0.6', '0.8', 'Error'])

    # pareto x axis
    # x_ticks = sorted(list(df_ablated[weight].apply(lambda x: round(x * 100) / 100).unique()))
    # ax3.set_xticks(x_ticks,
    #                [objectives_normed[1].split('_normed')[0]] + x_ticks[1:-1] + ['Error'], rotation=90)

    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.01, 1.0), ncol=1)
    plt.tight_layout()
    os.makedirs(out_path, exist_ok=True)
    fig.savefig(os.path.join(out_path, 'combined_plot.png'), bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p_in', '--in_path', default='', help='path to smac run')
    parser.add_argument('-p_out', '--out_path', default='', help='path to save figures')
    parser.add_argument('-o', '--objectives', nargs='+', default=['1-accuracy', 'time'], help='objectives to consider')
    args = parser.parse_args()

    # Data preparation
    group = read_runs(args.in_path)
    objectives = [o for o in group.get_objectives() if o.name in args.objectives] if len(
        args.objectives) > 0 else group.get_objectives()
    objectives_normed = [o.name + '_normed' for o in objectives]
    df = encode_and_normalize(group, objectives)
    # df = df.head(1000) #--> for energy

    # Calculate and plot HPI
    weightings = get_weightings(objectives_normed, df)
    df_ablated, default = do_weighted_ablation_mo(group, df, objectives_normed, weightings)
    df_fanova = do_weighted_fanova_mo(group, df, objectives_normed, weightings)

    plot(df_fanova, df_ablated, objectives_normed, args.out_path, df,
         group.configspace.get_hyperparameter_names(), default)

    print('Done')
