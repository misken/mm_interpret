from pathlib import Path
import pickle

import pandas as pd
import matplotlib.pyplot as plt


def create_cv_plots(results_dict, figures_path):

    for key in results_dict.keys():
        scenario = results_dict[key]['scenario']
        scatter_plot = results_dict[key]['fitplot']
        plot_name = f"{scenario}_cv_scatter.png"
        scatter_plot.savefig(Path(figures_path, plot_name))

def create_coeff_plots(results_dict, figures_path):
    for key in results_dict.keys():
        scenario = results_dict[key]['scenario']
        if 'coefplot' in results_dict[key].keys():
            scatter_plot = results_dict[key]['coefplot']
            plot_name = f"{scenario}_cv_coeff.png"
            scatter_plot.savefig(Path(figures_path, plot_name))

def create_metrics_df(results_dict, output_path):
    dfs = []
    for key in results_dict.keys():
        scenario = results_dict[key]['scenario']

        metrics_df = results_dict[key]['metrics_df']
        metrics_df['scenario'] = scenario
        metrics_df['flavor'] = results_dict[key]['flavor']
        metrics_df['unit'] = results_dict[key]['unit']
        dfs.append(metrics_df)

    consolidated_metrics_df = pd.concat(dfs)
    consolidated_metrics_df.reset_index(inplace=True)
    consolidated_metrics_df.rename(columns={'index': 'fold'}, inplace=True)
    return consolidated_metrics_df


#units_to_process = ['pp', 'ldr', 'obs']
units_to_process = ['pp']

for unit in units_to_process:
    with open(Path("output", f"{unit}_results.pkl"), 'rb') as pickle_file:
        pickeled_results = pickle.load(pickle_file)

#create_cv_plots(pickeled_results, Path("output", "figures"))
#create_coeff_plots(pickeled_results, Path("output", "figures"))

metrics_df = create_metrics_df(pickeled_results, Path("output"))
metrics_df.to_csv(Path("output", f"{unit}_metrics_df.csv"))