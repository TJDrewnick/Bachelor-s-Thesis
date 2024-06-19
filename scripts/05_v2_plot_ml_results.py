import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import joblib as jbl

from utils import settings as s

files_prefix = 'ml_v2/'
title_modifier = ''
if s.ml['random noise']:
    files_prefix = 'ml_v2_random_noise/'
    title_modifier = ' with Random Noise Feature'
elif s.ml['knockout']:
    files_prefix = 'ml_v2_knockout/'
    title_modifier = ' with 1. Feature removed'


def plot_boxplot(country: str) -> None:
    data = pd.read_hdf(f'../results/{files_prefix}predictions/{country}_detrended_predictions.h5')

    fig, ax = plt.subplots(2, 1, figsize=(10, 5.5))

    labels = [
        'Calculated Results',
        'GBT LightGBM',
        'GBT XGBoost\nSquared Error',
        'GBT XGBoost\nAbsolute Error',
        'RF LightGBM',
        'MLP'
    ]

    ax[0].set_title(f'Detrended {country} Drift Predictions', fontsize=s.plotting['subplot title size'])
    ax[1].set_title(f'Detrended {country} Diffusion Predictions', fontsize=s.plotting['subplot title size'])

    ax[0].set_ylabel(r'Drift $D_1(\omega)$', fontsize=s.plotting['fontsize'])
    ax[1].set_ylabel(r'Diffusion $D_2(\omega)$', fontsize=s.plotting['fontsize'])

    ax[0].yaxis.grid(True)
    ax[1].yaxis.grid(True)

    bplot_drift = ax[0].boxplot(
        [data['drift_true'],
         data['drift_gbt_lgb'],
         data['drift_gbt_xgb_squarederror'],
         data['drift_gbt_xgb_absoluteerror'],
         data['drift_rf_lgb'],
         data['drift_mlp']],
        vert=True, patch_artist=True, labels=labels, medianprops=dict(color='black')
    )

    bplot_diffusion = ax[1].boxplot(
        [data['diffusion_true'],
         data['diffusion_gbt_lgb'],
         data['diffusion_gbt_xgb_squarederror'],
         data['diffusion_gbt_xgb_absoluteerror'],
         data['diffusion_rf_lgb'],
         data['diffusion_mlp']],
        vert=True, patch_artist=True, labels=labels, medianprops=dict(color='black')
    )

    # color plot
    targets = ['drift', 'diffusion']
    plots = [bplot_drift, bplot_diffusion]
    ml_colors = [s.plotting['color_gbt_lgb'], s.plotting['color_gbt_xgb_squarederror'],
                 s.plotting['color_gbt_xgb_absoluteerror'], s.plotting['color_rf_lgb'], s.plotting['color_mlp']]
    for target, bplot in zip(targets, plots):
        color_list = ([s.plotting[f'color {country}'][f'detrended {target}']] + ml_colors)
        for patch, color in zip(bplot['boxes'], color_list):
            patch.set_facecolor(color)

    plt.tight_layout()
    plt.savefig(f'../results/{files_prefix}plots/boxplot/{country}_{files_prefix.replace('/', '_')}boxplot.png')
    plt.savefig(f'../results/{files_prefix}plots/boxplot/{country}_{files_prefix.replace('/', '_')}boxplot.pdf')


def all_predictions(country: str) -> None:
    data = pd.read_hdf(f'../results/{files_prefix}predictions/{country}_detrended_all_predictions.h5')

    # default width for plots is 10, increasing this because a higher resolution is needed
    scaling = 1.5

    for target in ['drift', 'diffusion']:
        fig, ax = plt.subplots(6, 1, sharex=True, sharey=True)
        fig.set_figwidth(10 * scaling)
        fig.set_figheight(8 * scaling)

        ax[0].plot(data[f'{target}_true_all'], color=s.plotting[f'color {country}'][f'detrended {target}'])
        ax[1].plot(data[f'{target}_gbt_lgb_all'], color=s.plotting['color_gbt_lgb'])
        ax[2].plot(data[f'{target}_gbt_xgb_squarederror_all'], color=s.plotting['color_gbt_xgb_squarederror'])
        ax[3].plot(data[f'{target}_gbt_xgb_absoluteerror_all'], color=s.plotting['color_gbt_xgb_absoluteerror'])
        ax[4].plot(data[f'{target}_rf_lgb_all'], color=s.plotting['color_rf_lgb'])
        ax[5].plot(data[f'{target}_mlp_all'], color=s.plotting['color_mlp'])

        ax[0].set_title(
            f'{country} {target.capitalize()}: Calculated Values',
            fontsize=scaling*s.plotting['subplot title size']
        )
        ax[1].set_title(
            f'{country} {target.capitalize()}: Gradient Boosted Tree LightGBM {title_modifier}',
            fontsize=scaling*s.plotting['subplot title size']
        )
        ax[2].set_title(
            f'{country} {target.capitalize()}: Gradient Boosted Tree XGBoost Squared Error {title_modifier}',
            fontsize=scaling*s.plotting['subplot title size']
        )
        ax[3].set_title(
            f'{country} {target.capitalize()}: Gradient Boosted Tree XGBoost Absolute Error {title_modifier}',
            fontsize=scaling*s.plotting['subplot title size']
        )
        ax[4].set_title(
            f'{country} {target.capitalize()}: Random Forest LightGBM {title_modifier}',
            fontsize=scaling*s.plotting['subplot title size']
        )
        ax[5].set_title(
            f'{country} {target.capitalize()}: Multi Layer Perceptron {title_modifier}',
            fontsize=scaling*s.plotting['subplot title size']
        )

        ax[0].set_ylabel(target.capitalize(), fontsize=scaling*s.plotting['fontsize'])
        ax[1].set_ylabel(target.capitalize(), fontsize=scaling*s.plotting['fontsize'])
        ax[2].set_ylabel(target.capitalize(), fontsize=scaling*s.plotting['fontsize'])
        ax[3].set_ylabel(target.capitalize(), fontsize=scaling*s.plotting['fontsize'])
        ax[4].set_ylabel(target.capitalize(), fontsize=scaling*s.plotting['fontsize'])
        ax[5].set_ylabel(target.capitalize(), fontsize=scaling*s.plotting['fontsize'])

        ax[5].set_xlabel('Time', fontsize=scaling*s.plotting['fontsize'])

        plt.tight_layout()
        plt.savefig(
            f'../results/{files_prefix}plots/over_time/{country}_{files_prefix.replace('/', '_')}{target}_over_time.png'
        )
        plt.savefig(
            f'../results/{files_prefix}plots/over_time/{country}_{files_prefix.replace('/', '_')}{target}_over_time.pdf'
        )


def visualize_shap_values(country: str, maxdisplay: int = 11, feature1: int = 1, feature2: int = 2) -> None:
    data_detrended = pd.read_hdf(f'../results/prepared_features/{country}_detrended_ml.h5')
    feature_columns = [column for column in data_detrended.columns if column not in ['drift', 'diffusion']]
    X = data_detrended[feature_columns].copy()

    if s.ml['random noise']:
        np.random.seed(42)
        X['random_noise'] = np.random.rand(X.shape[0])

    for target in ['drift', 'diffusion']:
        # get shap values
        path_prefix = f'../results/{files_prefix}explainers/{country}_detrended_{target}'
        with open(f'{path_prefix}_gbt_lgb_shap_values', 'rb') as file:
            shap_values_gbt_lgb = (jbl.load(file))(
                X.drop(columns=s.top_features[area][target]['gbt_lgb']) if s.ml['knockout'] else X
            )
        with open(f'{path_prefix}_gbt_xgb_squarederror_shap_values', 'rb') as file:
            shap_values_gbt_xgb_squarederror = (jbl.load(file))(
                X.drop(columns=s.top_features[area][target]['gbt_xgb_squarederror']) if s.ml['knockout'] else X
            )
        with open(f'{path_prefix}_gbt_xgb_absoluteerror_shap_values', 'rb') as file:
            shap_values_gbt_xgb_absoluteerror = (jbl.load(file))(
                X.drop(columns=s.top_features[area][target]['gbt_xgb_absoluteerror']) if s.ml['knockout'] else X
            )
        with open(f'{path_prefix}_rf_lgb_shap_values', 'rb') as file:
            shap_values_rf_lgb = (jbl.load(file))(
                X.drop(columns=s.top_features[area][target]['rf_lgb']) if s.ml['knockout'] else X
            )

        # plot beeswarm plot
        fig, ax = plt.subplots(nrows=2, ncols=2)
        fig.suptitle(f'{country}: {target.capitalize()} Feature Importance{title_modifier}',
                     fontsize=s.plotting['title size'])

        ax[0, 0].set_title(f'GBT LightGBM {target.capitalize()}',
                           fontsize=s.plotting['subplot title size'])
        ax[1, 0].set_title(f'GBT XGBoost {target.capitalize()}, Squared Error',
                           fontsize=s.plotting['subplot title size'])
        ax[0, 1].set_title(f'GBT XGBoost {target.capitalize()}, Absolute Error',
                           fontsize=s.plotting['subplot title size'])
        ax[1, 1].set_title(f'RF LightGBM {target.capitalize()}',
                           fontsize=s.plotting['subplot title size'])

        plt.sca(ax[0, 0])
        shap.plots.beeswarm(shap_values_gbt_lgb, max_display=maxdisplay, show=False)
        plt.sca(ax[1, 0])
        shap.plots.beeswarm(shap_values_gbt_xgb_squarederror, max_display=maxdisplay, show=False)
        plt.sca(ax[0, 1])
        shap.plots.beeswarm(shap_values_gbt_xgb_absoluteerror, max_display=maxdisplay, show=False)
        plt.sca(ax[1, 1])
        shap.plots.beeswarm(shap_values_rf_lgb, max_display=maxdisplay, show=False)

        fig.set_figwidth(20)
        fig.set_figheight(round(maxdisplay * 1.5))
        plt.tight_layout()
        plt.savefig(
            f'../results/{files_prefix}plots/shap/{country}_{files_prefix.replace('/', '_')}{target}_shap_beeswarm.png'
        )
        plt.savefig(
            f'../results/{files_prefix}plots/shap/{country}_{files_prefix.replace('/', '_')}{target}_shap_beeswarm.pdf'
        )

        # plot partial dependency plots
        fig, ax = plt.subplots(nrows=2, ncols=4)
        fig.suptitle(f'{country}: {target.capitalize()} The 2 most important features{title_modifier}',
                     fontsize=s.plotting['title size'])

        ax[0, 0].set_title(f'GBT LightGBM, {feature1}. Feature',
                           fontsize=s.plotting['subplot title size'])
        ax[1, 0].set_title(f'GBT LightGBM, {feature2}. Feature',
                           fontsize=s.plotting['subplot title size'])
        ax[0, 1].set_title(f'GBT XGBoost, Squared Error, {feature1}. Feature',
                           fontsize=s.plotting['subplot title size'])
        ax[1, 1].set_title(f'GBT XGBoost, Squared Error, {feature2}. Feature',
                           fontsize=s.plotting['subplot title size'])
        ax[0, 2].set_title(f'GBT XGBoost, Absolute Error, {feature1}. Feature',
                           fontsize=s.plotting['subplot title size'])
        ax[1, 2].set_title(f'GBT XGBoost, Absolute Error, {feature2}. Feature',
                           fontsize=s.plotting['subplot title size'])
        ax[0, 3].set_title(f'RF LightGBM, {feature1}. Feature',
                           fontsize=s.plotting['subplot title size'])
        ax[1, 3].set_title(f'RF LightGBM, {feature2}. Feature',
                           fontsize=s.plotting['subplot title size'])

        shap.plots.scatter(
            shap_values_gbt_lgb[:, shap_values_gbt_lgb.abs.mean(0).argsort[-feature1]],
            show=False, ax=ax[0, 0], color=s.plotting[f'color {country}'][f'detrended {target}']
        )
        shap.plots.scatter(
            shap_values_gbt_lgb[:, shap_values_gbt_lgb.abs.mean(0).argsort[-feature2]],
            show=False, ax=ax[1, 0], color=s.plotting[f'color {country}'][f'detrended {target}']
        )
        shap.plots.scatter(
            shap_values_gbt_xgb_squarederror[:, shap_values_gbt_xgb_squarederror.abs.mean(0).argsort[-feature1]],
            show=False, ax=ax[0, 1], color=s.plotting[f'color {country}'][f'detrended {target}']
        )
        shap.plots.scatter(
            shap_values_gbt_xgb_squarederror[:, shap_values_gbt_xgb_squarederror.abs.mean(0).argsort[-feature2]],
            show=False, ax=ax[1, 1], color=s.plotting[f'color {country}'][f'detrended {target}']
        )
        shap.plots.scatter(
            shap_values_gbt_xgb_absoluteerror[:, shap_values_gbt_xgb_absoluteerror.abs.mean(0).argsort[-feature1]],
            show=False, ax=ax[0, 2], color=s.plotting[f'color {country}'][f'detrended {target}']
        )
        shap.plots.scatter(
            shap_values_gbt_xgb_absoluteerror[:, shap_values_gbt_xgb_absoluteerror.abs.mean(0).argsort[-feature2]],
            show=False, ax=ax[1, 2], color=s.plotting[f'color {country}'][f'detrended {target}']
        )
        shap.plots.scatter(
            shap_values_rf_lgb[:, shap_values_rf_lgb.abs.mean(0).argsort[-feature1]],
            show=False, ax=ax[0, 3], color=s.plotting[f'color {country}'][f'detrended {target}']
        )
        shap.plots.scatter(
            shap_values_rf_lgb[:, shap_values_rf_lgb.abs.mean(0).argsort[-feature2]],
            show=False, ax=ax[1, 3], color=s.plotting[f'color {country}'][f'detrended {target}']
        )

        fig.set_figwidth(30)
        fig.set_figheight(10)
        plt.tight_layout()
        plt.savefig(
            f'../results/{files_prefix}plots/shap/{country}_{files_prefix.replace('/', '_')}{target}_shap_scatter.png'
        )
        plt.savefig(
            f'../results/{files_prefix}plots/shap/{country}_{files_prefix.replace('/', '_')}{target}_shap_scatter.pdf'
        )


def plot_full_beeswarm_with_random(country: str) -> None:
    # plot the beeswarm plots in a single file each with up to 100 features. Also includes the random noise feature
    data_detrended = pd.read_hdf(f'../results/prepared_features/{country}_detrended_ml.h5')
    feature_columns = [column for column in data_detrended.columns if column not in ['drift', 'diffusion']]
    X = data_detrended[feature_columns].copy()

    np.random.seed(42)
    X['random_noise'] = np.random.rand(X.shape[0])

    for target in ['drift', 'diffusion']:
        # get shap values
        path_prefix = f'../results/ml_v2_random_noise/explainers/{country}_detrended_{target}'
        with open(f'{path_prefix}_gbt_lgb_shap_values', 'rb') as file:
            shap_values_gbt_lgb = (jbl.load(file))(X)
        with open(f'{path_prefix}_gbt_xgb_squarederror_shap_values', 'rb') as file:
            shap_values_gbt_xgb_squarederror = (jbl.load(file))(X)
        with open(f'{path_prefix}_gbt_xgb_absoluteerror_shap_values', 'rb') as file:
            shap_values_gbt_xgb_absoluteerror = (jbl.load(file))(X)
        with open(f'{path_prefix}_rf_lgb_shap_values', 'rb') as file:
            shap_values_rf_lgb = (jbl.load(file))(X)

        # plot beeswarm plot
        fig = plt.figure()
        plt.title(f'{country}: GBT LightGBM {target.capitalize()} \n'
                  f'Feature Importance with Random Noise Feature',
                  fontsize=s.plotting['title size'])
        shap.plots.beeswarm(shap_values_gbt_lgb, max_display=100, show=False)
        fig.set_figwidth(10)
        plt.tight_layout()
        plt.savefig(f'../results/ml_v2_random_noise/plots/shap/full/{country}_gbt_lgb_full_{target}_beeswarm.png')
        plt.savefig(f'../results/ml_v2_random_noise/plots/shap/full/{country}_gbt_lgb_full_{target}_beeswarm.pdf')

        fig = plt.figure()
        plt.title(
            f'{country}: GBT XGBoost, Squared Error {target.capitalize()} \n'
            f'Feature Importance with Random Noise Feature',
            fontsize=s.plotting['title size']
        )
        shap.plots.beeswarm(shap_values_gbt_xgb_squarederror, max_display=100, show=False)
        fig.set_figwidth(10)
        plt.tight_layout()
        plt.savefig(f'../results/ml_v2_random_noise/plots/shap/full/{country}_gbt_xgb_se_full_{target}_beeswarm.png')
        plt.savefig(f'../results/ml_v2_random_noise/plots/shap/full/{country}_gbt_xgb_se_full_{target}_beeswarm.pdf')

        fig = plt.figure()
        plt.title(
            f'{country}: GBT XGBoost, Absolute Error {target.capitalize()} \n'
            f'Feature Importance with Random Noise Feature',
            fontsize=s.plotting['title size']
        )
        shap.plots.beeswarm(shap_values_gbt_xgb_absoluteerror, max_display=100, show=False)
        fig.set_figwidth(10)
        plt.tight_layout()
        plt.savefig(f'../results/ml_v2_random_noise/plots/shap/full/{country}_gbt_xgb_ae_full_{target}_beeswarm.png')
        plt.savefig(f'../results/ml_v2_random_noise/plots/shap/full/{country}_gbt_xgb_ae_full_{target}_beeswarm.pdf')

        fig = plt.figure()
        plt.title(f'{country}: RF LightGBM {target.capitalize()} \n'
                  f'Feature Importance with Random Noise Feature',
                  fontsize=s.plotting['title size']
                  )
        shap.plots.beeswarm(shap_values_rf_lgb, max_display=100, show=False)
        fig.set_figwidth(10)
        plt.tight_layout()
        plt.savefig(f'../results/ml_v2_random_noise/plots/shap/full/{country}_rf_lgb_full_{target}_beeswarm.png')
        plt.savefig(f'../results/ml_v2_random_noise/plots/shap/full/{country}_rf_lgb_full_{target}_beeswarm.pdf')


for area in ['AUS', 'CE']:
    plot_boxplot(area)
    all_predictions(area)
    visualize_shap_values(area, 11, 1, 2)
#   plot_full_beeswarm_with_random(area)
