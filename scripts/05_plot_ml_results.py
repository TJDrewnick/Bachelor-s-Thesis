import matplotlib.pyplot as plt
import pandas as pd
import shap
import joblib as jbl

from utils import settings as s


def plot_boxplot(country: str) -> None:
    data_original = pd.read_hdf(f'../results/ml_v1/predictions/{country}_original_predictions.h5')
    data_detrended = pd.read_hdf(f'../results/ml_v1/predictions/{country}_detrended_predictions.h5')

    fig, ax = plt.subplots(2, 2, figsize=(15, 15), sharey='row')
    fig.suptitle(f'{country}: ML Predictions', fontsize=s.plotting['title size'])

    labels = ['Calculated Results', 'LightGBM Predictions', 'XGBoost Predictions']

    ax[0, 0].set_title('Original Data Drift', fontsize=s.plotting['subplot title size'])
    ax[0, 1].set_title('Detrended Data Drift', fontsize=s.plotting['subplot title size'])
    ax[1, 0].set_title('Original Data Diffusion', fontsize=s.plotting['subplot title size'])
    ax[1, 1].set_title('Detrended Data Diffusion', fontsize=s.plotting['subplot title size'])

    ax[0, 0].set_ylabel(r'Drift $D_1(\omega)$', fontsize=s.plotting['fontsize'])
    ax[1, 0].set_ylabel(r'Diffusion $D_2(\omega)$', fontsize=s.plotting['fontsize'])

    ax[0, 0].yaxis.grid(True)
    ax[0, 1].yaxis.grid(True)
    ax[1, 0].yaxis.grid(True)
    ax[1, 1].yaxis.grid(True)

    bplot_drift_original = ax[0, 0].boxplot(
        [data_original['drift_true'],
         data_original['drift_gbt_lgb'],
         data_original['drift_gbt_xgb_squarederror']],
        vert=True, patch_artist=True, labels=labels, medianprops=dict(color='black')
    )
    bplot_drift_detrended = ax[0, 1].boxplot(
        [data_detrended['drift_true'],
         data_detrended['drift_gbt_lgb'],
         data_detrended['drift_gbt_xgb_squarederror']],
        vert=True, patch_artist=True, labels=labels, medianprops=dict(color='black')
    )
    bplot_diffusion_original = ax[1, 0].boxplot(
        [data_original['diffusion_true'],
         data_original['diffusion_gbt_lgb'],
         data_original['diffusion_gbt_xgb_squarederror']],
        vert=True, patch_artist=True, labels=labels, medianprops=dict(color='black')
    )
    bplot_diffusion_detrended = ax[1, 1].boxplot(
        [data_detrended['diffusion_true'],
         data_detrended['diffusion_gbt_lgb'],
         data_detrended['diffusion_gbt_xgb_squarederror']],
        vert=True, patch_artist=True, labels=labels, medianprops=dict(color='black')
    )

    # color plot
    targets = ['original drift', 'detrended drift', 'original diffusion', 'detrended diffusion']
    plots = [bplot_drift_original, bplot_drift_detrended, bplot_diffusion_original, bplot_diffusion_detrended]
    for target, bplot in zip(targets, plots):
        color_list = [
            s.plotting[f'color {country}'][target],
            s.plotting['color_gbt_lgb'],
            s.plotting['color_gbt_xgb_squarederror']
        ]
        for patch, color in zip(bplot['boxes'], color_list):
            patch.set_facecolor(color)

    plt.tight_layout()
    plt.savefig(f'../results/ml_v1/plots/boxplot/{country}_ml_v1_boxplot.png')
    plt.savefig(f'../results/ml_v1/plots/boxplot/{country}_ml_v1_boxplot.pdf')


def all_predictions(country: str) -> None:
    data_original = pd.read_hdf(f'../results/ml_v1/predictions/{country}_original_all_predictions.h5')
    data_detrended = pd.read_hdf(f'../results/ml_v1/predictions/{country}_detrended_all_predictions.h5')

    for target in ['drift', 'diffusion']:
        fig = plt.figure()
        fig.set_figwidth(30)
        fig.set_figheight(18)

        ax0 = plt.subplot(6, 1, 1)
        ax1 = plt.subplot(6, 1, 2, sharex=ax0, sharey=ax0)
        ax2 = plt.subplot(6, 1, 3, sharex=ax0, sharey=ax0)
        ax3 = plt.subplot(6, 1, 4, sharex=ax0)
        ax4 = plt.subplot(6, 1, 5, sharex=ax0, sharey=ax3)
        ax5 = plt.subplot(6, 1, 6, sharex=ax0, sharey=ax3)

        ax1.tick_params(axis='y', labelbottom='off')

        ax0.plot(data_original[f'{target}_true_all'], color=s.plotting[f'color {country}'][f'original {target}'])
        ax1.plot(data_original[f'{target}_gbt_lgb_all'], color=s.plotting['color_gbt_lgb'])
        ax2.plot(data_original[f'{target}_gbt_xgb_squarederror_all'], color=s.plotting['color_gbt_xgb_squarederror'])
        ax3.plot(data_detrended[f'{target}_true_all'], color=s.plotting[f'color {country}'][f'detrended {target}'])
        ax4.plot(data_detrended[f'{target}_gbt_lgb_all'], color=s.plotting['color_gbt_lgb'])
        ax5.plot(data_detrended[f'{target}_gbt_xgb_squarederror_all'], color=s.plotting['color_gbt_xgb_squarederror'])

        ax0.set_title(f'{country}: {target.capitalize()} using Original Data', fontsize=s.plotting['title size'])
        ax3.set_title(f'{country}: {target.capitalize()} using Detrended Data', fontsize=s.plotting['title size'])

        ax0.set_ylabel(f'Calculated {target.capitalize()}', fontsize=s.plotting['fontsize'])
        ax1.set_ylabel(f'LightGBM predicted {target.capitalize()}', fontsize=s.plotting['fontsize'])
        ax2.set_ylabel(f'XGBoost predicted {target.capitalize()}', fontsize=s.plotting['fontsize'])
        ax3.set_ylabel(f'Calculated {target.capitalize()}', fontsize=s.plotting['fontsize'])
        ax4.set_ylabel(f'LightGBM predicted {target.capitalize()}', fontsize=s.plotting['fontsize'])
        ax5.set_ylabel(f'XGBoost predicted {target.capitalize()}', fontsize=s.plotting['fontsize'])

        ax5.set_xlabel('Time', fontsize=s.plotting['fontsize'])

        plt.tight_layout()
        plt.savefig(f'../results/ml_v1/plots/over_time/{country}_ml_v1_{target}_over_time.png')
        plt.savefig(f'../results/ml_v1/plots/over_time/{country}_ml_v1_{target}_over_time.pdf')


def visualize_shap_values(country: str, maxdisplay: int = 11, feature1: int = 1, feature2: int = 2) -> None:
    data_original = pd.read_hdf(f'../results/prepared_features/{country}_original_ml.h5')
    feature_columns = [column for column in data_original.columns if column not in ['drift', 'diffusion']]
    X_original = data_original[feature_columns]

    data_detrended = pd.read_hdf(f'../results/prepared_features/{country}_detrended_ml.h5')
    feature_columns = [column for column in data_detrended.columns if column not in ['drift', 'diffusion']]
    X_detrended = data_detrended[feature_columns]

    for target in ['drift', 'diffusion']:

        # get shap values
        path_prefix = f'../results/ml_v1/explainers/{country}'
        with open(f'{path_prefix}_original_{target}_gbt_lgb_shap_values', 'rb') as file:
            shap_values_original_lgb = (jbl.load(file))(X_original)
        with open(f'{path_prefix}_original_{target}_gbt_xgb_squarederror_shap_values', 'rb') as file:
            shap_values_original_xgb = (jbl.load(file))(X_original)
        with open(f'{path_prefix}_detrended_{target}_gbt_lgb_shap_values', 'rb') as file:
            shap_values_detrended_lgb = (jbl.load(file))(X_detrended)
        with open(f'{path_prefix}_detrended_{target}_gbt_xgb_squarederror_shap_values', 'rb') as file:
            shap_values_detrended_xgb = (jbl.load(file))(X_detrended)

        # plot beeswarm plot
        fig, ax = plt.subplots(nrows=2, ncols=2)
        fig.suptitle(f'{country}: {target.capitalize()} Feature Importance', fontsize=s.plotting['title size'])

        plt.sca(ax[0, 0])
        ax[0, 0].set_title('LightGBM Original Data', fontsize=s.plotting['subplot title size'])
        shap.plots.beeswarm(shap_values_original_lgb, max_display=maxdisplay, show=False)

        plt.sca(ax[0, 1])
        ax[0, 1].set_title('LightGBM Detrended Data', fontsize=s.plotting['subplot title size'])
        shap.plots.beeswarm(shap_values_detrended_lgb, max_display=maxdisplay, show=False)

        plt.sca(ax[1, 0])
        ax[1, 0].set_title('XGBoost Original Data', fontsize=s.plotting['subplot title size'])
        shap.plots.beeswarm(shap_values_original_xgb, max_display=maxdisplay, show=False)

        plt.sca(ax[1, 1])
        ax[1, 1].set_title('XGBoost Detrended Data', fontsize=s.plotting['subplot title size'])
        shap.plots.beeswarm(shap_values_detrended_xgb, max_display=maxdisplay, show=False)

        fig.set_figwidth(20)
        fig.set_figheight(15)
        plt.tight_layout()
        plt.savefig(f'../results/ml_v1/plots/shap/{country}_ml_v1_{target}_shap_beeswarm.png')
        plt.savefig(f'../results/ml_v1/plots/shap/{country}_ml_v1_{target}_shap_beeswarm.pdf')

        # plot partial dependency plots
        fig, ax = plt.subplots(nrows=2, ncols=4)
        fig.suptitle(
            f'{country}: {target.capitalize()} The 2 most important features', fontsize=s.plotting['title size']
        )

        ax[0, 0].set_title(f'LightGBM, Original, {feature1}. Feature', fontsize=s.plotting['subplot title size'])
        shap.plots.scatter(
            shap_values_original_lgb[:, shap_values_original_lgb.abs.mean(0).argsort[-feature1]],
            show=False, ax=ax[0, 0], color=s.plotting[f'color {country}'][f'original {target}']
        )
        ax[0, 1].set_title(f'LightGBM, Original, {feature2}. Feature', fontsize=s.plotting['subplot title size'])
        shap.plots.scatter(
            shap_values_original_lgb[:, shap_values_original_lgb.abs.mean(0).argsort[-feature2]],
            show=False, ax=ax[0, 1], color=s.plotting[f'color {country}'][f'original {target}']
        )

        ax[0, 2].set_title(f'LightGBM, Detrended, {feature1}. Feature', fontsize=s.plotting['subplot title size'])
        shap.plots.scatter(
            shap_values_detrended_lgb[:, shap_values_detrended_lgb.abs.mean(0).argsort[-feature1]],
            show=False, ax=ax[0, 2], color=s.plotting[f'color {country}'][f'detrended {target}']
        )
        ax[0, 3].set_title(f'LightGBM, Detrended, {feature2}. Feature', fontsize=s.plotting['subplot title size'])
        shap.plots.scatter(
            shap_values_detrended_lgb[:, shap_values_detrended_lgb.abs.mean(0).argsort[-feature2]],
            show=False, ax=ax[0, 3], color=s.plotting[f'color {country}'][f'detrended {target}']
        )

        ax[1, 0].set_title(f'XGBoost, Original, {feature1}. Feature', fontsize=s.plotting['subplot title size'])
        shap.plots.scatter(
            shap_values_original_xgb[:, shap_values_original_xgb.abs.mean(0).argsort[-feature1]],
            show=False, ax=ax[1, 0], color=s.plotting[f'color {country}'][f'original {target}']
        )
        ax[1, 1].set_title(f'XGBoost, Original, {feature2}. Feature', fontsize=s.plotting['subplot title size'])
        shap.plots.scatter(
            shap_values_original_xgb[:, shap_values_original_xgb.abs.mean(0).argsort[-feature2]],
            show=False, ax=ax[1, 1], color=s.plotting[f'color {country}'][f'original {target}']
        )

        ax[1, 2].set_title(f'XGBoost, Detrended, {feature1}. Feature', fontsize=s.plotting['subplot title size'])
        shap.plots.scatter(
            shap_values_detrended_xgb[:, shap_values_detrended_xgb.abs.mean(0).argsort[-feature1]],
            show=False, ax=ax[1, 2], color=s.plotting[f'color {country}'][f'detrended {target}']
        )
        ax[1, 3].set_title(f'XGBoost, Detrended, {feature2}. Feature', fontsize=s.plotting['subplot title size'])
        shap.plots.scatter(
            shap_values_detrended_xgb[:, shap_values_detrended_xgb.abs.mean(0).argsort[-feature2]],
            show=False, ax=ax[1, 3], color=s.plotting[f'color {country}'][f'detrended {target}']
        )

        fig.set_figwidth(26)
        fig.set_figheight(10)
        plt.tight_layout()
        plt.savefig(f'../results/ml_v1/plots/shap/{country}_ml_v1_{target}_shap_scatter.png')
        plt.savefig(f'../results/ml_v1/plots/shap/{country}_ml_v1_{target}_shap_scatter.pdf')


for area in ['AUS', 'CE']:
    plot_boxplot(area)
    all_predictions(area)
    visualize_shap_values(area, 11, 1, 2)
