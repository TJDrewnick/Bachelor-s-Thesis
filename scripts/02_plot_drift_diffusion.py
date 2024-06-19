# This script plots the results of the calculated drift and diffusion for multiple different time intervals
import numpy as np
import pandas as pd

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

from utils import settings as s
from utils.helper_functions import get_frequency_data


def save_statistics(data: pd.DataFrame, model_type: str) -> None:
    with open('../results/km/km_results.txt', mode='a') as file:
        file.write(f'\n{model_type}\n')
        # Average value of the parameter
        file.write(f'Mean: {data.mean()}\n')
        # Average of the squared differences from the mean
        file.write(f'Variance: {data.var()}\n')
        # Measures the asymmetry of the parameter distribution
        file.write(f'Skewness: {data.skew()}\n')
        # Measures the peakedness or flatness of the parameter distribution
        file.write(f'Kurtosis: {data.kurt()}\n')
        # Measures the spread of the data from its mean
        file.write(f'Standard Deviation: {data.std()}\n')


def filter_outliers(df: pd.DataFrame, parameter: str, percent: float) -> pd.DataFrame:
    lower = np.percentile(df[parameter], percent)
    upper = np.percentile(df[parameter], 100 - percent)
    return df[parameter][(df[parameter] >= lower) & (df[parameter] <= upper)]


def plot_time_interval(values: pd.DataFrame, days: int, country: str, parameter: str, limit: int) -> None:
    fig, (ax_original, ax_detrended) = plt.subplots(2, sharex=True, sharey=True)
    fig.set_figwidth(30)
    fig.set_figheight(10)
    fig.suptitle(f'{country}: {days} days {parameter.capitalize()}', fontsize=s.plotting['title size'])

    hours = days * 24
    amount = limit if (round(values.shape[0] / hours) > limit) else round(values.shape[0] / hours)
    legend_labels = []

    alpha = 0.5 if s.plotting['rolling average'] else 1

    for i in range(amount):
        offset = i * hours
        first_day = values.index[0] + pd.Timedelta(offset, 'h')
        last_day = first_day + pd.Timedelta(hours - 1, 'h')

        ax_original.plot(
            values[first_day:last_day].index - pd.Timedelta(offset, 'h'),
            values[first_day:last_day]['original'],
            alpha=alpha,
        )
        ax_detrended.plot(
            values[first_day:last_day].index - pd.Timedelta(offset, 'h'),
            values[first_day:last_day]['detrended'],
            alpha=alpha,
        )

        legend_labels.append(f'+{i*days} days offset')

    # add rolling average
    if s.plotting['rolling average']:
        first_day = values.index[0]
        last_day = first_day + pd.Timedelta(hours, 'h')

        ax_original.plot(
            values[first_day:last_day].index,
            values[first_day:last_day]['original'].rolling(window=5).mean(),
            color='green'
        )
        ax_detrended.plot(
            values[first_day:last_day].index,
            values[first_day:last_day]['detrended'].rolling(window=5).mean(),
            color='green'
        )
        legend_labels.append('Rolling average')

    ax_original.set_ylabel(f'Original data\n\n{parameter.capitalize()}', fontsize=s.plotting['fontsize'])
    ax_detrended.set_ylabel(f'Detrended data\n\n{parameter.capitalize()}', fontsize=s.plotting['fontsize'])
    ax_detrended.set_xlabel('Time', fontsize=s.plotting['fontsize'])

    plt.figlegend(legend_labels, framealpha=1, prop=fm.FontProperties(size=s.plotting['fontsize']))
    fig.tight_layout()
    plt.savefig(f'../results/km/plots/drift_diffusion_sequential/{country}_{parameter}_{days}days_sequential.png')
    plt.savefig(f'../results/km/plots/drift_diffusion_sequential/{country}_{parameter}_{days}days_sequential.pdf')


def plot_time_interval_averaged(values: pd.DataFrame, days: int, country: str, parameter: str) -> None:
    fig, (ax_original, ax_detrended) = plt.subplots(2, sharex=True)
    fig.set_figwidth(30)
    fig.set_figheight(10)
    fig.suptitle(f'{country}: {days} days {parameter.capitalize()} averaged', fontsize=3*s.plotting['title size'])

    hours = days * 24

    values['group'] = ((values.index - values.index[0]).total_seconds() // 3600) % hours

    original_avg = values.groupby('group')['original'].mean()
    detrended_avg = values.groupby('group')['detrended'].mean()

    ax_original.plot(original_avg, color=s.plotting[f'color {country}'][f'original {parameter}'], alpha=0.7)
    ax_detrended.plot(detrended_avg, color=s.plotting[f'color {country}'][f'detrended {parameter}'], alpha=0.7)

    xticks = [x for x in original_avg.index if x % (24 if days < 50 else (24 * 7 if days < 100 else 24 * 28)) == 0]
    xtick_labels = [f'{int(x // 24)}' for x in xticks]
    plt.xticks(ticks=xticks, labels=xtick_labels)

    ax_original.tick_params(axis='both', which='major', labelsize=1.5*s.plotting['fontsize'])
    ax_detrended.tick_params(axis='both', which='major', labelsize=1.5*s.plotting['fontsize'])
    ax_original.tick_params(axis='both', which='minor', labelsize=1.5*s.plotting['fontsize'])
    ax_detrended.tick_params(axis='both', which='minor', labelsize=1.5*s.plotting['fontsize'])

    ax_original.set_ylabel(f'Original\n{parameter.capitalize()}', fontsize=3*s.plotting['fontsize'])
    ax_detrended.set_ylabel(f'Detrended\n{parameter.capitalize()}', fontsize=3*s.plotting['fontsize'])
    ax_detrended.set_xlabel('Time in days', fontsize=3*s.plotting['fontsize'])

    # rolling average for windows of size 28 or larger
    if days > 27 and s.plotting['rolling average']:
        window = days // 7 if days < 49 else 7
        ax_original.plot(original_avg.rolling(window=window).mean(), color='red')
        ax_detrended.plot(detrended_avg.rolling(window=window).mean(), color='red')
        plt.figlegend(['avg data', 'rolling average of avg data'], framealpha=1)

    fig.tight_layout()
    plt.savefig(f'../results/km/plots/drift_diffusion_averaged/{country}_{parameter}_{days}days_averaged.png')
    plt.savefig(f'../results/km/plots/drift_diffusion_averaged/{country}_{parameter}_{days}days_averaged.pdf')


def plot_time_interval_averaged_with_frequency(values: pd.DataFrame, freq: pd.DataFrame, days: int, country: str) -> None:
    fig, (ax_freq, ax_drift, ax_diffusion) = plt.subplots(3)
    fig.set_figwidth(30)
    fig.set_figheight(14)
    fig.suptitle(
        f'{country}: {days} days detrended Drift, Diffusion and Frequency averaged',
        fontsize=3*s.plotting['title size']
    )

    hours = days * 24
    sph = 3600
    values['group'] = ((values.index - values.index[0]).total_seconds() // sph) % hours

    match days:
        case x if 0 <= x <= 10:
            tick_interval = 24
            res = 2
        case x if 11 <= x <= 30:
            tick_interval = 24
            res = 60
        case x if 31 <= x <= 100:
            tick_interval = 24 * 7
            res = 60 * 10
        case _:
            tick_interval = 24 * 28
            res = 60 * 60
    freq['group'] = ((freq.index - freq.index[0]).total_seconds() // res) % (hours * (sph / res))

    drift_avg = values.groupby('group')['drift'].mean()
    diffusion_avg = values.groupby('group')['diffusion'].mean()
    freq_avg = freq.groupby('group')['frequency'].mean()

    ax_drift.plot(
        drift_avg,
        color=s.plotting[f'color {country}']['detrended drift'],
        alpha=0.7,
        label='avg drift'
    )
    ax_diffusion.plot(
        diffusion_avg,
        color=s.plotting[f'color {country}']['detrended diffusion'],
        alpha=0.7,
        label='avg diffusion'
    )
    ax_freq.plot(
        freq_avg,
        color=s.plotting[f'color {country}']['frequency'],
        alpha=0.7)

    xticks = [x for x in diffusion_avg.index if x % tick_interval == 0]
    xticks_freq = [x for x in freq_avg.index if x % (tick_interval * (sph / res)) == 0]

    ax_freq.set_xticks(xticks_freq)
    ax_diffusion.set_xticks(xticks)
    ax_drift.set_xticks(xticks)

    xtick_labels = [f'{int(x // 24)}' for x in xticks]
    for ax in (ax_freq, ax_drift, ax_diffusion):
        ax.set_xticklabels(xtick_labels)
        ax.tick_params(axis='both', which='major', labelsize=s.plotting['fontsize'])
        ax.tick_params(axis='both', which='minor', labelsize=s.plotting['fontsize'])

    ax_drift.set_ylabel('Drift', fontsize=3*s.plotting['fontsize'])
    ax_diffusion.set_ylabel('Diffusion', fontsize=3*s.plotting['fontsize'])
    ax_freq.set_ylabel(f'Frequency', fontsize=3*s.plotting['fontsize'])

    ax_drift.set_xlabel('Time in days', fontsize=3*s.plotting['fontsize'])
    ax_diffusion.set_xlabel('Time in days', fontsize=3*s.plotting['fontsize'])
    ax_freq.set_xlabel('Time in days', fontsize=3*s.plotting['fontsize'])

    # rolling average for windows of size 28 or larger
    if days > 27 and s.plotting['rolling average']:
        window = days // 7 if days < 49 else 7
        ax_drift.plot(
            drift_avg.rolling(window=window).mean(),
            color='red',
            label='rolling average of avg drift'
        )
        ax_diffusion.plot(
            diffusion_avg.rolling(window=window).mean(),
            color='red',
            label='rolling average of avg diffusion'
        )
        ax_drift.legend(loc='upper left')
        ax_diffusion.legend(loc='upper left')

    fig.tight_layout()
    plt.savefig(f'../results/km/plots/drift_diffusion_averaged_with_freq/{country}_{days}days_averaged.png')
    plt.savefig(f'../results/km/plots/drift_diffusion_averaged_with_freq/{country}_{days}days_averaged.pdf')


histogram_bins = {
    'AUS':  40,
    'CE':  50
}


def plot_histogram(data_original: pd.DataFrame, data_detrended: pd.DataFrame, country: str, target: str):
    counts_original, bins_original = np.histogram(data_original, bins=histogram_bins[country])
    counts_detrended, bins_detrended = np.histogram(data_detrended, bins=histogram_bins[country])

    title = f'Histogram of {country}: {target.capitalize()}'
    xlabel = r'Drift $D_1(\omega)$' if target == 'drift' else r'Diffusion $D_2(\omega)$'

    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(4)
    plt.stairs(counts_original, bins_original, color=s.plotting[f'color {country}'][f'original {target}'])
    plt.stairs(counts_detrended, bins_detrended, color=s.plotting[f'color {country}'][f'detrended {target}'])
    plt.legend(['Original Data', 'Detrended Data'], prop=fm.FontProperties(size=s.plotting['fontsize']))
    plt.title(title, fontsize=s.plotting['title size'])
    plt.xlabel(xlabel, fontsize=s.plotting['fontsize'])
    plt.ylabel("Occurrences", fontsize=s.plotting['fontsize'])

    plt.tight_layout()
    plt.savefig(f'../results/km/plots/histograms/{country}_{target}_histogram.png')
    plt.savefig(f'../results/km/plots/histograms/{country}_{target}_histogram.pdf')


# clear km results file
open('../results/km/km_results.txt', 'w').close()

for area in ['AUS', 'CE']:
    df_original: pd.DataFrame = pd.read_hdf(f'../results/km/{area}_original_drift_diffusion.h5')
    df_detrended: pd.DataFrame = pd.read_hdf(f'../results/km/{area}_detrended_drift_diffusion.h5')

    # for australia remove first day so both time series begin on monday
    if area == 'AUS':
        df_original = df_original[df_original.index >= '2023-01-02 00:00:00']
        df_detrended = df_detrended[df_detrended.index >= '2023-01-02 00:00:00']

    # plot the drift and diffusion as defined in the settings.py file
    for param in ['drift', 'diffusion']:
        # remove outliers
        original = filter_outliers(df_original, param, s.settings[area]['outlier percent'])
        detrended = filter_outliers(df_detrended, param, s.settings[area]['outlier percent'])

        save_statistics(original, f'{area}_Original_{param.capitalize()}')
        save_statistics(detrended, f'{area}_Detrended_{param.capitalize()}')

        plot_histogram(original, detrended, area, param)

        combined = pd.DataFrame({'original': original, 'detrended': detrended})

        for time_interval in [7, 28, 91, 365]:
            if s.plotting['plotting type'] == 'average':
                plot_time_interval_averaged(combined, time_interval, area, param)
            elif s.plotting['plotting type'] == 'sequential':
                plot_time_interval(combined, time_interval, area, param, s.plotting['sequential limit'])

    if s.plotting['plotting type'] == 'average and frequency':
        drift = filter_outliers(df_detrended, 'drift', s.settings[area]['outlier percent'])
        diffusion = filter_outliers(df_detrended, 'diffusion', s.settings[area]['outlier percent'])
        frequency = get_frequency_data(area)
        frequency.set_index('timestamp', inplace=True)

        combined = pd.DataFrame({'drift': drift, 'diffusion': diffusion})
        for time_interval in [7, 28, 91, 365]:
            plot_time_interval_averaged_with_frequency(combined, frequency, time_interval, area)