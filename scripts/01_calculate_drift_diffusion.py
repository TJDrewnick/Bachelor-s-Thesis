# This script calculates and stores hourly drift and diffusion for AUS and CE

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Library for the gaussian kernel filter
from scipy.ndimage import gaussian_filter1d

from utils import settings as s
from utils.km_functions import km_get_drift, km_get_primary_control, km_get_diffusion
from utils.helper_functions import get_frequency_data, to_angular_freq


histogram_bins = {
    'AUS': 905,
    'CE': 745
}


def plot_frequency(data: pd.DataFrame, country: str) -> None:
    fig, ax = plt.subplots(2, 1)
    fig.suptitle(f'Frequency statistics of {s.settings[country]['name']}', fontsize=s.plotting['title size'])
    fig.set_figwidth(10)
    fig.set_figheight(5)

    # plot frequency over time
    ax[0].plot(data['timestamp'], data['frequency'], color=s.plotting[f'color {country}']['frequency'])
    ax[0].set_xlabel('Time', fontsize=s.plotting['fontsize'])
    ax[0].set_ylabel('Frequency (Hz)', fontsize=s.plotting['fontsize'])
    ax[0].set_ylim(49.7, 50.2)

    # plot histogram of frequency
    counts, bins = np.histogram(data['frequency'], bins=histogram_bins[country])
    ax[1].stairs(counts, bins, color=s.plotting[f'color {country}']['frequency'], fill=True)
    ax[1].set_xlabel('Frequency (Hz)', fontsize=s.plotting['fontsize'])
    ax[1].set_ylabel('Occurrences', fontsize=s.plotting['fontsize'])
    ax[1].set_xlim(49.9, 50.1)

    fig.tight_layout()
    plt.savefig(f'../results/km/plots/frequency/{country}_frequency.png')
    plt.savefig(f'../results/km/plots/frequency/{country}_frequency.pdf')


for area in ['AUS', 'CE']:
    print(f"Calculating drift and diffusion for {s.settings[area]['name']}")
    freq = get_frequency_data(area)
    plot_frequency(freq, area)
    angular_freq = to_angular_freq(freq, area)

    for datatype in ['detrended', 'original']:
        frequency = angular_freq['frequency']

        # frequency measurements/values per hour
        vph = s.settings[area]['values per hour']
        hours = frequency.size // vph
        drifts = np.empty(hours)
        diffusions = np.empty(hours)

        # detrend data
        if datatype == 'detrended':
            data_filter = gaussian_filter1d(frequency, sigma=s.settings[area]['detrend sigma'])
            frequency = frequency - data_filter

        # calculate the drift and diffusion for each hour
        for i in range(hours):
            drift, space = km_get_drift(
                frequency[vph * i:vph * i + vph], s.km[area]['drift bw'], s.km[area]['delta_t']
            )
            drifts[i] = km_get_primary_control(drift, space)
            diffusions[i] = km_get_diffusion(
                frequency[vph * i:vph * i + vph], s.km[area]['diffusion bw'], s.km[area]['delta_t']
            )

        # store the results
        index = angular_freq['hour'].unique()
        drift_diffusion = pd.DataFrame({'drift': drifts, 'diffusion': diffusions}, index=index)
        drift_diffusion.to_hdf(f'../results/km/{area}_{datatype}_drift_diffusion.h5', key='df', mode='w')
