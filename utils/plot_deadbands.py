import random
from matplotlib import pyplot as plt
import numpy as np

# Library for the gaussian kernel filter
from scipy.ndimage import gaussian_filter1d

from km_functions import km_get_drift
from helper_functions import get_frequency_data, to_angular_freq
from utils import settings as s


for area in ['AUS', 'CE']:
    freq = get_frequency_data(area)
    angular_freq = to_angular_freq(freq, area)
    frequency = angular_freq['frequency']

    # detrend data
    data_filter = gaussian_filter1d(frequency, sigma=s.settings[area]['detrend sigma'])
    frequency = frequency - data_filter

    # frequency measurements/values per hour
    vph = s.settings[area]['values per hour']
    hours = frequency.size // vph

    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(3)

    # plot the drift for 4 random hours
    for i in random.sample(range(hours), 4):
        drift, space = km_get_drift(
            frequency[vph * i:vph * i + vph], s.km[area]['drift bw'], s.km[area]['delta_t']
        )

        offset = 500

        plt.plot(
            space[0][offset:-offset], drift[1][offset:-offset],
            color=s.plotting[f'color {area}']['detrended drift']
        )

    ax = plt.gca()
    ax.set_xlim(-0.3, 0.3)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, pos: f'{s.settings[area]['reference frequency'] + x / (2 * np.pi):.3f}')
    )

    plt.xlabel('f (Hz)', fontsize=s.plotting['fontsize'])
    plt.ylabel(r'Drift $D_1(\omega)$', fontsize=s.plotting['fontsize'])
    plt.title(f'{area} Detrended Drift coefficient', fontsize=s.plotting['title size'])
    plt.tight_layout()
    plt.savefig(f'../results/km/plots/deadbands/{area}_detrended_deadband_samples.png')
    plt.savefig(f'../results/km/plots/deadbands/{area}_detrended_deadband_samples.pdf')
