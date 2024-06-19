import random
from matplotlib import pyplot as plt
import numpy as np

# Library for the gaussian kernel filter
from scipy.ndimage import gaussian_filter1d

# Library for calculation Kramers-Moyal coefficients
from kramersmoyal import km

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

    # plot the drift for k random hours
    for i in random.sample(range(hours), 4):
        bins = np.array([6000])
        powers = [0, 2]

        # get diffusion coefficient
        diffusion, space = km(frequency[vph * i:vph * i + vph], powers=powers, bins=bins, bw=s.km[area]['diffusion bw'])

        # normalize diffusion
        diffusion = diffusion / s.km[area]['delta_t']

        offset = 1000
        plt.plot(
            space[0][offset:-offset], diffusion[1][offset:-offset],
            color=s.plotting[f'color {area}']['detrended diffusion']
        )

    ax = plt.gca()
    ax.set_xlim(-0.3, 0.3)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, pos: f'{s.settings[area]['reference frequency'] + x / (2 * np.pi):.3f}')
    )

    plt.xlabel('f (Hz)', fontsize=s.plotting['fontsize'])
    plt.ylabel(r'Diffusion $D_2(\omega)$', fontsize=s.plotting['fontsize'])
    plt.title(f'{area} Detrended Diffusion coefficient', fontsize=s.plotting['title size'])
    plt.tight_layout()
    plt.savefig(f'../results/km/plots/diffusions/{area}_detrended_diffusion_samples.png')
    plt.savefig(f'../results/km/plots/diffusions/{area}_detrended_diffusion_samples.pdf')
