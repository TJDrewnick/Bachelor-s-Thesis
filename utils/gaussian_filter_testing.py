import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from utils import settings as s
from utils.helper_functions import get_frequency_data


fig, ax = plt.subplots(4, 3, sharey='row')
fig.set_figwidth(10)
fig.set_figheight(7.5)
fig.suptitle(f'Gaussian Filter Sigma Variations', fontsize=s.plotting['title size'])

# calculate gaussian filter for AUS and CE with various filter lengths
row = 0
col = 0
for area in ['CE', 'AUS']:

    df = get_frequency_data(area)
    frequency_data = df['frequency'][0:(s.settings[area]['values per hour'] // 2)]

    for sigma in [5, 15, 30, 60, 90, 120]:
        trend = gaussian_filter1d(frequency_data, sigma=sigma)
        detrended_frequency_data = frequency_data - trend

        x = np.linspace(0, frequency_data.size // (s.settings[area]['values per hour'] // 60), frequency_data.size)
        ax[row, col].set_title(f'{area}: trend with sigma={sigma}', fontsize=s.plotting['subplot title size'])
        ax[row, col].plot(x, frequency_data, color=s.plotting[f'color {area}']['frequency'])
        ax[row, col].plot(x, trend, color='green')

        row = (row + 1) if col == 2 else row
        col = (col + 1) if col < 2 else 0


# customize plot
ax[0, 0].set_ylabel('Frequency (Hz)', fontsize=s.plotting['fontsize'])
ax[1, 0].set_ylabel('Frequency (Hz)', fontsize=s.plotting['fontsize'])
ax[2, 0].set_ylabel('Frequency (Hz)', fontsize=s.plotting['fontsize'])
ax[3, 0].set_ylabel('Frequency (Hz)', fontsize=s.plotting['fontsize'])

ax[3, 0].set_xlabel('Time (min)', fontsize=s.plotting['fontsize'])
ax[3, 1].set_xlabel('Time (min)', fontsize=s.plotting['fontsize'])
ax[3, 2].set_xlabel('Time (min)', fontsize=s.plotting['fontsize'])

plt.tight_layout()
plt.savefig(f'../results/utils/plots/gaussian_filter_test.png')
plt.savefig(f'../results/utils/plots/gaussian_filter_test.pdf')

# Australian data has a resolution of 4 seconds. To be in line with CE data which has a resolution of 1 second,
# a time-resolution of 60 seconds was chosen for the filter. This results in a gaussian filter of 15 seconds.
# It also allows us to capture more of the frequency variation without following it too strongly.
