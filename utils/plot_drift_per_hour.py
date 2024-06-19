# We had the assumption that the reason for the drift being positive and negative on the original data was due to
# different power dispatches at different hours. This plot however shows that during all times of day no matter how
# the power dispatch is, the drift is still distributed over both positive and negative values.
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils import settings as s

for area in ['AUS', 'CE']:

    drift_original = pd.read_hdf(f'../results/km/{area}_original_drift_diffusion.h5')['drift']

    lower = np.percentile(drift_original, 2)
    upper = np.percentile(drift_original, 98)

    drift_original = drift_original[(drift_original >= lower) & (drift_original <= upper)]

    # filter the drift into 4 specific hours
    original_6am = drift_original[drift_original.index.hour == 6]
    original_11am = drift_original[drift_original.index.hour == 11]
    original_6pm = drift_original[drift_original.index.hour == 18]
    original_11pm = drift_original[drift_original.index.hour == 23]

    bins = 25

    # plot a histogram for each hour
    counts_6am, bins_6am = np.histogram(original_6am, bins)
    counts_11am, bins_11am = np.histogram(original_11am, bins)
    counts_6pm, bins_6pm = np.histogram(original_6pm, bins)
    counts_11pm, bins_11pm = np.histogram(original_11pm, bins)

    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(4)
    plt.stairs(counts_6am, bins_6am)
    plt.stairs(counts_11am, bins_11am)
    plt.stairs(counts_6pm, bins_6pm)
    plt.stairs(counts_11pm, bins_11pm)
    plt.legend(
        ['6am drift', '11am drift', '6pm drift', '11pm drift'], prop=fm.FontProperties(size=s.plotting['fontsize'])
    )
    plt.title(f'{area}: Drift of original data at specific hours', fontsize=s.plotting['title size'])
    plt.xlabel(r'Drift $D_1(\omega)$', fontsize=s.plotting['fontsize'])
    plt.ylabel("Occurrences", fontsize=s.plotting['fontsize'])

    plt.tight_layout()
    plt.savefig(f'../results/utils/plots/{area}_original_drift_per_hour.png')
    plt.savefig(f'../results/utils/plots/{area}_original_drift_per_hour.pdf')
