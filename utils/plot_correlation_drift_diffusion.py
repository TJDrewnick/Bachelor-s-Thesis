import matplotlib.pyplot as plt
import pandas as pd

from utils import settings as s

data_aus = pd.read_hdf(f'../results/km/AUS_detrended_drift_diffusion.h5')
data_ce = pd.read_hdf(f'../results/km/CE_detrended_drift_diffusion.h5')

# plot the correlation of the drift with diffusion for Australia and Continental Europe
fig, ax = plt.subplots(1, 2)
fig.set_figwidth(10)
fig.set_figheight(6)
fig.suptitle(f'Correlation of Drift with Diffusion', fontsize=s.plotting['title size'])

ax[0].set_title(
    f'Australia\ncorr = {round(data_aus['drift'].corr(data_aus['diffusion']), 2)}',
    fontsize=s.plotting['subplot title size']
)
ax[1].set_title(
    f'Continental Europe\ncorr = {round(data_ce['drift'].corr(data_ce['diffusion']), 2)}',
    fontsize=s.plotting['subplot title size']
)

ax[0].scatter(data_aus['drift'], data_aus['diffusion'], color=s.plotting['color AUS']['detrended drift'])
ax[1].scatter(data_ce['drift'], data_ce['diffusion'], color=s.plotting['color CE']['detrended drift'])

ax[0].set_ylabel(r'Diffusion $D_2(\omega)$', fontsize=s.plotting['fontsize'])
ax[1].set_ylabel(r'Diffusion $D_2(\omega)$', fontsize=s.plotting['fontsize'])

ax[0].set_xlabel(r'Drift $D_1(\omega)$', fontsize=s.plotting['fontsize'])
ax[1].set_xlabel(r'Drift $D_1(\omega)$', fontsize=s.plotting['fontsize'])

ax[0].set_xlim(0, 0.2)
ax[0].set_ylim(0, 0.00125)
ax[1].set_xlim(0, 0.04)
ax[1].set_ylim(0, 0.00025)

plt.tight_layout()
plt.savefig(f'../results/utils/plots/correlation_drift_with_diffusion.png')
plt.savefig(f'../results/utils/plots/correlation_drift_with_diffusion.pdf')
