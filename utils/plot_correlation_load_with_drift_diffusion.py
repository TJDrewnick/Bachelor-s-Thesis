import matplotlib.pyplot as plt
import pandas as pd

from utils import settings as s

data_aus = pd.read_hdf(f'../results/km/AUS_detrended_drift_diffusion.h5')
data_ce = pd.read_hdf(f'../results/km/CE_detrended_drift_diffusion.h5')

# get load for Australia and Continental Europe
load_aus = pd.read_hdf(f'../results/prepared_features/AUS_original_ml.h5')[
    ['Battery (Charging)', 'Battery (Discharging)', 'Biomass', 'Coal (Black)', 'Coal (Brown)', 'Distillate',
     'Gas (CCGT)', 'Gas (OCGT)', 'Gas (Reciprocating)', 'Gas (Steam)', 'Gas (Coal Mine Waste)', 'Hydro', 'Pumps',
     'Solar (Utility)', 'Solar (Thermal)', 'Wind', 'Nuclear', 'Biogas', 'Solar (Rooftop)']].sum(axis=1)
load_ce = pd.read_hdf(f'../results/prepared_features/CE_original_ml.h5')['load']

aus = data_aus.join(pd.DataFrame({'load': load_aus}), how='inner')
ce = data_ce.join(pd.DataFrame({'load': load_ce}), how='inner')

# plot correlation of drift and diffusion with the load for Australia and Continental Europe  
fig, ax = plt.subplots(2, 2, sharey='row')
fig.set_figwidth(10)
fig.set_figheight(10)
fig.suptitle(f'Correlation of Drift/Diffusion with Load', fontsize=s.plotting['title size'])

ax[0, 0].set_title(
    f'AUS: Drift with Load\ncorr = {round(aus['drift'].corr(aus['load']), 2)}',
    fontsize=s.plotting['subplot title size']
)
ax[0, 1].set_title(
    f'AUS: Diffusion with Load\ncorr = {round(aus['diffusion'].corr(aus['load']), 2)}',
    fontsize=s.plotting['subplot title size']
)
ax[1, 0].set_title(
    f'CE: Drift with Load\ncorr = {round(ce['drift'].corr(ce['load']), 2)}',
    fontsize=s.plotting['subplot title size']
)
ax[1, 1].set_title(
    f'CE: Diffusion with Load\ncorr = {round(ce['diffusion'].corr(ce['load']), 2)}',
    fontsize=s.plotting['subplot title size']
)

ax[0, 0].scatter(aus['drift'], aus['load'], color=s.plotting['color AUS']['detrended drift'])
ax[0, 1].scatter(aus['diffusion'], aus['load'], color=s.plotting['color AUS']['detrended diffusion'])
ax[1, 0].scatter(ce['drift'], ce['load'], color=s.plotting['color CE']['detrended drift'])
ax[1, 1].scatter(ce['diffusion'], ce['load'], color=s.plotting['color CE']['detrended diffusion'])

ax[0, 0].set_ylabel('AUS Load', fontsize=s.plotting['fontsize'])
ax[1, 0].set_ylabel('CE Load', fontsize=s.plotting['fontsize'])

ax[0, 0].set_xlabel(r'Drift $D_1(\omega)$', fontsize=s.plotting['fontsize'])
ax[0, 1].set_xlabel(r'Diffusion $D_2(\omega)$', fontsize=s.plotting['fontsize'])
ax[1, 0].set_xlabel(r'Drift $D_1(\omega)$', fontsize=s.plotting['fontsize'])
ax[1, 1].set_xlabel(r'Diffusion $D_2(\omega)$', fontsize=s.plotting['fontsize'])

plt.tight_layout()
plt.savefig(f'../results/utils/plots/correlation_load_with_drift_diffusion.png')
plt.savefig(f'../results/utils/plots/correlation_load_with_drift_diffusion.pdf')
