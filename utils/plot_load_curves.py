import pandas as pd
from matplotlib import pyplot as plt
from utils import settings as s

# AUS load features
data_aus = pd.read_hdf(f'../results/prepared_features/AUS_original_ml.h5')
load_aus = data_aus[['Battery (Charging)', 'Battery (Discharging)', 'Biomass', 'Coal (Black)', 'Coal (Brown)',
                     'Distillate', 'Gas (CCGT)', 'Gas (OCGT)', 'Gas (Reciprocating)', 'Gas (Steam)',
                     'Gas (Coal Mine Waste)', 'Hydro', 'Pumps', 'Solar (Utility)', 'Solar (Thermal)',
                     'Wind', 'Nuclear', 'Biogas', 'Solar (Rooftop)']].sum(axis=1)

# CE load features:
data_ce = pd.read_hdf(f'../results/prepared_features/CE_original_ml.h5')
load_ce = data_ce['load']

# plot 7 day load curve for AUS and CE
fig, (ax_aus, ax_ce) = plt.subplots(2, 1)
fig.set_figwidth(10)
fig.set_figheight(5)
fig.suptitle('Load Curves', fontsize=s.plotting['title size'])

hours = 7 * 24
first_day = load_aus.index[0]
last_day = first_day + pd.Timedelta(hours - 1, 'h')

# offset to sync days of the year between AUS and CE
aus_days_ahead = 29 * 24

# offset AUS data from beginning because of missing values
offset = 56 * 24

ax_aus.plot(load_aus[load_aus.index[0] + pd.Timedelta(offset + aus_days_ahead, 'h'):
                     load_aus.index[0] + pd.Timedelta(offset + aus_days_ahead + hours - 1, 'h')],
            color=s.plotting['color AUS']['frequency'])
ax_ce.plot(load_ce[load_ce.index[0] + pd.Timedelta(offset, 'h'):
                   load_ce.index[0] + pd.Timedelta(offset + hours - 1, 'h')],
           color=s.plotting['color CE']['frequency'])

ax_aus.set_ylabel('AUS Load', fontsize=s.plotting['fontsize'])
ax_ce.set_ylabel('CE Load', fontsize=s.plotting['fontsize'])
ax_aus.set_xlabel('Time', fontsize=s.plotting['fontsize'])
ax_ce.set_xlabel('Time', fontsize=s.plotting['fontsize'])
fig.tight_layout()
plt.savefig(f'../results/utils/plots/load_curves.png')
plt.savefig(f'../results/utils/plots/load_curves.pdf')
