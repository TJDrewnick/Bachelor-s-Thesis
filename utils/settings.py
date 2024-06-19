# This file holds the settings for customizing each script. As some settings apply to multiple scripts,
# they are stored here instead

import matplotlib as mpl
import pandas as pd

########################
### GENERAL SETTINGS ###
########################

settings = {
    'AUS': {
        'name': 'Australia',           # full name of the area
        'reference frequency': 50,     # reference frequency of the area
        'values per hour': 900,        # number of values of the time series for a full hour
        'detrend sigma': 15,           # gaussian filter sigma used for detrending
        'outlier percent': 1           # percent of outliers that is filtered out
    },
    'CE': {
        'name': 'Continental Europe',  # full name of the area
        'reference frequency': 50,     # reference frequency of the area
        'values per hour': 3600,       # number of values of the time series for a full hour
        'detrend sigma': 60,           # gaussian filter sigma used for detrending
        'outlier percent': 2           # percent of outliers that is filtered out
    }
}

# display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)


########################
##### KRAMERS MOYAL ####
########################

km = {
    'AUS': {
        'drift bw': 0.05,      # Drift bandwidth
        'diffusion bw': 0.05,  # Diffusion bandwidth
        'delta_t': 4           # Resolution of time series in seconds
    },
    'CE': {
        'drift bw': 0.05,      # Drift bandwidth
        'diffusion bw': 0.05,  # Diffusion bandwidth
        'delta_t': 1           # Resolution of time series in seconds
    }
}


########################
### MACHINE LEARNING ###
########################

ml = {
    'random noise': False,                         # Add random noise feature to models
    'knockout': False,                             # Knockout the most important feature (config below)
    'test size': 0.2,                              # Test size for Machine Learning models
    'random search gbt_lgb': False,                # Perform random search: Gradient Boosted Trees, LightGBM
    'random search gbt_xgb_squarederror': False,   # Perform random search: Gradient Boosted Trees, XGBoost, Squared Error
    'random search gbt_xgb_absoluteerror': False,  # Perform random search: Gradient Boosted Trees, XGBoost, Absolute Error
    'random search rf_lgb': False,                 # Perform random search: Random Forest, LightGBM
    'grid search mlp': False,                      # Perform grid search: Multi Layer Perceptron
    'random search iterations': 1000,              # number of iterations for the random search
}

top_features = {
    'AUS': {
        'drift': {
            'gbt_lgb': 'hour_sin',
            'gbt_xgb_squarederror': 'hour_sin',
            'gbt_xgb_absoluteerror': 'hour_sin',
            'rf_lgb': 'hour_sin',
            'mlp': 'hour_sin'  # feature importance not implemented but most common upon other models chosen
        },
        'diffusion': {
            'gbt_lgb': 'hour_cos',
            'gbt_xgb_squarederror': 'hour_cos',
            'gbt_xgb_absoluteerror': 'hour_cos',
            'rf_lgb': 'hour_cos',
            'mlp': 'hour_cos'  # feature importance not implemented but most common upon other models chosen
        }
    },
    'CE': {
        'drift': {
            'gbt_lgb': 'load',
            'gbt_xgb_squarederror': 'load',
            'gbt_xgb_absoluteerror': 'total_gen',
            'rf_lgb': 'total_gen',
            'mlp': 'total_gen'  # feature importance not implemented but most common upon other models chosen
        },
        'diffusion': {
            'gbt_lgb': 'gen_nuclear',
            'gbt_xgb_squarederror': 'hour_cos',
            'gbt_xgb_absoluteerror': 'gen_nuclear',
            'rf_lgb': 'gen_nuclear',
            'mlp': 'gen_nuclear'  # feature importance not implemented but most common upon other models chosen
        }
    }
}


########################
####### PLOTTING #######
########################

# this colormap has 20 different colors, grouped in blocks of 2 that have the same base color
colormap = mpl.colormaps['tab20']

plotting = {
    'font':  'Times New Roman',
    'fontsize': 16,
    'subplot title size': 18,
    'title size': 20,
    'color AUS': {
        'frequency': colormap(14),               # grey
        'detrended drift': colormap(4),          # green
        'detrended diffusion': colormap(2),      # orange
        'original drift': colormap(5),           # light green
        'original diffusion': colormap(3),       # light orange
    },
    'color CE': {
        'frequency': colormap(10),               # brown
        'detrended drift': colormap(0),          # blue
        'detrended diffusion': colormap(6),      # red
        'original drift': colormap(1),           # light blue
        'original diffusion': colormap(7),       # light red
    },
    'color_gbt_lgb': colormap(12),               # pink
    'color_gbt_xgb_squarederror': colormap(8),   # purple
    'color_gbt_xgb_absoluteerror': colormap(9),  # light purple
    'color_rf_lgb': colormap(18),                # cyan
    'color_mlp': colormap(19),                   # light cyan

    # specific plotting settings for script 02
    'plotting type': 'average and frequency',    # possible options 'average', 'sequential', 'average and frequency'
    #                 average:                     avg over all time intervals of the given length
    #                 sequential:                  plot the intervals on top of each other without averaging
    #                 average and frequency:       plot the avg of the drift, diffusion and avg of the frequency
    'sequential limit': 2,                       # max number of sequential intervals plotted
    'rolling average': True                      # adds the rolling average to the plot
}

# set font
mpl.rcParams['font.family'] = plotting['font']
mpl.rc('legend', fontsize=plotting['fontsize'])

