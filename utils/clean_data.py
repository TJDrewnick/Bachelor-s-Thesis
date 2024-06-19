# Based on data cleaning procedures in:
# https://github.com/thlonsaker/predictability-of-island-power-grids/blob/master/scripts/clean_corrupt_data.py

import os
import numpy as np
import pandas as pd

settings = {
    # Maximum length of NaN values to fill
    'NaN fill': 6,
    # Maximum allowed length of constant windows
    'constant windows': 15,
    # Minimum height of isolated peaks
    'isolated peaks': 0.05
}


# retrieve and combine all data that should be cleaned
def load_data() -> pd.DataFrame:
    # get 2023 frequency data
    df_2023 = pd.read_excel('../data/AUS_2023_2024/AUS_2023_frequency_data.xlsb', sheet_name="Frequency data")
    df_2023 = pd.lreshape(df_2023, {'frequency': [column for column in df_2023.columns if column not in ['Time']]})
    df_2023['timestamp'] = pd.date_range(start='2023-01-01 00:00:00', periods=len(df_2023), freq='4s')
    df_2023.drop(columns='Time', inplace=True)

    # get weekly 2024 frequency data
    files = os.listdir('../data/AUS_2023_2024')
    files = [f for f in files if f[:23] == 'Weekly Frequency Report']

    dataframes = []

    for filename in files:
        file = pd.read_excel(f'../data/AUS_2023_2024/{filename}', sheet_name='FrequencyData', engine='pyxlsb')
        frequency = file['Unnamed: 1'][2:-1]
        dataframes.append(frequency)

    df_2024 = pd.DataFrame({'frequency': pd.concat(dataframes, ignore_index=True).astype(float)})
    df_2024['timestamp'] = pd.date_range(start='2024-01-01 00:00:00', periods=len(df_2024), freq='4s')

    # combine 2023 and 2024 frequency data
    df = pd.concat([df_2023, df_2024], ignore_index=True)

    return df


# Load data
frequency_data = load_data()

# find isolated peaks in data (absolute increments > settings['isolated peaks'])
difference = frequency_data['frequency'].diff()
high_increments = difference.where(difference.abs() > settings['isolated peaks'])
peak_locations = np.argwhere((high_increments * high_increments.shift(-1) < 0).values)[:, 0]
print(f'Number of peak locations: {peak_locations.size}')

# find constant windows in data that are longer than settings['constant windows']
mask = np.concatenate([[True], (difference.abs() >= 1e-9), [True]])
window_bounds = np.flatnonzero(mask[1:] != mask[:-1]).reshape(-1, 2)
window_sizes = window_bounds[:, 1] - window_bounds[:, 0]

long_windows = [[]]
long_window_bounds = window_bounds[window_sizes > settings['constant windows']]

if long_window_bounds.size != 0:
    long_windows = np.hstack([np.r_[i:j] for i, j in long_window_bounds])

print(f'Number of windows with constant frequency for longer than {settings['constant windows']} measurements:')
print(long_window_bounds.shape[0])

# replace corrupt data with NaN
frequency_data.iloc[peak_locations, frequency_data.columns.get_loc('frequency')] = np.nan
frequency_data.iloc[long_windows, frequency_data.columns.get_loc('frequency')] = np.nan

# forward fill NaN values up the specified limit
frequency_data.ffill(inplace=True, limit=settings['NaN fill'])

# remove all rows containing NaN values from dataframe
frequency_data.dropna(how='any', inplace=True)

# store dataframe to file
frequency_data.to_hdf(f'../data/AUS_cleansed_frequency_data.h5', key='df', mode='w')

# Here only AUS data is cleaned. The available data has no extreme points which is why no cleaning procedure for those
# was provided. All large increments and long constant windows are being cleaned and forward filled for up to 6 values.
# Resulting gaps are deleted from the data.
