import pandas as pd
import numpy as np
import h5py

from utils import settings as s


def get_frequency_data(country: str) -> pd.DataFrame:
    """
    Returns frequency data for a given country without NaN values and only those hours
    that have no missing measurements
    """
    if country == 'AUS':
        df = pd.read_hdf('../data/AUS_cleansed_frequency_data.h5')

    elif country == 'CE':
        ce_data = h5py.File('../data/CE_cleansed_2015-01-01_to_2019-12-31.h5', 'r')['df']
        df = pd.DataFrame({'timestamp': ce_data['index'], 'frequency': ce_data['values']})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.dropna(how='any', inplace=True)
        # remove data before dip in drift dip at 2017-01-25 04:00:00,
        # but some missing data until 2017-03-13 00:00:00
        df = df[df['timestamp'] >= '2017-03-13 00:00:00']

    else:
        print('Invalid country')
        return pd.DataFrame()

    # add datetime rounded down to hour to each datapoint
    df['hour'] = df['timestamp'].dt.floor('h')

    # Group by date and hour and filter out hours that don't have enough data points
    hourly_amount = df.groupby('hour').size()
    uninterrupted_hours_index = hourly_amount[hourly_amount == s.settings[country]['values per hour']].index
    df = df[df['hour'].isin(uninterrupted_hours_index)]

    return df


def to_angular_freq(df: pd.DataFrame, country: str) -> pd.DataFrame:
    df['frequency'] = 2 * np.pi * (df['frequency'] - s.settings[country]['reference frequency'])

    return df
