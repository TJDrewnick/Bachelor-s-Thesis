import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

from utils import settings as s


# taken from https://developer.nvidia.com/blog/three-approaches-to-encoding-time-information-as-features-for-ml-models/
# transform time in sine and cosine values for cyclical continuity
def sin_transformer(period: int) -> FunctionTransformer:
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period: int) -> FunctionTransformer:
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))


def get_feature_data(country: str) -> pd.DataFrame:
    if country == 'AUS':
        feature_data = pd.read_pickle('../data/[feature_data]/feature_data_AUS.pkl')
        feature_data.index = pd.to_datetime(feature_data.index)
        feature_data.rename(columns={'Hyrdo': 'Hydro'}, inplace=True)

    elif country == 'CE':
        input_actual = pd.read_hdf('../data/[feature_data]/CE/input_actual.h5')
        input_forecast = pd.read_hdf('../data/[feature_data]/CE/input_forecast.h5')
        feature_data = pd.concat([input_actual, input_forecast], axis=1)
        # align with frequency data timestamps
        feature_data.index = feature_data.index.tz_localize(None) - pd.Timedelta(hours=2)
        feature_data.drop(columns=['month', 'weekday', 'hour'], inplace=True)

    else:
        print('Invalid country')
        return pd.DataFrame()

    feature_data = feature_data.resample('h').mean()
    return feature_data


def add_time_features(feature_data: pd.DataFrame) -> pd.DataFrame:
    feature_data['month'] = feature_data.index.month
    feature_data['month_sin'] = sin_transformer(12).fit_transform(feature_data)['month']
    feature_data['month_cos'] = cos_transformer(12).fit_transform(feature_data)['month']

    feature_data['weekday'] = feature_data.index.weekday
    feature_data['weekday_sin'] = sin_transformer(7).fit_transform(feature_data)['weekday']
    feature_data['weekday_cos'] = cos_transformer(7).fit_transform(feature_data)['weekday']

    feature_data['hour'] = feature_data.index.hour
    feature_data['hour_sin'] = sin_transformer(24).fit_transform(feature_data)['hour']
    feature_data['hour_cos'] = cos_transformer(24).fit_transform(feature_data)['hour']

    feature_data.drop(columns=['month', 'weekday', 'hour'], inplace=True)
    return feature_data


def filter_outliers(feature_data: pd.DataFrame, outlier_percent: float) -> pd.DataFrame:

    for param in ['drift', 'diffusion']:
        lower = np.percentile(feature_data[param], outlier_percent)
        upper = np.percentile(feature_data[param], 100 - outlier_percent)
        feature_data = feature_data[(feature_data[param] >= lower) & (feature_data[param] <= upper)]

    return feature_data


for area in ['AUS', 'CE']:

    features = get_feature_data(area)
    features = add_time_features(features)

    # analyzing the feature data shows, that AUS features all have the same 149 hours missing, so they are removed
    if area == 'AUS':
        features.dropna(subset=['Wind'], inplace=True)

    # add drift and diffusion to features to create one file for each model
    for datatype in ['detrended', 'original']:

        targets = pd.read_hdf(f'../results/km/{area}_{datatype}_drift_diffusion.h5')

        combined = features.join(targets, how='inner')
        combined = filter_outliers(combined, s.settings[area]['outlier percent'])
        combined.to_hdf(f'../results/prepared_features/{area}_{datatype}_ml.h5', key='df', mode='w')
