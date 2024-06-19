import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from utils import settings as s


def validate_AUS_solar():
    # The partial dependency plot of the Solar (Utility) and Solar (Rooftop) feature for Australia has a lot of values
    # at or close to 0 in its histogram. This file helps show if these are due to periodic changes throughout the day
    # (no Solar at night) or if there is a large interval of missing values and thus the feature should be removed.
    # it shows that most of the values are really around 0 and only include limited NaN values. The assumption of
    # periodical changes due to the day time are also correct. Thus, the feature can be kept in use.

    for feature in ['Solar (Utility)', 'Solar (Rooftop)']:
        data = pd.read_hdf(f'../results/prepared_features/AUS_original_ml.h5')
        solar_utility = data[feature]

        # replace NaN values with unique value otherwise not found in data to separate them from 0 values
        solar_utility.replace(np.nan, -1000, inplace=True)

        # plot histogram
        counts, bins = np.histogram(solar_utility, bins=20)
        fig = plt.figure()
        plt.stairs(counts, bins, color='goldenrod')
        plt.title(f'{feature} Histogram', fontsize=s.plotting['title size'])
        plt.xlabel(feature, fontsize=s.plotting['fontsize'])
        plt.ylabel('Occurrences', fontsize=s.plotting['fontsize'])
        # replace -1000 on x-axis with NaN
        plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: 'NaN' if x == -1000 else x))
        fig.tight_layout()
        plt.show()

        # plot feature over time
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.set_figwidth(30)
        fig.set_figheight(10)
        fig.suptitle(f'{feature} over time', fontsize=s.plotting['title size'])
        ax1.set_xlabel('Time', fontsize=s.plotting['fontsize'])
        ax2.set_xlabel('Time', fontsize=s.plotting['fontsize'])
        ax1.set_ylabel(feature, fontsize=s.plotting['fontsize'])
        ax2.set_ylabel(feature, fontsize=s.plotting['fontsize'])
        # first week has lots of NaN values, so skip those
        week_hours = 7*24
        ax1.plot(solar_utility[week_hours:2 * week_hours], color='goldenrod')
        ax2.plot(solar_utility[week_hours:5 * week_hours], color='goldenrod')
        # replace -1000 on y-axis with NaN
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, p: 'NaN' if y == -1000 else y))
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, p: 'NaN' if y == -1000 else y))
        fig.tight_layout()
        plt.show()


def check_random_original_AUS_features():
    # Print random row of feature values to cross-check on the website if data retrieval was without errors
    # AUS data archived on website: https://www.nemweb.com.au/REPORTS/ARCHIVE/

    # randomly checked value matched used data indicating no errors in data retrieval
    # however the Hydro values for AUS features are around half of what the daily total should be

    feature_data = pd.read_pickle('../data/[feature_data]/feature_data_AUS.pkl')
    feature_data.index = pd.to_datetime(feature_data.index)
    feature_data.rename(columns={'Hyrdo': 'Hydro'}, inplace=True)

    sample_row = feature_data.sample(n=1)
    print(sample_row)

    sample_day = feature_data.resample('d').sum().sample(n=1) / 12
    print(sample_day)


def validate_nan_counts():
    # Try to find if there is a large amount of NaN values in any specific feature.

    # AUS data showed that all features have 149 of 9317 hours during which all features are missing.
    # As they are the same hours across all features, these hours will be removed to improve the dataset.
    # CE data does not have such a distinct pattern in it's NaN values and will not be changed.

    for area in ['AUS', 'CE']:
        print(f'-----{area}-----')
        # original and detrended get same features
        data = pd.read_hdf(f'../results/prepared_features/{area}_original_ml.h5')
        feature_columns = [column for column in data.columns if column not in ['drift', 'diffusion']]

        for column in feature_columns:
            print(f'{round(data[column].isna().sum() * 100 / data[column].size, 2)}%, '
                  f'{data[column].isna().sum()}/{data[column].size}, {column}')


validate_AUS_solar()
validate_nan_counts()
check_random_original_AUS_features()
