from unicodedata import numeric
import pandas as pd
import numpy as np
import telecontrol_parser as tp
import timeseries_functions as tf
from tsfresh import extract_relevant_features
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters


def main():
    dataset = tf.tsdf_read_subsets(3)
    dataset = dataset.set_index(pd.to_datetime(dataset['datetime']))
    # create time features
    dataset = fm_time(dataset)
    # create lag features
    dataset = fm_lag(dataset, 6*24)
    # create first and second order differences as features
    dataset = fm_diff(dataset, 6*24, 2)
    dataset = fm_diff(dataset, 6*24, 1)

    # create rolling statistics 24 hours
    dataset = fm_rolling(dataset, 6*24, 'mean')
    dataset = fm_rolling(dataset, 6*24, 'min')
    dataset = fm_rolling(dataset, 6*24, 'max')
    dataset = fm_rolling(dataset, 6*24, 'median')
    dataset = fm_rolling(dataset, 6*24, 'std')
    # 1 hour
    dataset = fm_rolling(dataset, 6, 'mean')
    dataset = fm_rolling(dataset, 6, 'min')
    dataset = fm_rolling(dataset, 6, 'max')
    dataset = fm_rolling(dataset, 6, 'median')
    dataset = fm_rolling(dataset, 6, 'std')
    # 2 hours
    dataset = fm_rolling(dataset, 6*2, 'mean')
    dataset = fm_rolling(dataset, 6*2, 'min')
    dataset = fm_rolling(dataset, 6*2, 'max')
    dataset = fm_rolling(dataset, 6*2, 'median')
    dataset = fm_rolling(dataset, 6*2, 'std')
    # 12 hours
    dataset = fm_rolling(dataset, 6*12, 'mean')
    dataset = fm_rolling(dataset, 6*12, 'min')
    dataset = fm_rolling(dataset, 6*12, 'max')
    dataset = fm_rolling(dataset, 6*12, 'median')
    dataset = fm_rolling(dataset, 6*12, 'std')

    dataset.to_csv('data_sets/feats.csv')

    # fm_tsfresh(dataset)


def fm_tsfresh(dataset):
    numeric_cols = tp.egv_get_numeric_cols(dataset)
    feat_data = dataset[numeric_cols[1]].to_frame()
    feat_data['date'] = [dataset.index[i].date() for i in range(len(dataset))]
    feat_data = feat_data.reset_index(drop=True)
    feat_data['index'] = feat_data.index
    feat_data = feat_data.rename(columns={numeric_cols[1]: 'Y'})
    feats = extract_features(feat_data, column_value='Y', column_id='date')
    feats.to_excel('data_sets/TSfeats.xlsx')


def fm_lag(dataset, lag, start=1):
    numeric_cols = tp.egv_get_numeric_cols(dataset)
    for i in range(start, lag + 1):
        dataset['lag_' + str(i)] = dataset[numeric_cols[1]].shift(i)
    return dataset


def fm_diff(dataset, lag, stepsize=1, start=1):
    numeric_cols = tp.egv_get_numeric_cols(dataset)
    for i in range(start, lag + 1):
        if stepsize > 1:
            col_val = dataset[numeric_cols[1]].diff(i)
            for step in range(1, stepsize):
                col_val = np.diff(col_val.values, prepend=np.nan)
            dataset['diff_' + str(stepsize) + '_' + str(i)] = col_val
        else:
            dataset['diff_' + str(stepsize) + '_' + str(i)
                    ] = dataset[numeric_cols[1]].diff(i)
    return dataset


def fm_rolling(dataset, window_size, func='mean'):
    numeric_cols = tp.egv_get_numeric_cols(dataset)
    col_roll = dataset[numeric_cols[1]].rolling(window_size)
    if func == 'mean':
        col_val = col_roll.mean()
        col_val.name = 'mean_' + str(window_size)
    if func == 'min':
        col_val = col_roll.min()
        col_val.name = 'min_' + str(window_size)
    if func == 'max':
        col_val = col_roll.max()
        col_val.name = 'max_' + str(window_size)
    if func == 'sum':
        col_val = col_roll.sum()
        col_val.name = 'sum_' + str(window_size)
    if func == 'std':
        col_val = col_roll.std()
        col_val.name = 'std_' + str(window_size)
    if func == 'median':
        col_val = col_roll.quantile(0.5)
        col_val.name = 'median_' + str(window_size)
    dataset = pd.concat([dataset, col_val], axis=1)
    return dataset


def fm_time(dataset):
    dataset['hour'] = [dataset.index[i].hour for i in range(len(dataset))]
    dataset['hour_sin'] = np.sin(dataset['hour']*(2.*np.pi/23))
    dataset['hour_cos'] = np.cos(dataset['hour']*(2.*np.pi/23))

    dataset['weekday'] = [dataset.index[i].weekday()
                          for i in range(len(dataset))]
    dataset['weekday_sin'] = np.sin(dataset['weekday']*(2.*np.pi/6))
    dataset['weekday_cos'] = np.cos(dataset['weekday']*(2.*np.pi/6))

    dataset['monthday'] = [
        dataset.index[i].days_in_month for i in range(len(dataset))]
    dataset['monthday_sin'] = np.sin(dataset['monthday']*(2.*np.pi/31))
    dataset['monthday_cos'] = np.cos(dataset['monthday']*(2.*np.pi/31))

    dataset['month'] = [dataset.index[i].month for i in range(len(dataset))]
    dataset['month_sin'] = np.sin(dataset['month']*(2.*np.pi/12))
    dataset['month_cos'] = np.cos(dataset['month']*(2.*np.pi/12))
    return dataset


if __name__ == '__main__':
    main()
