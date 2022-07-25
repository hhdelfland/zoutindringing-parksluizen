from unicodedata import numeric
import pandas as pd
import numpy as np
import telecontrol_parser as tp
import timeseries_functions as tf
from tsfresh import extract_relevant_features
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from itertools import repeat

class TimeseriesDataset:

    def __init__(self, dataset):
        dataset = dataset.set_index(pd.to_datetime(dataset['datetime']))
        self.dataset = dataset
        self.ycol = tp.egv_get_numeric_cols(dataset)[1]

    def fm_create_tsfresh(self):
        dataset = self.dataset
        numeric_cols = tp.egv_get_numeric_cols(dataset)
        feat_data = dataset[numeric_cols[1]].to_frame()
        feat_data['date'] = [dataset.index[i].date()
                             for i in range(len(dataset))]
        feat_data = feat_data.reset_index(drop=True)
        feat_data['index'] = feat_data.index
        feat_data = feat_data.rename(columns={numeric_cols[1]: 'Y'})
        feats = extract_features(feat_data, column_value='Y', column_id='date')
        print('Saving TSfresh features...')
        feats.to_excel('data_sets/TSfeats.xlsx')

    def fm_lag(self, lag, start=1):
        dataset = self.dataset
        numeric_cols = tp.egv_get_numeric_cols(dataset)
        for i in range(start, lag + 1):
            dataset['lag_' + str(i)] = dataset[numeric_cols[1]].shift(i)
        return self.dataset

    def fm_diff(self, lag, stepsize=1, start=1):
        dataset = self.dataset
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
        return self.dataset

    def fm_rolling(self, window_size, func='mean'):
        dataset = self.dataset
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
        return self.dataset

    def fm_time(self):
        dataset = self.dataset
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

        dataset['month'] = [
            dataset.index[i].month for i in range(len(dataset))]
        dataset['month_sin'] = np.sin(dataset['month']*(2.*np.pi/12))
        dataset['month_cos'] = np.cos(dataset['month']*(2.*np.pi/12))
        return self.dataset

    def fm_exec_func(self, func_name, arg_dict=None):
        if not(isinstance(arg_dict, type(None))):
            vals = list(arg_dict.values())
            length = len(vals[0])
            if all(len(item) == length for item in vals):
                for i in range(0, length):
                    my_args = {}
                    for key in arg_dict.keys():
                        my_args[key] = arg_dict[key][i]
                    print('Executing: ' + func_name.__name__ +
                          ' with args: ' + str(my_args))
                    func_name(**my_args)
        else:
            print('Executing: ' + func_name.__name__)
            func_name()

    def fm_save(self, format='csv'):
        dataset = self.dataset
        if format == 'csv':
            print('Saving to csv')
            dataset.to_csv('data_sets/feats.csv')
        if format == 'xlsx':
            print('Saving to xlsx')
            dataset.to_excel('data_sets/feats.xlsx')


def main():
    TIMESTEP_IN_HOUR = int(60/10)  # How many measurements in 1 hour
    TSData = TimeseriesDataset(tf.tsdf_read_subsets(3))

    TSData.fm_exec_func(TSData.fm_time)
    TSData.fm_exec_func(TSData.fm_diff,
                        arg_dict={'lag': (1, 2),
                                  'stepsize': (1, 1)})
    TSData.fm_exec_func(TSData.fm_lag,
                        arg_dict={'lag': (TIMESTEP_IN_HOUR * 24,)})

    rolling_funcs = ('mean', 'min', 'max', 'median','std')
    rolling_funcs = [x for item in rolling_funcs for x in repeat(item, 4)]
    rolling_args = (TIMESTEP_IN_HOUR, TIMESTEP_IN_HOUR*2,TIMESTEP_IN_HOUR*12, TIMESTEP_IN_HOUR*24)*5
    TSData.fm_exec_func(
        TSData.fm_rolling,
        arg_dict={'func': rolling_funcs, 'window_size': rolling_args})

    TSData.fm_save(format='xlsx')
    TSData.fm_create_tsfresh()


if __name__ == '__main__':
    main()
