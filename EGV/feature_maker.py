from unicodedata import numeric
import pandas as pd
import numpy as np
import telecontrol_parser as tp
import timeseries_functions as tf
from tsfresh import extract_relevant_features
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from itertools import repeat


def fm_standard_run(subset, save=False, TIMESTEP_IN_HOUR=6, future_steps = 6):
    TSData = TimeseriesDataset(tf.tsdf_read_subsets(subset))
    rolling_funcs = ('mean', 'min', 'max', 'median', 'std', 'sum')
    rolling_args = (TIMESTEP_IN_HOUR, TIMESTEP_IN_HOUR*2,
                    TIMESTEP_IN_HOUR*12, TIMESTEP_IN_HOUR*24)

    rolling_args1 = rolling_args*len(rolling_funcs)
    rolling_funcs1 = [x for item in rolling_funcs for x in repeat(
        item, len(rolling_args))]

    TSData.fm_exec_func(TSData.fm_time)

    TSData.rename_ycol('EGV_OPP')

    TSData.fm_create_future_steps(future_steps)

    TSData.fm_exec_func(
        TSData.fm_diff,
        arg_dict={'lag': (1, 2),
                  'stepsize': (1, 1)})
    TSData.fm_exec_func(
        TSData.fm_lag,
        arg_dict={'lag': (TIMESTEP_IN_HOUR * 24,)})
    TSData.fm_exec_func(
        TSData.fm_rolling,
        arg_dict={'func': rolling_funcs1, 'window_size': rolling_args1})

    TSData.ycol = TSData.get_numeric_cols()[0]
    TSData.rename_ycol('TEMP_OPP')

    TSData.fm_exec_func(
        TSData.fm_diff,
        arg_dict={'lag': (1, 2),
                  'stepsize': (1, 1)})
    TSData.fm_exec_func(
        TSData.fm_lag,
        arg_dict={'lag': (TIMESTEP_IN_HOUR * 24,)})
    TSData.fm_exec_func(
        TSData.fm_rolling,
        arg_dict={'func': rolling_funcs1, 'window_size': rolling_args1})

    if save:
        TSData.fm_save(format='csv')
        #TSData.fm_create_tsfresh()


class TimeseriesDataset:

    def __init__(self, dataset):
        dataset = dataset.set_index(pd.to_datetime(dataset['datetime']))
        self.dataset = dataset
        self.ycol = tp.egv_get_numeric_cols(dataset)[1]

    def get_numeric_cols(self):
        return tp.egv_get_numeric_cols(self.dataset)

    def rename_ycol(self, newname):
        self.dataset = self.dataset.rename(columns={self.ycol: newname})
        self.ycol = newname

    def fm_create_tsfresh(self):
        dataset = self.dataset
        ycol = self.ycol
        feat_data = dataset[ycol].to_frame()
        feat_data['date'] = [dataset.index[i].date()
                             for i in range(len(dataset))]
        feat_data = feat_data.reset_index(drop=True)
        feat_data['index'] = feat_data.index
        feat_data = feat_data.rename(columns={ycol: 'Y'})
        feats = extract_features(feat_data, column_value='Y', column_id='date')
        start = str(dataset.index[0])[:10]
        end = str(dataset.index[-1])[:10]
        size = len(dataset)
        print('Saving TSfresh features...')
        feats.to_excel(f'data_sets/TSfeats_{size}.xlsx')

    def fm_lag(self, lag, start=1):
        dataset = self.dataset
        ycol = self.ycol
        for i in range(start, lag + 1):
            dataset[ycol + '_lag_' + str(i)] = dataset[ycol].shift(i)
        return self.dataset

    def fm_diff(self, lag, stepsize=1, start=1):
        dataset = self.dataset
        ycol = self.ycol
        for i in range(start, lag + 1):
            if stepsize > 1:
                col_val = dataset[ycol].diff(i)
                for step in range(1, stepsize):
                    col_val = np.diff(col_val.values, prepend=np.nan)
                dataset[ycol + '_diff_' +
                        str(stepsize) + '_' + str(i)] = col_val
            else:
                dataset[ycol + '_diff_' + str(stepsize) + '_' + str(i)
                        ] = dataset[ycol].diff(i)
        return self.dataset

    def fm_rolling(self, window_size, func='mean'):
        dataset = self.dataset
        ycol = self.ycol
        col_roll = dataset[ycol].rolling(window_size)
        if func == 'mean':
            col_val = col_roll.mean()
            col_val.name = ycol + '_mean_' + str(window_size)
        if func == 'min':
            col_val = col_roll.min()
            col_val.name = ycol + '_min_' + str(window_size)
        if func == 'max':
            col_val = col_roll.max()
            col_val.name = ycol + '_max_' + str(window_size)
        if func == 'sum':
            col_val = col_roll.sum()
            col_val.name = ycol + '_sum_' + str(window_size)
        if func == 'std':
            col_val = col_roll.std()
            col_val.name = ycol + '_std_' + str(window_size)
        if func == 'median':
            col_val = col_roll.quantile(0.5)
            col_val.name = ycol + '_median_' + str(window_size)
        dataset = pd.concat([dataset, col_val], axis=1)
        return self.dataset

    def fm_time(self):
        dataset = self.dataset
        ycol = self.ycol
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

    def fm_create_future_steps(self, steps):
        for i in range(1, steps+1):
            self.dataset[self.ycol + '_(t+' + str(i)+')'] = \
                self.dataset[self.ycol].shift(-i)

    def fm_save(self, format='csv'):
        dataset = self.dataset
        start = str(dataset.index[0])[:10]
        end = str(dataset.index[-1])[:10]
        size = len(dataset)
        if format == 'csv':
            print('Saving to csv')
            dataset.to_csv(f'data_sets/feats_{size}.csv')
        elif format == 'xlsx':
            print('Saving to xlsx')
            dataset.to_excel(f'data_sets/feats_{size}.xlsx')
        else:
            raise ValueError(
                "Format not supported. Should be either 'csv' or 'xlsx'.")




def main():
    save = True
    TIMESTEP_IN_HOUR = int(60/10)  # How many measurements in 1 hour
    fm_standard_run(subset = 0, save = save, future_steps = 36)
    fm_standard_run(subset = 1, save = save, future_steps = 36)
    fm_standard_run(subset = 2, save = save, future_steps = 36)
    fm_standard_run(subset = 3, save = save, future_steps = 36)

if __name__ == '__main__':
    main()
