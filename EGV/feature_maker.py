from ast import arg
from pickle import TRUE
from unicodedata import numeric
import pandas as pd
import numpy as np
import telecontrol_parser as tp
import timeseries_functions as tf
# from tsfresh import extract_features
# from tsfresh.feature_extraction import MinimalFCParameters
from itertools import repeat
import itertools


def fm_window_functions(col, ycol, window_size, func):
    if func == 'mean':
        col_val = col.mean()
    if func == 'min':
        col_val = col.min()
    if func == 'max':
        col_val = col.max()
    if func == 'sum':
        col_val = col.sum()
    if func == 'std':
        col_val = col.std()
    if func == 'median':
        col_val = col.quantile(0.5)
    if func == 'range':
        col_val = col.max()-col.min()

    return col_val


def fm_args_combiner(*args):
    arg_list = list(itertools.product(*args))
    arg_list = (list(zip(*arg_list)))
    return arg_list


def fm_standard_run(subset, path, save_path='', save=False, ycol=None, TIMESTEP_IN_HOUR=6, future_steps=6):
    tf_read, locatie = tf.tsdf_read_subsets(subset, path=path)
    TSData = TimeseriesDataset(tf_read, ycol)
    rolling_funcs = ('mean', 'min', 'max', 'median', 'std', 'sum', 'range')
    rolling_shifts = (0, 1*6, 11*6, 12*6, 23*6, 24*6)
    rolling_window_sizes = (
        TIMESTEP_IN_HOUR, TIMESTEP_IN_HOUR*2, TIMESTEP_IN_HOUR*12, TIMESTEP_IN_HOUR*24)
    # TIMESTEP_IN_HOUR*6, TIMESTEP_IN_HOUR*7, TIMESTEP_IN_HOUR *
    # 8, TIMESTEP_IN_HOUR*9, TIMESTEP_IN_HOUR*10,
    # TIMESTEP_IN_HOUR*11, TIMESTEP_IN_HOUR*12, TIMESTEP_IN_HOUR *
    # 13, TIMESTEP_IN_HOUR*14, TIMESTEP_IN_HOUR*15,
    # TIMESTEP_IN_HOUR*16, TIMESTEP_IN_HOUR*17, TIMESTEP_IN_HOUR *
    # 18, TIMESTEP_IN_HOUR*19, TIMESTEP_IN_HOUR*20,
    # TIMESTEP_IN_HOUR*21, TIMESTEP_IN_HOUR*22, TIMESTEP_IN_HOUR*23, TIMESTEP_IN_HOUR*24)

    # rolling_window_sizes1 = rolling_window_sizes*len(rolling_funcs)
    # rolling_funcs1 = [x for item in rolling_funcs for x in repeat(
    #     item, len(rolling_window_sizes))]

    arg_list = fm_args_combiner(
        rolling_funcs, rolling_shifts, rolling_window_sizes)
    rolling_funcs = arg_list[0]
    rolling_shifts = arg_list[1]
    rolling_windows = arg_list[2]

    TSData.fm_exec_func(TSData.fm_time)

    TSData.rename_ycol('EGV_OPP')

    TSData.fm_create_future_steps(future_steps)

    TSData.fm_exec_func(
        TSData.fm_diff,
        arg_dict={'lag': (1, 1),
                  'stepsize': (1, 2)})
    TSData.fm_exec_func(
        TSData.fm_lag,
        arg_dict={'lag': (TIMESTEP_IN_HOUR * 6,)})
    TSData.fm_exec_func(
        TSData.fm_shifted_rolling,
        arg_dict={'func': rolling_funcs, 'window_size': rolling_windows, 'shift': rolling_shifts})

    TSData.ycol = TSData.get_numeric_cols()[0]
    TSData.rename_ycol('TEMP_OPP')

    TSData.fm_exec_func(
        TSData.fm_diff,
        arg_dict={'lag': (1, 1),
                  'stepsize': (1, 2)})
    TSData.fm_exec_func(
        TSData.fm_lag,
        arg_dict={'lag': (TIMESTEP_IN_HOUR * 1,)})
    TSData.fm_exec_func(
        TSData.fm_shifted_rolling,
        arg_dict={'func': rolling_funcs, 'window_size': rolling_windows, 'shift': rolling_shifts})

    # TSData.fm_exec_func(
    #     TSData.fm_diff,
    #     arg_dict={'lag': (1, 2),
    #               'stepsize': (1, 1)})
    # TSData.fm_exec_func(
    #     TSData.fm_lag,
    #     arg_dict={'lag': (TIMESTEP_IN_HOUR * 2,)})
    # TSData.fm_exec_func(
    #     TSData.fm_rolling,
    #     arg_dict={'func': rolling_funcs1, 'window_size': rolling_args1})

    if save:
        TSData.fm_save(format='parquet', path=save_path, title=locatie)
        # TSData.fm_create_tsfresh()
    else:
        pd.set_option('display.max_columns', None)
        print(list(TSData.dataset.columns))


class TimeseriesDataset:

    def __init__(self, dataset, ycol=None):
        if type(dataset.index) != pd.core.indexes.datetimes.DatetimeIndex:
            dataset = dataset.set_index(pd.to_datetime(dataset['datetime']))
        self.dataset = dataset
        self.ycol = ycol
        if isinstance(self.ycol, type(None)):
            self.ycol = tp.egv_get_numeric_cols(dataset)[1]
        elif isinstance(self.ycol, type(int)):
            self.ycol = dataset[ycol]

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
            dataset[ycol + '_lag_' +
                    str(i)+'_start_'+str(start)] = dataset[ycol].shift(i)
        self.dataset = dataset
        return self

    def fm_diff(self, lag, stepsize=1, start=1):
        dataset = self.dataset
        ycol = self.ycol
        for i in range(start, lag + 1):
            if stepsize > 1:
                col_val = dataset[ycol].diff(i)
                for step in range(1, stepsize):
                    col_val = np.diff(col_val.values, prepend=np.nan)
                dataset[ycol + '_diff_order_' +
                        str(stepsize) + '_lag_' + str(i)] = col_val
            else:
                dataset[ycol + '_diff_order_' + str(stepsize) + '_lag_' + str(i)
                        ] = dataset[ycol].diff(i)
        self.dataset = dataset
        return self

    # def fm_rolling(self, window_size, func='mean'):
    #     dataset = self.dataset
    #     ycol = self.ycol
    #     col_roll = dataset[ycol].rolling(window_size)
    #     col_val = fm_window_functions(col_roll, ycol, window_size, func)
    #     dataset = pd.concat([dataset, col_val], axis=1)
    #     self.dataset = dataset
    #     return self

    def fm_shifted_rolling(self, window_size=6, shift=0, func='mean'):
        dataset = self.dataset
        ycol = self.ycol
        col = dataset[ycol].shift(shift).rolling(window_size)
        new_col = fm_window_functions(col, ycol, window_size, func)
        new_col.name = '_'.join([ycol, 'shifted', str(
            shift), 'roll', str(window_size), 'func', func])
        dataset = pd.concat([dataset, new_col], axis=1)
        self.dataset = dataset
        return self

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
        dataset.self = dataset
        return self

    def fm_exec_func(self, func_name, arg_dict=None):
        if not (isinstance(arg_dict, type(None))):
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

    def fm_create_future_steps(self, steps, stride=1):
        for i in range(1, steps+1):
            self.dataset[self.ycol + '_(t+' + str(i*stride)+')'] = \
                self.dataset[self.ycol].shift(-i*stride)

    def fm_get_streak(self):
        dataset = self.dataset
        ycol = self.ycol
        data = dataset
        emptydb = pd.DataFrame()
        emptydb['datetime'] = data['datetime']
        emptydb['is_active'] = np.where(data[ycol] > 0, 1, 0)
        emptydb['start'] = emptydb['is_active'].ne(
            emptydb['is_active'].shift())
        emptydb['id'] = emptydb['start'].cumsum()
        emptydb['count'] = emptydb.groupby('id').cumcount() + 1
        emptydb['count_on'] = np.where(
            emptydb['is_active'] == 1, emptydb['count'], 0)
        emptydb['count_off'] = np.where(
            emptydb['is_active'] == 0, emptydb['count'], 0)
        emptydb.set_index('datetime', drop=True)
        kept_cols = ['is_active', 'count_on', 'count_off']
        emptydb = emptydb[kept_cols]
        dataset = pd.concat([dataset, emptydb], axis=1)
        self.dataset = dataset
        self.emptydb = emptydb
        return self

    def fm_save(self, path='', format='csv', title=None):
        dataset = self.dataset
        start = str(dataset.index[0])[:10]
        end = str(dataset.index[-1])[:10]
        size = len(dataset)

        for column in [
            'Unnamed: 0',
            'Datum',
            'Tijd (Europe/Amsterdam)'
        ]:
            if column in dataset.columns:
                print(f"Dropping column: {column}")
                dataset = dataset.drop(column, axis=1)

        if path == '':
            fname = f'data_sets/feats/feats_{start}_{end}_{size}'
        else:
            if title is not None:
                fname = f'{path}/feats_{title}'
            else:
                fname = f'{path}/feats_{start}_{end}_{size}'

        if format == 'csv':
            print('Saving to csv')
            dataset.to_csv(fname + '.csv')
        elif format == 'xlsx':
            print('Saving to xlsx')
            dataset.to_excel(fname + '.xlsx')
        elif format == 'parquet':
            print('Saving to parquet')
            dataset.to_parquet(fname + '.parquet', engine='pyarrow')
        else:
            raise ValueError(
                "Format not supported. Should be \
                either 'csv','xlsx' or 'parquet'.")


def main():
    import getpass
    save = True
    TIMESTEP_IN_HOUR = int(60/10)  # How many measurements in 1 hour
    path = fr'C:\Users\{getpass.getuser()}\OneDrive - Hoogheemraadschap van Delfland\3_Projecten\Zoutindringing\Data\datadumps\EGV_parsed\\'
    save_path = fr"C:\Users\{getpass.getuser()}\OneDrive - Hoogheemraadschap van Delfland\3_Projecten\Zoutindringing\Data\features\egv_feats_2"
    fm_standard_run(subset=0, path=path, save=save,
                    save_path=save_path, future_steps=6*36)
    fm_standard_run(subset=1, path=path, save=save,
                    save_path=save_path, future_steps=6*36)
    fm_standard_run(subset=2, path=path, save=save,
                    save_path=save_path, future_steps=6*36)
    fm_standard_run(subset=3, path=path, save=save,
                    save_path=save_path, future_steps=6*36)


if __name__ == '__main__':
    main()
