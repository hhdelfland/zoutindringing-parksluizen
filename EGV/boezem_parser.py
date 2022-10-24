import pandas as pd
import numpy as np
import os
import timeseries_functions as tf
import feature_maker as fm

def define_locations():
    location_dict = {
        'OW062C000' : 'beukelsbrug',
        'OW062E001' : 'lage_erf_brug',
        'OW095-000' : 'coolhaven'
    }
    return location_dict
    

def parse_boezem_egv(boezem_egv_path):
    files = os.listdir(boezem_egv_path)
    dfs = {}
    for file in files:
        print(file)
        db = pd.read_csv(boezem_egv_path + file,sep = r'\t',encoding = 'utf-8')
        db = db.drop(db.tail(5).index)
        db = db.replace('"',' ',regex = True)

        locs = define_locations()
        # loc = locs[db.iloc[1,0][1:-1]]
        loc = (db.columns[2][1:10])
        cols = list(db.columns)
        col = [s for s in cols if s.endswith('[mS/cm]"')]
        db1 = db.iloc[:,2:]
        # db1 = db1.drop(db1.tail(6).index)
        db1 = db1.replace(',','.',regex = True)
        db1 = db1.apply(pd.to_numeric,errors = 'coerce')
        datetime = pd.to_datetime(db.iloc[:,0] + db.iloc[:,1].str.strip() + ':00',format="%Y/%m/%d %H:%M:%S")
        db1 = db1.set_index(datetime)
        db1['datetime'] = datetime
        print(tf.tsdf_report_gaps(db1,1,0))
        db1 = tf.tsdf_interpolate_small_gaps(db1,9,0)
        dfs[loc] = db1
    return dfs

def check_gaps(dfs):
    for i in list(dfs.keys()):
        print(define_locations()[i])
        print(tf.tsdf_report_gaps(dfs[i],1,0))
    
def set_y(dfs,location,column,timesteps,stride = 1):
    locs = define_locations()
    for i in locs.keys():
        if locs[i] == location:
            dat = fm.TimeseriesDataset(dfs[i], ycol = dfs[i].columns[column])
    if timesteps > 0:
        dat.rename_ycol(location)
        dat.fm_create_future_steps(timesteps, stride = stride)
    dat.location = location
    return dat
    
def featurize_boezem(TSData,TIMESTEP_IN_HOUR = 6):
    location = TSData.location
    if location == 'beukelsbrug':
        rolling_funcs = ('mean', 'min', 'max', 'median', 'std', 'sum','range')
        rolling_shifts = (0, 1*6, 11*6, 12*6, 23*6, 24*6)
        rolling_window_sizes = (TIMESTEP_IN_HOUR, TIMESTEP_IN_HOUR*2, TIMESTEP_IN_HOUR*12, TIMESTEP_IN_HOUR*24)
        arg_list = fm.fm_args_combiner(
            rolling_funcs, rolling_shifts, rolling_window_sizes)
        rolling_funcs = arg_list[0]
        rolling_shifts = arg_list[1]
        rolling_windows = arg_list[2]
        TSData.rename_ycol(location + '_EGV_OPP')
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
    if location == 'lage_erf_brug':
        rolling_funcs = ('mean', 'min', 'max', 'median', 'std', 'sum','range')
        rolling_shifts = (0, 1*6, 11*6, 12*6, 23*6, 24*6)
        rolling_window_sizes = (TIMESTEP_IN_HOUR, TIMESTEP_IN_HOUR*2, TIMESTEP_IN_HOUR*12, TIMESTEP_IN_HOUR*24)
        arg_list = fm.fm_args_combiner(
            rolling_funcs, rolling_shifts, rolling_window_sizes)
        rolling_funcs = arg_list[0]
        rolling_shifts = arg_list[1]
        rolling_windows = arg_list[2]
        TSData.rename_ycol(location + '_EGV_OPP')
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
    if location == 'coolhaven':
        rolling_funcs = ('mean', 'min', 'max', 'median', 'std', 'sum','range')
        rolling_shifts = (0, 1*6, 11*6, 12*6, 23*6, 24*6)
        rolling_window_sizes = (TIMESTEP_IN_HOUR, TIMESTEP_IN_HOUR*2, TIMESTEP_IN_HOUR*12, TIMESTEP_IN_HOUR*24)
        arg_list = fm.fm_args_combiner(
            rolling_funcs, rolling_shifts, rolling_window_sizes)
        rolling_funcs = arg_list[0]
        rolling_shifts = arg_list[1]
        rolling_windows = arg_list[2]
        TSData.rename_ycol(location + '_EGV_OPP')
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

    
    
    TSData.fm_save(path = 'E:/Rprojects/zoutindringing-parksluizen/data_sets_boezem/features/'+location, format = 'parquet')

def main():
    dfs = (parse_boezem_egv('E:/Rprojects/zoutindringing-parksluizen/data_sets_boezem/EGV/'))
    print(dfs)
    check_gaps(dfs)
    TSD = set_y(dfs,'beukelsbrug',0,120, stride = 6)
    featurize_boezem(TSD)
    TSD = set_y(dfs,'lage_erf_brug',0,0)
    featurize_boezem(TSD)
    TSD = set_y(dfs,'coolhaven',0,0)
    featurize_boezem(TSD)

if __name__ == '__main__':
    main()
