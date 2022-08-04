import pandas as pd
import plotly
import matplotlib as mpl
import numpy as np
import os
import feature_maker as fm
from itertools import repeat
import timeseries_functions as tf

def parse_lobith(path):
    lobith_dbs = {}
    for file in os.listdir(path):
        raw_db = pd.read_csv(path + file,sep = ';')
        raw_db['NUMERIEKEWAARDE'] =  raw_db['NUMERIEKEWAARDE'].str.replace(',', '.').astype(float)
        db = pd.DataFrame(raw_db['NUMERIEKEWAARDE'])
        db['datetime'] = pd.to_datetime(raw_db['WAARNEMINGDATUM'] + ' ' + raw_db['WAARNEMINGTIJD (MET/CET)'],format='%d-%m-%Y %H:%M:%S')
        db = db.set_index('datetime',drop=False)
        db = db[~db.index.duplicated(keep='last')]
        db = db[np.logical_and(db['NUMERIEKEWAARDE'] < 1000000, db['NUMERIEKEWAARDE'] > 0)]
        lobith_dbs[file] = db
    lobith_db = pd.concat(lobith_dbs.values()) 
    lobith_db = lobith_db[~lobith_db.index.duplicated(keep='last')]
    lobith_db = lobith_db.rename(columns={'NUMERIEKEWAARDE':'lobith_debiet'})
    return lobith_db

def featurize_lobith(lobith_db):
    tsd = fm.TimeseriesDataset(lobith_db,ycol ='lobith_debiet')
    TIMESTEP_IN_HOUR = 6
    rolling_funcs = ('mean', 'min', 'max', 'median', 'std', 'sum')
    rolling_args = (TIMESTEP_IN_HOUR, TIMESTEP_IN_HOUR*2,
                    TIMESTEP_IN_HOUR*12, TIMESTEP_IN_HOUR*24)

    rolling_args1 = rolling_args*len(rolling_funcs)
    rolling_funcs1 = [x for item in rolling_funcs for x in repeat(
        item, len(rolling_args))]



    tsd.fm_exec_func(
        tsd.fm_diff,
        arg_dict={'lag': (1, 2),
                'stepsize': (1, 1)})
    tsd.fm_exec_func(
        tsd.fm_lag,
        arg_dict={'lag': (TIMESTEP_IN_HOUR * 24,TIMESTEP_IN_HOUR * 24,TIMESTEP_IN_HOUR * 24,TIMESTEP_IN_HOUR * 24,TIMESTEP_IN_HOUR * 24),'start' : (144*3,144*4,144*5,144*6,144*7)})
    tsd.fm_exec_func(
        tsd.fm_rolling,
        arg_dict={'func': rolling_funcs1, 'window_size': rolling_args1})
    return tsd.dataset

def save_lobith_feats(lobith_feats):
    del lobith_feats['datetime']
    lobith_feats.to_parquet('E:\Rprojects\zoutindringing-parksluizen\data_sets\lobith_feats\lobith_feats.parquet')



def main():
    print(os.getcwd())
    db = parse_lobith('lobith/')
    db = featurize_lobith(db)
    save_lobith_feats(db)


if __name__ == '__main__':
    main()
