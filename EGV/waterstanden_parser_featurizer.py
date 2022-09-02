import pandas as pd
import numpy as np
import feature_maker as fm
import timeseries_functions as tf

def parse_waterstanden(path='E:\Rprojects\zoutindringing-parksluizen\waterstanden\Buitenwaterstanden.csv'):
    db = pd.read_csv(path)
    db['datetime'] = pd.to_datetime(db['Unnamed: 0'])
    db['mins'] = db['datetime'].dt.minute % 10
    db = db[db['mins'] == 0]
    db = db.drop('mins', axis=1)
    db = db.reset_index(drop=True)
    db = db.set_index('datetime', drop=False)
    db1 = pd.DataFrame()
    db1['datetime'] = db['datetime']
    db1.set_index('datetime',drop=False)
    db1['parkhaven_stand'] = pd.to_numeric(db['Meetlocatie_OW000752'])
    db1['zaayer_stand'] = pd.to_numeric(db['Meetlocatie_OW000761'])
    db1['schiegemaal_stand'] = pd.to_numeric(db['Meetlocatie_OW000755'])
    db1['westland_stand'] = pd.to_numeric(db['Meetlocatie_OW000747'])
    db1 = db1.replace(-999,np.nan)
    db1 = db1['2015-01-01':]
    return db1

def interpolate_gaps(db):
    return db.interpolate()

def report_gaps(db,respective_col= None,size=1):
    if isinstance(respective_col,type(None)):
        report = {}
        for col in db.columns:
            mask = db[col].isna()
            d = db.index.to_series()[mask].groupby(
                (~mask).cumsum()[mask]).agg(['first', 'size'])
            report[col] = d[d['size'] > size]
        return report
    else:    
        mask = db[respective_col].isna()
        d = db.index.to_series()[mask].groupby(
            (~mask).cumsum()[mask]).agg(['first', 'size'])
    return(d[d['size'] > size])

def featurize_waterstanden(db):
    ycols = list(db.columns)
    ycols.remove('datetime')
    dbs = []
    for ycol in ycols:
        tsd = fm.TimeseriesDataset(db,ycol = ycol)
        rolling_funcs = ('mean','std','range')
        rolling_shifts = (0,6,24*6)
        rolling_window_size = (6,24*6)
        arg_list = fm.fm_args_combiner(
            rolling_funcs,
            rolling_shifts,
            rolling_window_size
        )
        rolling_funcs, rolling_shifts, rolling_windows = arg_list

        tsd.fm_exec_func(
            tsd.fm_diff,
            arg_dict = {'lag': (1,1),
                        'stepsize': (1,2)}
        )

        tsd.fm_exec_func(
            tsd.fm_shifted_rolling,
            arg_dict={
                'func' : rolling_funcs,
                'window_size': rolling_windows,
                'shift' : rolling_shifts
            }
        )
        dbs.append(tsd.dataset)
    return pd.concat(dbs,axis=1)

def main():
    db = parse_waterstanden()
    del db['zaayer_stand']
    # print(report_gaps(db))
    db = interpolate_gaps(db)
    # print(report_gaps(db))
    db = featurize_waterstanden(db)
    # del db['datetime']
    # coln = (list(db.columns))
    # dups = [n for n in coln if coln.count(n) > 1]
    # uniq = list(set(dups))
    # print(uniq)
    db = db.loc[:,~db.columns.duplicated()]
    db.to_parquet(
        'E:\Rprojects\zoutindringing-parksluizen\data_sets\waterstanden_feats\waterstanden_feats.parquet'
    )


if __name__ == '__main__':
    main()