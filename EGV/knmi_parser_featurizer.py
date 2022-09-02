import pandas as pd
import numpy as np
import feature_maker as fm

def parse_knmi(path):
    db = pd.read_csv(path,skiprows=9)
    db['datetime'] = pd.to_datetime(db['YYYYMMDD'].astype(str) + '-' +(db['H'] - 1).astype(str),format = '%Y%m%d-%H')
    db = db.set_index('datetime',drop = False)
    db = db.drop(['YYYYMMDD','H','# STN'],axis = 1 )
    db.columns = db.columns.str.lstrip()
    return db

def report_gaps(db,respective_col = 'DD',size = 1 ):
    mask = db[respective_col].isna()
    d = db.index.to_series()[mask].groupby(
        (~mask).cumsum()[mask]).agg(['first', 'size'])
    return(d[d['size'] > size])

def make_cyclic(db,col = 'DD',max = 360):
    db[col + '_sin'] = np.sin(db[col]*(2.*np.pi/max))
    db[col + '_cos'] = np.cos(db[col]*(2.*np.pi/max))
    return db

def combine_vector(db,col1 = 'DD',col2 = 'FH'):
    db[col1 +'_sin_FH'] = db[col1 + '_sin'] * db[col2]
    db[col1 +'_cos_FH'] = db[col1 + '_cos'] * db[col2]
    return db

def featurize_knmi(db):
    dbs = []
    ycols = list(db.columns)
    unwanted = ['DD','datetime']
    ycols = [e for e in ycols if e not in unwanted]
    for ycol in ycols:
        print(ycol)
        tsd = fm.TimeseriesDataset(db,ycol = ycol)
        rolling_funcs = ('mean',)
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
    db = parse_knmi('E:\Rprojects\zoutindringing-parksluizen\knmi\geulhaven.txt')
    db = make_cyclic(db,col = 'DD', max = 360)
    db = combine_vector(db)
    db = featurize_knmi(db)
    db = db.loc[:,~db.columns.duplicated()]
    print(db)
    db = db.resample('10min').mean().interpolate()
    print(db)
    db.to_parquet(
        'E:\Rprojects\zoutindringing-parksluizen\data_sets\knmi_feats\knmi_feats.parquet'
    )



if __name__ == '__main__':
    main()