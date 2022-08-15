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
    pass

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


def main():
    db = parse_waterstanden()
    del db['zaayer_stand']
    print(report_gaps(db))

if __name__ == '__main__':
    main()