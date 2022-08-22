import pandas as pd
import numpy as np
import feature_maker as fm

def parse_knmi(path):
    db = pd.read_csv(path,skiprows=9)
    db['datetime'] = pd.to_datetime(db['YYYYMMDD'].astype(str) + '-' +(db['H'] - 1).astype(str),format = '%Y%m%d-%H')
    db = db.set_index('datetime',drop = True)
    db = db.drop(['YYYYMMDD','H','# STN'],axis = 1 )
    db.columns = db.columns.str.lstrip()
    return db

def report_gaps(db,respective_col = 'DD',size = 1 ):
    mask = db[respective_col].isna()
    d = db.index.to_series()[mask].groupby(
        (~mask).cumsum()[mask]).agg(['first', 'size'])
    return(d[d['size'] > size])

def main():
    db = parse_knmi('E:\Rprojects\zoutindringing-parksluizen\knmi\geulhaven.txt')
    print(db.columns)
    print(db)
    print(report_gaps(db,'DD'))
    print(report_gaps(db,'FH'))


if __name__ == '__main__':
    main()