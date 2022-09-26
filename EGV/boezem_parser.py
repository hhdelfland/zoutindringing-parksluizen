import pandas as pd
import numpy as np
import os


def define_locations():
    location_dict = {
        'OW062C000' : 'beukelsbrug',
        'OW062E001' : 'lage_erf_brug',
        'OW095-000' : 'coolhaven'
    }
    return location_dict
    

def parse_boezem_egv(boezem_egv_path):
    files = os.listdir(boezem_egv_path)
    df = []
    for file in files:
        db = pd.read_csv(boezem_egv_path + file,sep = r'\t')
        locs = define_locations()
        # loc = locs[db.iloc[1,0][1:-1]]
        loc = (db.columns[2][1:10])
        cols = list(db.columns)
        col = [s for s in cols if s.endswith('[mS/cm]"') ]
        db1 = db[col]
        db1.drop(db1.tail(6).index)
        db1 = db1.replace(',','.',regex = True)
        db1 = db1.replace('"',' ',regex = True)
        db1 = db1.apply(pd.to_numeric,errors = 'coerce')
        df.append(db1[col])
    return db1
    
def main():
    print(parse_boezem_egv('E:/Rprojects/zoutindringing-parksluizen/data_sets_boezem/EGV/'))

if __name__ == '__main__':
    main()
