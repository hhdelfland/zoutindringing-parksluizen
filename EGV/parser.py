import pandas as pd
import os.path


## TODO
# data reader function
# data numeric sep function
# data datetime and index setter
# data cols to numeric function
# data remove outliers function
# data remove flatlines function
# data force time step function
# data select attached series function

def main():
    with open(os.path.dirname(__file__) + '/../teams_path') as f:
        lines = f.readlines()
    teams_path = lines[0] + '/'
    locatie = 'parkhaven'
    path = teams_path + 'telecontrol/'+locatie+'.csv'
    EGV_res = EGV_reader(path)
    EGV_db = EGV_res[0]
    EGV_numcols = EGV_res[1]
    print(EGV_db.head())
    print(EGV_numcols)

def EGV_reader(path, delimiter = '\t'):
    EGV_src = pd.read_table(path,delimiter = delimiter)
    numeric_cols = EGV_src.columns[2:]
    return((EGV_src,numeric_cols))


if __name__ == '__main__':
    main()

# df = pd.read_table(teams_path + 'telecontrol/'+locatie+'.csv',delimiter = '\t')
# numeric_cols = df.columns[2:]
# df[numeric_cols] = df[numeric_cols].stack().str.replace(',','.').unstack()
# df.head()
# df.drop(df.tail(5).index,inplace = True)
# df['datetime'] = pd.to_datetime(df['Datum'] + ' ' + df['Tijd (Europe/Amsterdam)'])
# df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
# df.head()
# df = df.set_index('Datum')
# df[numeric_cols] = np.where(df[numeric_cols]>=100 , np.nan, df[numeric_cols])
# df['maand'] = df['datetime'].dt.strftime('%b')
# cdf = df
# cdf[numeric_cols[1]] = cdf[numeric_cols[1]].where(cdf[numeric_cols[1]].diff(1)!=0.0,np.nan)
# df = cdf
# df = df.reset_index()
# df = df.set_index('datetime',drop = False)

