import pandas as pd
df = pd.read_table('telecontrol/parkhaven.csv',delimiter = '\t')
df.dtypes
df = df.stack().str.replace(',','.').unstack()
df.head()
df['datetime'] = pd.to_datetime(df['Datum'] + ' ' + df['Tijd (Europe/Amsterdam)'])
df['OW000-008/WNS7670 - gemeten waarde [mS/cm]'] = pd.to_numeric(df['OW000-008/WNS7670 - gemeten waarde [mS/cm]'],errors='coerce')
df.head()
df.plot('datetime', 'OW000-008/WNS7670 - gemeten waarde [mS/cm]')
