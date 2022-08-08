import pandas as pd
import feature_maker as fm


def parse_debiet(path='E:\Rprojects\zoutindringing-parksluizen\debiet\DebietParksluizen.csv'):
    db = pd.read_csv(path, skiprows=1)
    db.columns = ('datetime', 'debiet_gemaal', 'kwaliteit')
    db['datetime'] = pd.to_datetime(db['datetime'])
    db['mins'] = db['datetime'].dt.minute % 10
    db = db[db['mins'] == 0]
    db = db.drop('mins', axis=1)
    db = db.reset_index(drop=True)
    db = db.set_index('datetime', drop=False)
    db['debiet_gemaal'] = pd.to_numeric(db['debiet_gemaal'])
    return db


def featurize_debiet(db):
    tsd = fm.TimeseriesDataset(db, ycol='debiet_gemaal')
    tsd.fm_get_streak()
    tsd.fm_shifted_rolling()
    tsd.fm_shifted_rolling(window_size=6*24)
    return tsd


def save_debiet_gemaal_feats(debiet_gemaal_feats):
    del debiet_gemaal_feats['datetime']
    debiet_gemaal_feats.to_parquet(
        'E:\Rprojects\zoutindringing-parksluizen\data_sets\gemaal_feats\gemaal_feats.parquet')


def main():
    db = parse_debiet()
    debiet_gemaal_feats = featurize_debiet(db)
    save_debiet_gemaal_feats(debiet_gemaal_feats.dataset)


if __name__ == '__main__':
    main()
