from ast import While
import os
from tracemalloc import start
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import MultiTaskLassoCV
# from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.linear_model import *
from sklearn.metrics import mean_squared_error as mse
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# def mdl_get_feats(index):
#     files = []
#     for file in os.listdir('data_sets/feats'):
#         if file.startswith('feats_') and file.endswith('.parquet'):
#             files.append(file)
#     dataset = pd.read_parquet('data_sets/feats/'+files[index])
#     dataset = dataset.set_index('datetime', drop=True)
#     return dataset




class MLdata:
    def __init__(self, path):
        self.path = path

    def load_datasets(self, format='parquet'):
        datadict = {}
        cnt = 0
        for file in os.listdir(self.path):
            if file.startswith('feats_') and file.endswith('.'+format):
                key = 'dataset_'+str(cnt)+'_'+file.split('_')[3]
                dataset = pd.read_parquet(self.path+'/'+file)
                datadict[key] = dataset
                cnt += 1
        self.datadict = datadict
        return self

    def load_lobith_feats(self):
        self.lobith_feats = pd.read_parquet(
            'E:\Rprojects\zoutindringing-parksluizen\data_sets\lobith_feats\lobith_feats.parquet')
        return self

    def load_gemaal_feats(self):
        self.gemaal_feats = pd.read_parquet(
            'E:\Rprojects\zoutindringing-parksluizen\data_sets\gemaal_feats\gemaal_feats.parquet')
        return self

    def load_waterstanden_feats(self):
        self.waterstanden_feats = pd.read_parquet(
            'E:\Rprojects\zoutindringing-parksluizen\data_sets\waterstanden_feats\waterstanden_feats.parquet')
        return self

    def combine_datasets(self):
        self.datadict['ALL'] = pd.concat(self.datadict.values())
        return self

    def get_datasets(self):
        keys = list(self.datadict.keys())
        return keys

    def set_dataset(self, key, features=('lobith_feats',)):
        if isinstance(key, int):
            key = self.get_datasets()[key]
        dataset = self.datadict[key]
        if 'lobith_feats' in features:
            start_idx = dataset.index[0]
            end_idx = dataset.index[-1]
            self.load_lobith_feats()
            lobith_feats = self.lobith_feats[start_idx:end_idx]
            dataset[lobith_feats.columns] = lobith_feats
        if 'gemaal_feats' in features:
            start_idx = dataset.index[0]
            end_idx = dataset.index[-1]
            self.load_gemaal_feats()
            gemaal_feats = self.gemaal_feats[start_idx:end_idx]
            # dataset = pd.concat([dataset,gemaal_feats],axis=1)
            dataset[gemaal_feats.columns] = gemaal_feats
        if 'waterstanden_feats' in features:
            start_idx = dataset.index[0]
            end_idx = dataset.index[-1]
            self.load_waterstanden_feats()
            waterstanden_feats = self.waterstanden_feats[start_idx:end_idx]
            # dataset = pd.concat([dataset,waterstanden],axis=1)
            dataset[waterstanden_feats.columns] = waterstanden_feats
        
        self.dataset = dataset
        return self

    def drop_na(self):
        self.dataset = self.dataset.dropna()
        return self

    def clean_columns(self):
        self.dataset = self.dataset._get_numeric_data()
        return self

    def split_predictors(self, pattern='t+'):
        y_fut_cols = [s for s in self.dataset.columns if pattern in s]
        y_dataset = self.dataset[y_fut_cols]
        x_dataset = self.dataset.drop(y_fut_cols, axis=1)
        return (x_dataset, y_dataset)

    def create_train_test_split(self, ratio):
        train_rows = int(ratio*len(self.dataset))
        train_date = self.dataset.index[train_rows]
        self.train = self.dataset.loc[:train_date]
        self.test = self.dataset.loc[train_date:]
        x, y = self.split_predictors()
        self.x_dataset = x
        self.y_dataset = y
        self.train_x = x.loc[:train_date]
        self.test_x = x.loc[train_date:]
        self.train_y = y.loc[:train_date]
        self.test_y = y.loc[train_date:]
        return self

    def scale_data(self):
        sc = StandardScaler()
        x_cols = self.train_x.columns
        self.test_x_unscaled = self.test_x
        self.train_x_unscaled = self.train_x
        self.train_x = pd.DataFrame(sc.fit_transform(
            self.train_x), index=self.train_x.index, columns=x_cols)
        self.test_x = pd.DataFrame(sc.transform(
            self.test_x), index=self.test.index, columns=x_cols)
        self.scaler = sc
        return self

    def linear_regression(self):
        x = self.train_x
        y = self.train_y
        sk_model = LinearRegression().fit(x, y)
        y_pred = sk_model.predict(self.test_x)
        print(mse(self.test_y, y_pred)**(1/2))
        self.model = sk_model
        return self

    def multi_lassoCV(self, **kwargs):
        x = self.train_x
        y = self.train_y
        sk_model = MultiTaskLassoCV(**kwargs).fit(x, y)
        y_pred = sk_model.predict(self.test_x)
        print(mse(self.test_y, y_pred)**(1/2))
        self.model = sk_model
        return self

    def multi_elastic_net(self, **kwargs):
        x = self.train_x
        y = self.train_y
        sk_model = MultiTaskElasticNetCV(**kwargs).fit(x, y)
        y_pred = sk_model.predict(self.test_x)
        print(mse(self.test_y, y_pred)**(1/2))
        self.model = sk_model
        return self

    def use_model(self, modelfunc, **kwargs):
        x = self.train_x
        y = self.train_y
        modeltype = modelfunc()
        if modeltype._get_tags()['multioutput']:
            sk_model = modelfunc(**kwargs).fit(x, y)
        else:
            sk_model = MultiOutputRegressor(
                modelfunc(**kwargs)).fit(x, y)
        y_pred = sk_model.predict(self.test_x)
        print(mse(self.test_y, y_pred)**(1/2))
        self.model = sk_model
        return self

    def naive_predictive(self):
        # FIX HARDCODED Y!
        # ZOEK SHIFTS UIT VOOR NAIVE MODEL!!
        x = self.test_x_unscaled
        y = self.test_y.copy()
        for col in y.columns:
            y[col] = x['EGV_OPP']
        # for col in y.columns:
        #     y[col] = x[:,self.x_dataset.columns == 'EGV_OPP']
        return y

    def get_VIF(self, src='self', keep=True):
        df = self.train_x
        # if src == 'self':
        # VIF['feature'] = self.x_dataset.columns
        VIF = pd.Series(np.linalg.inv(df.corr().to_numpy()).diagonal(),
                        index=df.columns,
                        name='VIF')
        self.VIF = VIF
        if keep:
            self.train_x = self.train_x[self.train_x.columns[self.VIF < 10]]
            self.test_x = self.test_x[self.test_x.columns[self.VIF < 10]]
            self.test_x_unscaled = self.test_x_unscaled[self.test_x_unscaled.columns[self.VIF < 10]]
        return self

    def calc_VIF(self,n = 10):
        df = self.train_x
        VIF = pd.Series([10000000])
        while any(VIF>10):
            VIF = pd.Series(np.linalg.inv(df.corr().to_numpy()).diagonal(),
            index=df.columns,
            name='VIF')
            drop_cols = list(VIF[VIF>10].sort_values(ascending=False)[:n].index)
            df = df.drop(drop_cols,axis=1)
            print(len(df.columns))

        self.train_x = self.train_x[df.columns]
        self.test_x = self.test_x[df.columns]
        self.test_x_unscaled = self.test_x_unscaled[df.columns]
        return self

    def predict_window(self, startdate=False, past=10, target_var='EGV_OPP'):
        model = self.model
        if not startdate:
            startdate = self.test_x.index[0]
        else:
            startdate = pd.to_datetime(startdate)

        ranges = self.get_dateranges()
        if startdate > ranges['train'][0] and startdate < ranges['train'][1]:
            print('startdate is in train range!, you may see completely or partially fitted predicted values!')


        x = pd.concat([self.train_x,self.test_x])
        x = x[~x.index.duplicated(keep='first')]
        y_pred = model.predict(x)
        y_pred_df = pd.DataFrame(y_pred,index = x.index)
        num_y = y_pred.shape[1]
        y_pred_sq = y_pred_df.loc[startdate]


        # y_pred_sq = y_pred[0:num_y, :].diagonal()


        x_measured = self.x_dataset

        start_offset = pd.Timedelta(days=past)
        end_offset = pd.Timedelta(minutes=num_y*10)
        xrange = x_measured[startdate-start_offset:startdate+end_offset]
        comp = pd.DataFrame(xrange[target_var].copy())

        A = ([np.nan]*(comp.shape[0]-num_y))
        A.extend(y_pred_sq)
        comp['ypred'] = A

        return comp

    def get_dateranges(self):
        ranges = {'train': [self.train_x.index[0], self.train_x.index[-1]],
                  'test': [self.test_x.index[0], self.test_x.index[-1]]
                  }
        return ranges

    def get_random_date(self, subset = 'test'):
        ranges = self.get_dateranges()
        if subset == 'train':
            date = self.dataset[ranges['train'][0]:ranges['train'][1]].sample(1).index
        elif subset == 'test':
            date = self.dataset[ranges['test'][0]:ranges['test'][1]].sample(1).index
        elif subset == 'all':
            date = self.dataset.sample(1).index
        else:
            return
        return date[0]

def main():
    MLdb = MLdata('data_sets/feats')
    MLdb.load_datasets()
    # MLdb.combine_datasets()
    # print(MLdb.datadict[list(MLdb.datadict.keys())[2]])
    # print(MLdb.datadict['ALL'])
    # print(MLdb.get_datasets())
    MLdb.set_dataset(0, features=('lobith_feats',))
    MLdb.clean_columns()
    MLdb.drop_na()
    MLdb.create_train_test_split(0.8)
    # MLdb.linear_regression()
    # print(MLdb.dataset.shape)
    # start_idx = MLdb.dataset.index[0]
    # end_idx = MLdb.dataset.index[-1]
    # print(MLdb.lobith_feats[start_idx:end_idx].shape)


if __name__ == '__main__':
    main()
