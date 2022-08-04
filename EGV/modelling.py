import os
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import MultiTaskLassoCV
# from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.linear_model import *
from sklearn.metrics import mean_squared_error as mse
from sklearn.multioutput import MultiOutputRegressor

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

    def combine_datasets(self):
        self.datadict['ALL'] = pd.concat(self.datadict.values()) 
        return self

    def get_datasets(self):
        keys = list(self.datadict.keys())
        return keys

    def set_dataset(self,key):
        if isinstance(key,int):
            key = self.get_datasets()[key]
        self.dataset = self.datadict[key]
        return self

    def drop_na(self):
        self.dataset = self.dataset.dropna()
        return self
    
    def clean_columns(self):
        self.dataset = self.dataset._get_numeric_data()
        return self

    def split_predictors(self,pattern = 't+'):
        y_fut_cols = [s for s in self.dataset.columns if pattern in s]
        y_dataset = self.dataset[y_fut_cols]
        x_dataset = self.dataset.drop(y_fut_cols,axis =1)
        return (x_dataset,y_dataset)

    def create_train_test_split(self,ratio):
        train_rows = int(ratio*len(self.dataset))
        train_date = self.dataset.index[train_rows]
        self.train = self.dataset.loc[:train_date]
        self.test = self.dataset.loc[train_date:]
        x,y = self.split_predictors()
        self.x_dataset = x
        self.y_dataset = y
        self.train_x = x.loc[:train_date]
        self.test_x = x.loc[train_date:]
        self.train_y = y.loc[:train_date]
        self.test_y = y.loc[train_date:]
        return self


    def scale_data(self):
        pass

    def linear_regression(self):
        x = self.train_x
        y = self.train_y
        sk_model = LinearRegression().fit(x,y)
        y_pred = sk_model.predict(self.test_x)
        print(mse(self.test_y,y_pred)**(1/2))
        self.model = sk_model
        return self

    def multi_lassoCV(self,**kwargs):
        x = self.train_x
        y = self.train_y
        sk_model = MultiTaskLassoCV(**kwargs).fit(x,y)
        y_pred = sk_model.predict(self.test_x)
        print(mse(self.test_y,y_pred)**(1/2))
        self.model = sk_model
        return self

    def multi_elastic_net(self,**kwargs):
        x = self.train_x
        y = self.train_y
        sk_model = MultiTaskElasticNetCV(**kwargs).fit(x,y)
        y_pred = sk_model.predict(self.test_x)
        print(mse(self.test_y,y_pred)**(1/2))
        self.model = sk_model
        return self
    
    def use_model(self,modelfunc,**kwargs):
        x = self.train_x
        y = self.train_y
        modeltype = modelfunc()
        if modeltype._get_tags()['multioutput']:
            sk_model = modelfunc(**kwargs).fit(x,y)
        else:
            sk_model = MultiOutputRegressor(modelfunc(**kwargs),n_jobs=-1).fit(x,y)
        y_pred = sk_model.predict(self.test_x)
        print(mse(self.test_y,y_pred)**(1/2))
        self.model = sk_model
        return self

    def naive_predictive(self):
        # FIX HARDCODED Y!
            # ZOEK SHIFTS UIT VOOR NAIVE MODEL!!

        x = self.test_x
        y = self.test_y.copy()
        for col in y.columns:
            y[col] = x['EGV_OPP']
        return y

def main():
    MLdb = MLdata('data_sets/feats')
    MLdb.load_datasets()
    # MLdb.combine_datasets()
    # print(MLdb.datadict[list(MLdb.datadict.keys())[2]])
    # print(MLdb.datadict['ALL'])
    # print(MLdb.get_datasets())
    MLdb.set_dataset(1)
    MLdb.clean_columns()
    MLdb.drop_na()
    MLdb.create_train_test_split(0.8)
    MLdb.linear_regression()
    # ZOEK SHIFTS UIT VOOR NAIVE MODEL!!

if __name__ == '__main__':
    main()
