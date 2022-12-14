from ast import While
import os
from tkinter import X
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
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
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
        self.vif_threshold = 10

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

    def load_beukelsbrug_data(self, start = '', features = tuple(),format = 'parquet'):
        for file in os.listdir(self.path):
            if file.startswith('beukelsbrug'):
                print('loading beukelsbrug')
                dataset = pd.read_parquet(self.path+file)
                if start != '':
                    dataset = dataset[start:]
                self.dataset = dataset
        dataset = self.dataset
        start_idx = dataset.index[0]
        end_idx = dataset.index[-1]
        for lf in features:
            print(lf)
            call = getattr(self,'load_'+lf)
            feat_data = call()[start_idx:end_idx]
            dataset[feat_data.columns] = feat_data
        self.dataset = dataset
        return self

    def load_coolhaven(self):
        feat_data = pd.read_parquet(
            r'E:\Rprojects\zoutindringing-parksluizen\data_sets_boezem\features\coolhaven.parquet')
        return feat_data       

    def load_lage_erf_brug(self):
        feat_data = pd.read_parquet(
            r'E:\Rprojects\zoutindringing-parksluizen\data_sets_boezem\features\lage_erf_brug.parquet')
        return feat_data      

    def load_lobith_feats(self):
        feat_data = pd.read_parquet(
            'E:\Rprojects\zoutindringing-parksluizen\data_sets\lobith_feats\lobith_feats.parquet')
        return feat_data

    def load_gemaal_feats(self):
        feat_data = pd.read_parquet(
            'E:\Rprojects\zoutindringing-parksluizen\data_sets\gemaal_feats\gemaal_feats.parquet')
        return feat_data

    def load_waterstanden_feats(self):
        feat_data = pd.read_parquet(
            'E:\Rprojects\zoutindringing-parksluizen\data_sets\waterstanden_feats\waterstanden_feats.parquet')
        return feat_data

    def load_knmi_feats(self):
        feat_data = pd.read_parquet(
            'E:\Rprojects\zoutindringing-parksluizen\data_sets\knmi_feats\knmi_feats.parquet' 
        )
        return feat_data

    def combine_datasets(self):
        self.datadict['ALL'] = pd.concat(self.datadict.values())
        return self

    def get_datasets(self):
        keys = list(self.datadict.keys())
        return keys

    def set_dataset(self, key, features=('lobith_feats',),mode='full'):
        if isinstance(key, int):
            key = self.get_datasets()[key]
        dataset = self.datadict[key]
        self.loaded_feats = features
        start_idx = dataset.index[0]
        end_idx = dataset.index[-1]
        if mode == 'full':
            for lf in features:
                call = getattr(self,'load_'+lf)
                feat_data = call()[start_idx:end_idx]
                dataset[feat_data.columns] = feat_data
        if mode == 'basic':
            y_fut_cols = [s for s in dataset.columns if 't+' in s]
            y_data = dataset[y_fut_cols]
            dataset = dataset.iloc[:,2:6]
            dataset = pd.concat([dataset,y_data],axis=1)

            relevant_col_list = {
                'lobith_feats' : [0,],
                'knmi_feats' : [1,2,3],
                'waterstanden_feats' : [1,2,3],
                'gemaal_feats' : [0,]
            }
            for lf in features:
                call = getattr(self,'load_'+lf)
                feat_data = call()[start_idx:end_idx]
                col_idxs = relevant_col_list[lf]
                feat_data = feat_data.iloc[:,col_idxs]
                dataset[feat_data.columns] = feat_data        
        self.dataset = dataset
        return self

        # if 'lobith_feats' in features:
        #     start_idx = dataset.index[0]
        #     end_idx = dataset.index[-1]
        #     self = self.load_lobith_feats()
        #     lobith_feats = self.lobith_feats[start_idx:end_idx]
        #     dataset[lobith_feats.columns] = lobith_feats
        # if 'gemaal_feats' in features:
        #     start_idx = dataset.index[0]
        #     end_idx = dataset.index[-1]
        #     self = self.load_gemaal_feats()
        #     gemaal_feats = self.gemaal_feats[start_idx:end_idx]
        #     # dataset = pd.concat([dataset,gemaal_feats],axis=1)
        #     dataset[gemaal_feats.columns] = gemaal_feats
        # if 'waterstanden_feats' in features:
        #     start_idx = dataset.index[0]
        #     end_idx = dataset.index[-1]
        #     self = self.load_waterstanden_feats()
        #     waterstanden_feats = self.waterstanden_feats[start_idx:end_idx]
        #     # dataset = pd.concat([dataset,waterstanden],axis=1)
        #     dataset[waterstanden_feats.columns] = waterstanden_feats
        # if 'knmi_feats' in features:
        #     start_idx = dataset.index[0]
        #     end_idx = dataset.index[-1]
        #     self = self.load_knmi_feats()
        #     knmi_feats = self.knmi_feats[start_idx:end_idx]
        #     # dataset = pd.concat([dataset,knmi_feats],axis=1)
        #     dataset[knmi_feats.columns] = knmi_feats


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

    def scale_data(self,mode='x',scaler = 'standard'):
        sc_x = StandardScaler()
        sc_y = StandardScaler()
        if scaler == 'normal':
            sc_x = Normalizer()
            sc_y = Normalizer()
        if scaler == 'minmax':
            sc_x = MinMaxScaler()
            sc_y = MinMaxScaler()
        x_cols = self.train_x.columns
        y_cols = self.train_y.columns
        self.scale_mode = mode
        self.test_x_unscaled = self.test_x
        self.train_x_unscaled = self.train_x
        self.train_x = pd.DataFrame(sc_x.fit_transform(
            self.train_x), index=self.train_x.index, columns=x_cols)
        self.test_x = pd.DataFrame(sc_x.transform(
            self.test_x), index=self.test_x.index, columns=x_cols)
        self.train_y_scaled = pd.DataFrame(sc_y.fit_transform(
            self.train_y), index=self.train_y.index, columns=y_cols)
        self.test_y_scaled = pd.DataFrame(sc_y.transform(
            self.test_y), index=self.test_y.index, columns=y_cols)
        self.scaler_x = sc_x
        self.scaler_y = sc_y
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
        if self.scale_mode == 'y':
            y = self.train_y_scaled

        modeltype = modelfunc()
        if modeltype._get_tags()['multioutput']:
            sk_model = modelfunc(**kwargs).fit(x, y)
        else:
            sk_model = MultiOutputRegressor(
                modelfunc(**kwargs)).fit(x, y)
        y_pred = sk_model.predict(self.test_x)
        if self.scale_mode == 'y':
            y_pred = self.scaler.inverse_transform(sk_model.predict(self.test_x))
        print(mse(self.test_y, y_pred)**(1/2))
        self.model = sk_model
        return self

    def naive_predictive(self,ycol = ''):
        if ycol == '':
            ycol = 'EGV_OPP'

        x = self.test_x_unscaled
        y = self.test_y.copy()
        for col in y.columns:
            y[col] = x[ycol]
        # for col in y.columns:
        #     y[col] = x[:,self.x_dataset.columns == 'EGV_OPP']
        return y

    def get_VIF(self, keep=True):
        df = self.train_x
        # if src == 'self':
        # VIF['feature'] = self.x_dataset.columns
        VIF = pd.Series(np.linalg.inv(df.corr().to_numpy()).diagonal(),
                        index=df.columns,
                        name='VIF')
        self.VIF = VIF
        if keep:
            self.train_x = self.train_x[self.train_x.columns[self.VIF < self.vif_threshold]]
            self.test_x = self.test_x[self.test_x.columns[self.VIF < self.vif_threshold]]
            self.test_x_unscaled = self.test_x_unscaled[self.test_x_unscaled.columns[self.VIF < self.vif_threshold]]
        return self

    def calc_VIF(self,n = 10):
        df = self.train_x
        VIF = pd.Series([10000000])
        while any(VIF>self.vif_threshold):
            VIF = pd.Series(np.linalg.inv(df.corr().to_numpy()).diagonal(),
            index=df.columns,
            name='VIF')
            drop_cols = list(VIF[VIF>self.vif_threshold].sort_values(ascending=False)[:n].index)
            df = df.drop(drop_cols,axis=1)
            print(len(df.columns))

        self.train_x = self.train_x[df.columns]
        self.test_x = self.test_x[df.columns]
        self.test_x_unscaled = self.test_x_unscaled[df.columns]
        return self

    def predict_window(self, startdate=False, past=10, target_var='EGV_OPP',model = ''):
        if model == '':
            model_predictor = self.model
            if not startdate:
                startdate = self.test_x.index[0]
            else:
                startdate = pd.to_datetime(startdate)
            ranges = self.get_dateranges()
            if startdate > ranges['train'][0] and startdate < ranges['train'][1]:
                print('startdate is in train range!, you may see completely or partially fitted predicted values!')
            x = pd.concat([self.train_x,self.test_x])
            x = x[~x.index.duplicated(keep='first')]
            y_pred = model_predictor.predict(x)
            y_pred_df = pd.DataFrame(y_pred,index = x.index)
            num_y = y_pred.shape[1]
            y_pred_sq = y_pred_df.loc[startdate]
        else: #REWRITE
            model_predictor = model
            if not startdate:
                startdate = self.test_x.index[0]
            else:
                startdate = pd.to_datetime(startdate)
            ranges = self.get_dateranges()
            if startdate > ranges['train'][0] and startdate < ranges['train'][1]:
                print('startdate is in train range!, you may see completely or partially fitted predicted values!')
            x = pd.concat([self.train_x,self.test_x])
            x = x[~x.index.duplicated(keep='first')]
            y_pred = (model_predictor.predict(np.reshape(np.array(x),(x.shape[0],1,x.shape[1]))))
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

    def create_predictions(self):
        x = pd.concat([self.train_x,self.test_x])
        x = x[~x.index.duplicated(keep='first')]
        y_pred = self.model.predict(x)
        y_pred_df = pd.DataFrame(y_pred,index = x.index)
        y_pred_df = pd.DataFrame(self.scaler_y.inverse_transform(y_pred_df),index = x.index)
        self.y_pred_df = y_pred_df
        return self

    def predict_window2(self,startdate,past = 2,target_var = 'EGV_OPP',stride = 1):
        ## HOW TO IMPLEMENT STRIDE?!
        num_y = self.y_pred_df.shape[1]
        y_pred_sq = self.y_pred_df.loc[startdate]
        # y_pred_sq = y_pred[0:num_y, :].diagonal()
        x_measured = self.x_dataset
        start_offset = pd.Timedelta(days=past)
        end_offset = pd.Timedelta(minutes=num_y*stride*10)
        xrange = x_measured[startdate-start_offset:startdate+end_offset]
        comp = pd.DataFrame(xrange[target_var].copy())
        # A = ([np.nan]*(int(comp.shape[0])-int(num_y))) # eigenlijk meetpunten in verleden = 2*24*6
        A = [np.nan] * (int(start_offset/pd.Timedelta(minutes=10)) + 1)
        # print('len nan: ' + str(len(A)))
        # print('len num_y: ' + str(num_y))
        y_pred_sq_empty = [np.nan] * (num_y*stride) 
        l = y_pred_sq_empty
        l = [y_pred_sq[(i// stride)] if not i % stride else x for i, x in enumerate(l)]
        A.extend(l)
        # A.extend(y_pred_sq) # direct geplakt, geen rekening met stride
        print('len nan+pred: ' + str(len(A)))
        # print('y_pred_sq_empty: ' + str(len(y_pred_sq_empty)))
        # print(comp.shape)
        # print(l)
        # print(len(l))
        # print(A)
        comp['ypred'] = A
        comp[comp.columns[1]] = comp[comp.columns[1]].shift(stride-1)
        print(comp.tail(20))
        return comp

    def simulate_live(self,times = 2):
        random_date = self.get_random_date('test')
        predictions = []
        for i in range(0,times):
            prediction = self.predict_window2(random_date)
            random_date = random_date + pd.Timedelta(minutes=10)
            print(random_date)
            prediction = prediction.rename(columns = {'ypred' : 'ypred' + str(i)})
            predictions.append(prediction.iloc[:,1])
        res = pd.concat(predictions,axis=1)
        res['EGV_OPP'] = self.x_dataset['EGV_OPP'][res.index[0]:res.index[-1]]
        return res

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
