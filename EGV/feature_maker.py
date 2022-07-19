import pandas as pd
import telecontrol_parser as tp
import timeseries_functions as tf
from tsfresh import extract_features

def main():
    dataset = tf.tsdf_read_subsets(3)
    numeric_cols = tp.egv_get_numeric_cols(dataset)
    y = dataset[numeric_cols[1]].to_frame()
    y = y.reset_index(drop = True)
    y['index'] = y.index
    print(y)
    print(extract_features(y,column_id= 'index'))





if __name__ == '__main__':
    main()
    # egv_db = tp.egv_standard_run('parkhaven')
    # tsdf = tf.tsdf_interpolate_small_gaps(egv_db)
    # tsdf['hour'] = [tsdf.index[i].hour for i in range(len(tsdf))]
    # tsdf['weekday'] = [tsdf.index[i].weekday for i in range(len(tsdf))]
    # tsdf['monthday'] =[tsdf.index[i].days_in_month for i in range(len(tsdf))]
    # tsdf['month'] = [tsdf.index[i].month for i in range(len(tsdf))]
