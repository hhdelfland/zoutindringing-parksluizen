import pandas as pd
import telecontrol_parser as tp
import numpy as np
import os


def main():
    locatie = 'westland'
    path = locatie + '_datasets/'
    isExist = os.path.exists(path)
    if not isExist:  
        os.makedirs(path)

    tsdf = tp.egv_standard_run(locatie=locatie, threshold=0)
    print(tsdf)




    # tsdf = tp.egv_standard_run(locatie=locatie, threshold=24)
    # print(tsdf_report_gaps(tsdf, 1))
    # tsdf = tsdf_interpolate_small_gaps(tsdf, 12)
    # print(tsdf_report_gaps(tsdf, 1))
    # subset_datasets_dates = tsdf_subset_datasets_dates(tsdf)
    # print(subset_datasets_dates)
    # datasets = tsdf_get_datasets(tsdf, subset_datasets_dates)
    # print(datasets)
    # tsdf_save_subsets(datasets,path)
    # print(tsdf_read_subsets(1,path))



def tsdf_report_gaps(tsdf, size=6):
    numeric_cols = tp.egv_get_numeric_cols(tsdf)
    respective_col = numeric_cols[1]
    mask = tsdf[respective_col].isna()
    d = tsdf.index.to_series()[mask].groupby(
        (~mask).cumsum()[mask]).agg(['first', 'size'])
    return(d[d['size'] > size])


def tsdf_interpolate_small_gaps(tsdf, max_gapsize=6):
    # src = https://stackoverflow.com/questions/69154946/fill-nan-gaps-in-pandas-df-only-if-gaps-smaller-than-n-nans # noqa
    numeric_cols = tp.egv_get_numeric_cols(tsdf)
    respective_col = numeric_cols[1]
    tsdf_interpolated = tsdf[numeric_cols].interpolate()

    c = respective_col
    mask = tsdf[c].isna()
    x = (
        mask.groupby((mask != mask.shift()).cumsum()).transform(
            lambda x: len(x) > max_gapsize
        )
        * mask
    )
    tsdf_interpolated[c] = tsdf_interpolated.loc[~x, c]
    tsdf[numeric_cols] = tsdf_interpolated
    return tsdf


def tsdf_subset_datasets_dates(tsdf, timestep=10):
    gaps = tsdf_report_gaps(tsdf, 1)
    minutes = timestep*(gaps['size']-1)
    offset = pd.to_timedelta(minutes, unit='m')
    gaps['end'] = gaps['first'] + offset
    last_dates = gaps['first'] - pd.to_timedelta(timestep, unit='m')
    first_dates = gaps['end'] + pd.to_timedelta(timestep, unit='m')
    subset_dates = pd.concat([first_dates, last_dates], axis=1)
    subset_dates.columns = ['subset_start', 'subset_end']
    subset_dates.loc[len(subset_dates)] = [np.nan, np.nan]
    subset_dates['subset_start'] = subset_dates['subset_start'].shift(1)
    subset_dates = subset_dates.reset_index(drop=True)
    subset_dates.loc[0, 'subset_start'] = (tsdf.index[0])
    subset_dates.loc[len(subset_dates)-1, 'subset_end'] = (tsdf.index[-1])
    measurements = subset_dates['subset_end'] - subset_dates['subset_start']
    subset_dates['measurements'] = (
        measurements.dt.total_seconds()/600).astype('int')
    return subset_dates


def tsdf_get_datasets(tsdf, subset_datasets_dates):
    subset_datasets = {}
    for index, row in subset_datasets_dates.iterrows():
        subset_datasets[index] = tsdf[row['subset_start']:row['subset_end']]
    return subset_datasets


def tsdf_standard_run(locatie='parkhaven', threshold=0, interpolate=12):
    tsdf = tp.egv_standard_run(locatie=locatie, threshold=threshold)
    tsdf = tsdf_interpolate_small_gaps(tsdf, interpolate)
    print(tsdf_report_gaps(tsdf, 1))
    subset_datasets_dates = tsdf_subset_datasets_dates(tsdf)
    print(subset_datasets_dates)
    datasets = tsdf_get_datasets(tsdf, subset_datasets_dates)
    return datasets


def tsdf_save_subsets(datasets,path = ''):
    for i in datasets:
        subdata = datasets[i]
        start = str(subdata.index[0])[:10]
        end = str(subdata.index[-1])[:10]
        size = len(subdata)
        subdata.to_csv(path_or_buf=path + f'{start}_{end}_{size}.csv', index=False)


def tsdf_read_subsets(index, path = ''):
    files = []
    for file in os.listdir(path):
        if file.endswith('.csv'):
            files.append(file)
    dataset = pd.read_csv(path + files[index])
    dataset = dataset.set_index('datetime', drop=False)
    return dataset


if __name__ == '__main__':
    main()
