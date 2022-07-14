import pandas as pd
import telecontrol_parser as tp
import numpy as np


def main():
    tsdf = tp.egv_standard_run(locatie='parkhaven', threshold=24)
    tsdf = tsdf_interpolate_small_gaps(tsdf, 12)
    print(tsdf_report_gaps(tsdf, 1))
    subset_datasets_dates = tsdf_subset_datasets_dates(tsdf)
    print(subset_datasets_dates)
    print(tsdf_get_datasets(tsdf, subset_datasets_dates))


def tsdf_get_timesteps(tsdf):
    """Gets time steps or 'jumps in time' of a pandas
    dataframe with a datetime index, after parsing should be 10 mins

    Parameters
    ----------
    tsdf : pandas dataframe
        pandas dataframe with a datetime index

    Returns
    -------
    IntegerArray
        list or array with time jumps found
    """
    minute_steps = tsdf['datetime'].diff(1).dt.seconds/60
    present_step_sizes = minute_steps.astype(
        'Int64', errors='ignore').unique()[1:]
    return present_step_sizes


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


if __name__ == '__main__':
    main()
