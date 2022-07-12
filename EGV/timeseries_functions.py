import pandas as pd
import telecontrol_parser as tp


def main():
    tsdf = tp.egv_standard_run('parkhaven')
    tsdf_report_gaps(tsdf, 1)
    tsdf = tsdf_interpolate_small_gaps(tsdf)
    tsdf_report_gaps(tsdf, 1)


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
    print(d[d['size'] > size])


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


if __name__ == '__main__':
    main()
