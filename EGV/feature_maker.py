import pandas as pd
import telecontrol_parser as tp
import timeseries_functions as tf


def main():
    datasets = tf.tsdf_standard_run(
        locatie='parkhaven',
        threshold=24,
        interpolate=12
    )


if __name__ == '__main__':
    main()
    # egv_db = tp.egv_standard_run('parkhaven')
    # tsdf = tf.tsdf_interpolate_small_gaps(egv_db)
    # tsdf['hour'] = [tsdf.index[i].hour for i in range(len(tsdf))]
    # tsdf['weekday'] = [tsdf.index[i].weekday for i in range(len(tsdf))]
    # tsdf['monthday'] =[tsdf.index[i].days_in_month for i in range(len(tsdf))]
    # tsdf['month'] = [tsdf.index[i].month for i in range(len(tsdf))]
