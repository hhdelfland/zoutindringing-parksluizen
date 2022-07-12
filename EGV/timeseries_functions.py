import pandas as pd
import telecontrol_parser as tp


def main():
    tsdf = tp.egv_standard_run('parkhaven')
    print(tsdf_get_timesteps(tsdf))


def tsdf_get_timesteps(tsdf):
    """Gets time steps or 'jumps in time' of a pandas
    dataframe with a datetime index

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


if __name__ == '__main__':
    main()
