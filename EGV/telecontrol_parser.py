import pandas as pd
import os.path
import numpy as np


# TODO
# data reader function
# data to numeric sep function
# data datetime and index setter
# data cols to numeric function
# data remove flatlines function
# data force time step function
# data select attached series function

def main():
    egv_db = egv_standard_run('parkhaven')
    egv_inspect_ends(egv_db, 4)


def egv_reader(path, delimiter='\t'):
    """Reads telecontrol egv csv files according to path, standard delim is tab

    Parameters
    ----------
    path : str
        complete path to telecontrol csv file
    delimiter : str, optional
        column seperator in csv file, by default '\t'

    Returns
    -------
    tuple
        tuple containing :
        (0): raw contents EGV csv file
        (1): list of columns (str) that can be interpreted as numeric
    """
    egv_src = pd.read_table(path, delimiter=delimiter)
    return egv_src


def egv_get_numeric_cols(egv_db, mode='parse'):
    if mode == 'load':
        return egv_db.columns[2:]
    if mode == 'parse':
        return egv_db.select_dtypes('number').columns


def egv_make_path(locatie):
    """Creates the filepath for a telecontrol file
    based on the physical location

    Parameters
    ----------
    locatie : str
        physical location of EGV measurements

    Returns
    -------
    str
        path pointing to telecontrol csv file
    """
    with open(os.path.dirname(__file__) + '/../teams_path',
              encoding='utf-8') as file:
        lines = file.readlines()
    teams_path = lines[0] + '/'
    path = teams_path + 'telecontrol/'+locatie+'.csv'
    return(path)


def egv_replace_decimal(egv_db, numeric_cols, pattern=',', replace='.'):
    """Replaces decimal commas with points to enable numeric operations

    Parameters
    ----------
    egv_db : pandas dataframe
        dataframe containing columns to have decimal commas replaced
    numeric_cols : list
        list of column names that contain values in the dataframe that
        can be transfromed into numeric values
    pattern : str, optional
        character to be replaced, by default ','
    replace : str, optional
        replacement character, by default '.'

    Returns
    -------
    pandas dataframe
        dataframe containing EGV and temperature values as text
        with datetime related columns
    """
    egv_db[numeric_cols] = egv_db[numeric_cols].stack(
    ).str.replace(pattern, replace).unstack()
    egv_db = egv_db.drop(egv_db.tail(5).index)
    return egv_db


def egv_to_numeric(egv_db, numeric_cols, error_handling='coerce'):
    """Transforms textual numeric columns to actual numeric columns

    Parameters
    ----------
    egv_db : pandas dataframe
        dataframe containing columns with values to be transformed
        into numeric values
    numeric_cols : list
        list of column names that contain values in the dataframe that
        can be transfromed into numeric values
    error_handling : str, optional
        argument for how to handle parsing errors passed on to pd.to_numeric,
        by default 'coerce'

    Returns
    -------
    pandas dataframe
        dataframe containing EGV and temperature values as float
        with datetime related columns
    """
    egv_db[numeric_cols] = egv_db[numeric_cols].apply(
        pd.to_numeric, errors=error_handling)
    return(egv_db)


def egv_index_datetime(egv_db):
    """Creates a month column and a datetime columns which
    it sets as a datetime index whilst keeping it as a column
    itself as well

    Parameters
    ----------
    egv_db : pandas dataframe
        dataframe containing date and time columns called:
        'Datum' & 'Tijd (Europe/Amsterdam)'

    Returns
    -------
    pandas dataframe
        dataframe indexed by datetime containing EGV and
        temperature values with datetime related columns
    """

    egv_db['datetime'] = pd.to_datetime(
        egv_db['Datum'] + ' ' + egv_db['Tijd (Europe/Amsterdam)'])
    egv_db['maand'] = egv_db['datetime'].dt.strftime('%b')
    egv_db = egv_db.set_index('datetime', drop=False)
    return egv_db


def egv_force_time_step(egv_db, step_size='10min'):
    """Forces datframe to have a row every 10 minutes
    by filling with nan

    Parameters
    ----------
    egv_db : pandas dataframe
        dataframe with datetime index
    step_size : str, optional
        time fill, should be chosen based
        on most prevalent timestep in data,
        by default '10min'

    Returns
    -------
    _type_
        _description_
    """
    series_start = egv_db['datetime'][0]
    series_end = egv_db['datetime'][-1]
    datetime_range = pd.date_range(series_start, series_end, freq='10min')
    egv_db = egv_db.reindex(datetime_range, fill_value=np.nan)
    return egv_db


def egv_remove_outliers(egv_db, numeric_cols):
    """Replaces outliers (pending conditions) from numeric columns
    in a pandas dataframe with NaN

    Parameters
    ----------
    egv_db : pandas dataframe
        dataframe containing EGV and temperature values as float
        with datetime related columns
    numeric_cols : list
        list of column names that contain values in the dataframe that
        can be transfromed into numeric values

    Returns
    -------
    pandas dataframe
        dataframe indexed by datetime containing EGV and
        temperature values with datetime related columns
    """
    for col in numeric_cols:
        egv_db[col][egv_db[col] >= 100] = np.nan
        egv_db[col][egv_db[col] < 0] = np.nan
    return egv_db


def egv_remove_repeated_sensor_data(egv_db, numeric_cols,threshold):
    """Sets row per designated numeric columns to NaN if it equals
    previous value in order to remove 'flat lines' from curves
    due to faulty sensor data

    Parameters
    ----------
    egv_db : pandas dataframe
        dataframe indexed by datetime containing EGV and
        temperature values with datetime related columns
    numeric_cols : list
        list of column names that contain values in the dataframe that
        can be transfromed into numeric values
    threshold : int
        size necessary for repeating chunks to be set to NaN. 
        if 0 then no repeating chunks will be set to NaN
        if 1 then consecutive duplicates will be set to NaN
        if >1 then chunks of data that are repeating for size
        threshold will be set to NaN

    Returns
    -------
    pandas dataframe
        dataframe indexed by datetime containing EGV and
        temperature values with datetime related columns
    """
    if threshold == 1:
        egv_db[numeric_cols[1]] = egv_db[numeric_cols[1]].where(
            egv_db[numeric_cols[1]].diff(1) != 0.0, np.nan)
        return egv_db
    else:
        roller = egv_db[numeric_cols[1]].rolling(threshold+1).std().round(6)
        replacement = egv_db[numeric_cols[1]].where(roller != 0, np.nan)
        egv_db[numeric_cols[1]] = replacement
        return egv_db


def egv_inspect_ends(egv_db, size=1):
    """prints head and tail of pandas dataframe

    Parameters
    ----------
    egv_db : pandas dataframe
        dataframe indexed by datetime containing EGV and
        temperature values with datetime related columns
    size : int, optional
        how many rows to print from start and end, by default 1
    """
    print(egv_db.head(size))
    print(egv_db.tail(size))


def egv_standard_run(locatie='parkhaven',threshold=1):
    """wrapper function that runs a standard set
    of cuntions to load telecontrol data

    Parameters
    ----------
    locatie : str, optional
        physical location of measurements, by default 'parkhaven'

    Returns
    -------
    pandas dataframe
        prepared dataframe with numeric values and datetime
        columns
    """

    path = egv_make_path(locatie)
    egv_db = egv_reader(path)
    numeric_cols = egv_get_numeric_cols(egv_db, mode='load')
    egv_db = egv_replace_decimal(egv_db, numeric_cols)
    egv_db = egv_to_numeric(egv_db, numeric_cols)
    egv_db = egv_remove_outliers(egv_db, numeric_cols)
    if threshold > 0:
        egv_db = egv_remove_repeated_sensor_data(egv_db, numeric_cols,threshold = threshold)
    egv_db = egv_index_datetime(egv_db)
    egv_db = egv_force_time_step(egv_db)

    return egv_db


if __name__ == '__main__':
    main()
