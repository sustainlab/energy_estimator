# © All rights reserved. University of Edinburgh, United Kingdom
# IDEAL Project, 2019

import pickle

import numpy as np
import pandas as pd

from pathlib import Path
from datetime import timedelta

from IdealDataInterface import IdealDataInterface

from config import SENSOR_DATA_FOLDER, META_DATA_FOLDER, CACHE_FOLDER, CLICK_DATA_FOLDER, READING_FREQ
from config import IGNORE_HOMES, EVALUATION_PERIOD, AVAILABLE_DATA_THRESHOLD, FFILL_LIMIT

def treatment_control():
    """ Return a DataFrame with details about control/treatment assignment.

    This is used to distinguish which homes fall into which group as well as to find the date
    when a home was assigned one group. Electricity readings dating prior to that date can be
    used to estimate the homes energy use baseline.

    If the energy use estimator should be used for other data, this functions needs to be re-implemented.
    The expected output is this function is a DataFrame with four columns,
      homeid: Identifier for the home
      group: Identifier for treatment or control group
      start_date: The beginning of the study period. Readings before this date can be used for the baseline estimate
      end_date: No reading after this date should be used. Useful for example if a home dropped out etc.
    """
    # Load the home metadata
    fn = META_DATA_FOLDER / Path('home.csv')
    df_homes = pd.read_csv(fn, sep='\t', parse_dates=['endtime'])

    # Use the beginning of the day as dropout day. It could be that a home dropped out in the middle of the day.
    df_homes['end_date'] = df_homes['endtime'].dt.floor(freq='D')

    # Extract the cohort
    # << You can now just use study_class - this can take values of control, treatment or enhanced only. (I'll leave it for now but good to now that this can be made easier)
    df_homes['group'] = df_homes['cohortid'].apply(lambda s: s.split('_')[0])

    # Use str as dtype for homeid
    df_homes['homeid'] = df_homes['homeid'].astype(str)

    df_homes = df_homes[['homeid', 'cohortid', 'end_date', 'group']]
    
    # Load the feature release data (used to determine start dates of control/treatment assignment)
    fn = CLICK_DATA_FOLDER / Path('featureaccess.csv')
    df_features = pd.read_csv(fn, sep=',', parse_dates=['time'])

    # Only keep the WEEK_VIEW as this is the only feature released to control as well as treatment
    df_features = df_features[df_features.featurename == 'WEEK_VIEW']

    df_features = df_features[['cohortid', 'time']]
    df_features.columns = ['cohortid', 'start_date']

    # Use the following day as start date of home belonging to control or treatment. This might be slightly problematic
    # for the baseline computation (as theoretically the baseline could be include already a few hours of 'treatment',
    # however, I think it is the better option than including pre-treatment time in the treatment-phase
    df_features['start_date'] = df_features['start_date'].dt.ceil(freq='D')
    
    # Join the information and select relevant columns
    df = df_homes.merge(df_features, right_on='cohortid', left_on='cohortid', how='left')
    df = df[['homeid', 'group', 'start_date', 'end_date']]

    # Remove homes that are specified on the config file
    df = df.loc[~df['homeid'].isin(IGNORE_HOMES.keys()),:]

    df.drop_duplicates(inplace=True)
    
    return df


def limit_to_full_days(ts):
    """ Limit the time series to full days. """
    assert isinstance(ts.index, pd.DatetimeIndex)
    lower = ts.index.min().ceil('D')

    # Add one second as this will deal correctly with ts that are already ending at the last second
    # of the day and won't interfere with the others.
    upper = (ts.index.max() + timedelta(seconds=1)).floor('D')

    return ts[(ts.index >= lower) & (ts.index < upper)]


def load_mains(homeid):
    """ Load and forward fill the electricity readings using the IdealDataInterface. """
    data_interface = IdealDataInterface(SENSOR_DATA_FOLDER)

    # Load the electricity readings
    ts = data_interface.get(homeid=homeid, subtype='electric-combined')[0]['readings']
    ts = ts.asfreq(READING_FREQ)
    if FFILL_LIMIT is not None:
        ts.ffill(limit=FFILL_LIMIT, inplace=True)

    # The timestamp is right bound, i.e. the reading was taken before the actual timestamp.
    # This has the disadvantage that the full 24h of a day would have as start and end date e.g.,
    # 2017-01-05 00:00:01 and 2017-01-06 00:00:00 (as the timestamp belongs to the reading taken
    # before that time point. We'll subtract one second from the timestamp here to make it left bound
    ts.index = ts.index - timedelta(seconds=1)

    # Limit the data to start and end at midnight
    ts = ts[(ts.index >= ts.index.min().ceil('D')) & (ts.index < ts.index.max().floor('D'))]

    return ts


def load_cached_data(period, full=False):
    """ Load precomputed readings per period. Need to run notebook 02.1... first which will cache the readings.

    args:
       period: The name of the period to load (see config file)
       full (bool): If True all homes will be returned. If False, only homes with sufficient data quality will be returned
                    See notebook 02.2... for finding a good cutoff.

    """
    # Define the folder where the cached files are located
    fpath = CACHE_FOLDER / Path('sampling_cache/')

    if not fpath.is_dir():
        raise IOError('The input data does not seem to be computed yet. Have you run notebook 02.1...?')

    # Get which home is treatment which is control
    df_group = treatment_control()
    
    # Load the complementary information to reconstrut the readings
    homeids, date_ranges, shape = pickle.load(open(fpath / Path('mmap_supplement.pkl'), 'rb'))

    # Open the numpy memmap file containing the readings and load into a DataFrame
    fname = lambda s: fpath / Path('mmap_readings_period_{}.npy'.format(s))
    dat = np.memmap(fname(period), dtype='float32', mode='r', shape=shape[period])
    df = pd.DataFrame(dat, index=date_ranges[period], columns=homeids)

    # Limit to homes that were already allocated during the respective period
    start_date = EVALUATION_PERIOD[period][0]
    end_date = EVALUATION_PERIOD[period][1]# + timedelta(days=EVALUATION_NR_DAYS)

    homes_to_keep = df_group.loc[(df_group['start_date'] <= start_date) & (df_group['end_date'] >= end_date), 'homeid']
    df = df.loc[:, df.columns.isin(homes_to_keep)]

    if not full:
        # Remove all homes with less data then defined in config.py
        idx = df.notna().mean(axis=0) >= AVAILABLE_DATA_THRESHOLD
        df = df.loc[:,idx]

    return df
