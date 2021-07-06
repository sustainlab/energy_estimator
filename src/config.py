# © All rights reserved. University of Edinburgh, United Kingdom
# IDEAL Project, 2018

"""
Configurations
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


#################################
#                               #
#        General Settings       #
#                               #
#################################


## PublicRelease data folder Path
DATA_FOLDER = Path('/path/to/IDEAL/PublicRelease')
SENSOR_DATA_FOLDER = DATA_FOLDER / Path('IDEALsensordata/csv')
META_DATA_FOLDER = DATA_FOLDER / Path('IDEALmetadata/csv')
CLICK_DATA_FOLDER = DATA_FOLDER / Path('IDEALclickdata/csv')

## Temporary folder for caching outputs. <------------------------------------------------------- NEEDS TO BE ADJUSTED -
CACHE_FOLDER = Path('/path/to/impact/cache/')

## Time format string
## See: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

## CPU's to use for multiprocessing. I roughly adjusted this to work on kapok and sloth.
CPU_HIGH_MEMMORY = 8 # assumes about 15G RAM per core and some swap space
CPU_LOW_MEMMORY = os.cpu_count()

#################################
#                               #
#       Evaluation Periods      #
#                               #
#################################

# The frequency the electricity readings is in. This should be set in the DateTimeindex of the
# electricity readings in pandas. See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
READING_FREQ = 's'


# Helper function to shorten the expressions below
date = lambda s: datetime.strptime(s, TIME_FORMAT)

# Define the evaluation periods. The time index should be left-bound, i.e. the right
# side will be included in the analysis.
# It will not cause any problems to have overlapping evaluation periods, e.g. if you want to include
# an evaluation period covering the whole study, as well as shorter periods within that.
# << Niklas, is this correct?: Yes, this is correct and added as comment above
EVALUATION_PERIOD = {'P1_1': (date('2017-07-03 00:00:00'), date('2017-09-10 23:59:59')),
                     'P1_2': (date('2017-09-11 00:00:00'), date('2017-11-19 23:59:59')),
                     'P2_1': (date('2017-11-23 00:00:00'), date('2018-01-31 23:59:59')),
                     'P2_2': (date('2018-02-01 00:00:00'), date('2018-04-11 23:59:59')),
                     'P3': (date('2018-04-20 00:00:00'), date('2018-06-28 23:59:59')),
                     }

# Define the times during the day which should be used for the energy estimation.
# This will be used for all evaluation periods defined above and for the baseline.
# It should be a list of tuples indicating the times of day which should be used
# for the estimate. The tuple ('06:00:00', '06:59:59') for example indicates that
# the time between 6am and 06:59:59 inclusively(!) will be used for the estimate.
# At least one period must be defined and multiple periods can be added, e.g.
#    EVALUATION_TIMES = [('00:00:01', '00:00:50'),
#                        ('00:01:00', '00:01:02')]
#
EVALUATION_TIMES = [('00:00:00', '23:59:59'), ]

# Sanity check the EVALUATION_TIME and assemble the sampling index
assert len(EVALUATION_TIMES) > 0
assert all([ len(i) == 2 for i in EVALUATION_TIMES ])
assert all([ pd.to_timedelta(i, unit='s') < pd.to_timedelta(j, unit='s') for i,j in EVALUATION_TIMES ])

dt = pd.timedelta_range(start='0 day', end='1 day', freq=READING_FREQ, closed='left')

# Assemble the mask that is used to determine which bits of the day to use.
SAMPLING_MASK = np.full(dt.shape, False)
for i,j in EVALUATION_TIMES:
    SAMPLING_MASK[(dt >= i) & (dt <= j)] = True


#################################
#                               #
#        Analysis Settings      #
#                               #
#################################

# This will be dict whose keys are removed from the used homes. The value gives a
# brief description of why the home is removed.
IGNORE_HOMES = {'223':'This home has only very few electricity readings from one clamp spread over '\
                      'the time period',
                '267':'The home reported very few data followed by a long period of missing data. '\
                      'There then is data seemingly reported normaly, however, it is impossible '\
                      'to compute a baseline.',
                '47':'This home does not have electricity readings after 07.05.2017'
                }

# Missing data interpolation. This is the limit to the ffill() call of pandas and will be in units of READING_FREQ
FFILL_LIMIT = 10 # c.f. 'Estimate error with different ffill thresholds.ipynb'
                 # Set to None if no forward filling should be done.

# The estimate of the average energy will only be computed for homes which do not have
# more data missing than set below. The scale is on 0 to 1 with 1 meaning that all readings
# for a period must be present for the home to be included. A value of e.g. 0.7
# denotes that 70% or more of the readings must be available.
AVAILABLE_DATA_THRESHOLD = 0.7


# Maximum number of days to compute the baseline on
BASELINE_NR_DAYS = 7*6


# Start of the rolling window computation
ROLLING_START_DATE = EVALUATION_PERIOD['P1_1'][0] - pd.Timedelta(days=BASELINE_NR_DAYS)


# Length of the rolling windows (used for the average energy computation). This is in in timesteps, that means in units of READING_FREQ
# <<Are the units for ROLLING_WINDOW_WIDTH and ROLLING_WINDOW_STEP seconds, or is it the time steps set by READING_FREQ above? (unit of READING_FREQ)
ROLLING_WINDOW_WIDTH = 60*60*24*7*2 # 2 weeks
ROLLING_WINDOW_STEP = 60*60*24*7 # 1 week

# Sanity check that the rolloing window width is large enough
DELTA_T = pd.to_timedelta(pd.tseries.frequencies.to_offset(s)).total_seconds()
if not ROLLING_WINDOW_WIDTH * DELTA_T > 60*60*24: # The sampling only works for full days
    raise ValueError('The ROLLING_WINDOW_WIDTH seems to be too small. It needs to be at least one full day.')


# Do not compute the average gas use if the gap between two readings is larger or equal than GAS_GAP_THRESHOLD in days.
GAS_GAP_THRESHOLD = 7*5

#################################
#                               #
#        Sampling Settings      #
#                               #
#################################

# Number of time-steps around the time of the day that should be included in the samples (in units of READING_FREQ)
# This will determine how many readings are available to sample from.
SAMPLING_WINDOW = 60
SAMPLING_WINDOW_BASELINE = 60
SAMPLING_WINDOW_ROLLING = 60

# Number of samples to be drawn for the evaluation periods and the rolling estimates
N_SAMPLES = 5000
N_SAMPLES_BASELINE = 5000
N_SAMPLES_BASELINE_SEASONAL = 5000
N_SAMPLES_ROLLING = 2500
