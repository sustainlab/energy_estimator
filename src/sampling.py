# © All rights reserved. University of Edinburgh, United Kingdom
# IDEAL Project, 2019

import numpy as np
import pandas as pd

from random import choice
from copy import deepcopy

from numpy.lib.stride_tricks import as_strided as stride

from config import READING_FREQ, SAMPLING_MASK
from utils import limit_to_full_days


def data_to_sample_array(df, window_width=None, copy=True, force_freq=True):
    """ Convert the DataFrame to an array to be sampled from.

    Each column of the DataFrame should be one home, each row one reading.

    The returned array will have the sample readings per second in each row.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError('The DataFrame must have a DatetimeIndex.')
    if force_freq and not df.index.freq == READING_FREQ:
        raise ValueError('The DatetimeIndex must be of frequency {}.'.format(READING_FREQ))

    if copy:
        t = df.copy()
    else:
        t = df

    if isinstance(t, pd.Series):
        t = t.to_frame()

    # Limit the data to start and end at midnight
    t = limit_to_full_days(t)

    # Sanity check
    if t.isna().all().all():
        raise ValueError('all missing values encountered.')

    # Get some basic info about the DataFrame
    nr_homes = t.shape[1]
    nr_days = (t.index.max() - t.index.min()).ceil('D').days # ceil() as the last timestamp ends one second before midnight
    if not force_freq:
        # In this case, the above way of calculating the number of days does not work. We will instead actually
        # count the indices of days that are in the data. This assumes that if a day has an entry in the DatetimeIndex,
        # it will be the full day
        nr_days = np.unique((t.index.year, t.index.month, t.index.day), axis=1).shape[1]

    # This will be used as "row" for the returned array
    t['second_of_day'] = 60 * 60 * t.index.hour + 60 * t.index.minute + t.index.second

    # Move (melt) all columns so that each electricity reading is in one single row
    # Remaining columns will be second_of_day, variable (i.e. homeid), value (i.e. the reading)
    t = t.melt(id_vars='second_of_day')

    # Sort by second_of_day to allow simple reshaping of the values in order
    # to have the readings of one second_of_day in one row of the array. The
    # additional sort by value is to ensure that the missing values per row
    # will be always at the end of the final array.
    t.sort_values(['second_of_day', 'value'], inplace=True)

    # Sanity check to make sure the reshape does what we expect. Reshape the second_of_day column.
    # This should result in an array with only one value repeated in each row (the second_of_day)
    # There should be one second per day in the evaluation period and per home in the input DataFrame
    shape = (-1, nr_days * nr_homes)
    arr = t['second_of_day'].values.reshape(shape)
    unique_per_row = np.unique(arr, axis=1)
    assert all([len(u) == 1 for u in unique_per_row])

    # If the above is OK we're good to reshape
    arr = t['value'].values.reshape(shape)

    # Depending on the data quality there might not be readings for every second. This will lead
    # to NaN values on the 'right' side of the array (due to sorting them). We can thus 'cut' all
    # columns with all NaNs
    arr = arr[:, :np.where(np.isfinite(arr))[1].max() + 1]

    # We can now add the readings of the previous and successive seconds by rolling and concatenating the array.
    if window_width is not None:
        # Roll the arr and concatenate.
        arr = np.concatenate([ np.roll(arr, shift, axis=0) for shift in range(-window_width, window_width + 1) ], axis=1)
        # Sort again to have NaNs all to the right
        arr.sort(axis=1)

    # We've sorted the values above. It shouldn't matter, but to be safe, shuffle them again
    def f(x):
        idx = np.isfinite(x)
        x[idx] = np.random.permutation(x[idx])
        return x

    arr = np.apply_along_axis(f, axis=1, arr=arr)

    # Get the number of samples per row for which the value is valid, i.e. after that only NaNs are found.
    # If arr has all missing value in one row, this will cause the groupby to omit this row and
    # this has to taken care of by .reindex()
    idx = np.where(np.isfinite(arr))
    idx = pd.DataFrame({'row': idx[0], 'column': idx[1]}).groupby('row').max()
    idx += 1 # convert from index based to counts
    idx = idx.reindex(np.arange(arr.shape[0]), fill_value=0).values
    assert idx.shape[0] == arr.shape[0]

    # arr contains the readings, for each second of the day one row. idx contains the index
    # of the last valid reading (i.e. column) for each row.
    return arr, idx



def sample_energy(N, arr, idx, mask, window_width):
    """ Given an array with readings to sample from, generate N samples of whole day electricity use.
        idx is currently not used and is only kept for backwards compatibility.
        mask must be a boolean array of the same length as arr.shape[0] indicating which time points should be
        included in the estimate (set to True).
        The window_width defines how much rows around the one to sample are included in the sampling pool. """
    # Add the time points that "wrap around" the beginning and end (at midnight)
    # to the arr
    v = arr
    if window_width > 0:
        v = np.concatenate([arr[-window_width:, :], arr, arr[:window_width, :]])

    # Create the strided array, this is inspired by https://stackoverflow.com/a/38879051
    # This will create a 3D array with the first array being the time steps and the
    # other two dimensions the window to sample from. As it's a view its memmory and time efficient
    d0, d1 = v.shape
    s0, s1 = v.strides

    w = 2 * window_width + 1

    a = stride(v, (d0 - (w - 1), w, d1), (s0, s0, s1), writeable=False)

    # Iterate over each time point and sample the number of requested estimates. We can compute
    # the rolling sum, ignoring the time points that are excluded by the mask.
    estimates = np.zeros(N)
    for i, samples in enumerate(a):
        if not mask[i]:
            continue

        samples = samples[np.isfinite(samples)]

        idx = np.random.randint(len(samples), size=N)

        estimates += samples[idx]

    # Convert estimates to kWh
    estimates /= (3.6 * 10 ** 6)

    return estimates


def compute_estimate(ts, N, SAMPLING_MASK, SAMPLING_WINDOW):
    """ Compute the energy estimates. """
    # Limit the data to start and end at midnight and make a copy
    ts = limit_to_full_days(ts.copy())

    # Compute the baseline estimate
    arr, idx = None, np.zeros((len(SAMPLING_MASK),1))
    if not ts.isna().all():
        arr, idx = data_to_sample_array(ts, window_width=None)

    # Compute the number of samples per time point including the sampling window
    if SAMPLING_WINDOW is not None:
        assert SAMPLING_WINDOW >= 0
        # Add the count of the additional rows that are used for sampling
        idx_sampling_window = np.concatenate([np.roll(idx, shift, axis=0) for shift in range(-SAMPLING_WINDOW, SAMPLING_WINDOW + 1)], axis=1)
        idx_sampling_window = idx_sampling_window.sum(axis=1)
    else:
        idx_sampling_window = deepcopy(idx)

    # Estimates will be nan if they can't be computed
    estimates = np.array([np.nan, ])
    if (idx_sampling_window > 0).all():
        estimates = sample_energy(N, arr, idx, mask=SAMPLING_MASK, window_width=SAMPLING_WINDOW)

    return {'estimate_mean': np.mean(estimates),
            'estimate_std': np.std(estimates),
            'missing_data': ts.isna().mean(),
            'min_sample_size': np.min(idx_sampling_window).astype(int) if np.isfinite(idx_sampling_window).all() else np.nan,
            'samples_missing': np.sum(idx_sampling_window == 0) if np.isfinite(idx_sampling_window).all() else np.nan
            }


def compute_sample_sizes(ts, seconds):
    """ Compute how many readings are available to sample from given number of seconds
    around to use for sampling as well. """
    # Compute the sampling data
    arr, idx = data_to_sample_array(ts, window_width=None)

    # Compute how many elements there are depending on the window width taken around
    result = dict()
    result[0] = idx
    for s in seconds:
        if s == 0:
            continue
        # This will concatenate 'shifted' copies of the idx values so that each row remains
        # the second of the day and each column will hold one value of the periods before, at, and
        # after that second of the day. These values can then be summed up to obtain the number of readings
        # one would get if s seconds were to be included in the sampling.
        result[s] = np.concatenate([np.roll(idx, shift, axis=0) for shift in range(-s, s + 1)], axis=1).sum(
            axis=1).reshape(-1, 1)

    return result
