# © All rights reserved. University of Edinburgh, United Kingdom
# IDEAL Project, 2018

import re

#from warnings import warn
import warnings

import numpy as np
import pandas as pd

from pathlib import Path


#################################################################
#                                                               #
#                       WORK IN PROGRESS!                       #
#                                                               #
#################################################################



__TIME_FORMAT__ = '%Y-%m-%d %H:%M:%S'


class IdealDataInterface(object):
    """Interface to the IDEAL Local Data Interface."""

    def __init__(self, folder_path):

        # Make sure the warning is issued every time the user instantiates the class
        warnings.filterwarnings("always", category=UserWarning,
                                module='IdealDataInterface')

        self.folder_path = Path(folder_path)
        self.sensorid_mapping = self._mapping(self.folder_path)

        if len(self.sensorid_mapping) == 0:
            warnings.warn('The specified folder path does not seem to contain any sensor reading files.')

    def _mapping(self, folder_path):
        homeid = list()
        roomid = list()
        roomtype = list()
        sensorid = list()
        category = list()
        subtype = list()
        filename = list()

        for file in folder_path.glob('*.csv.gz'):
            if file.name == 'weatherreading.csv.gz':
                continue

            home_, room_, sensor_, category_, subtype_ = file.name.split('_')

            filename.append(str(file.name))
            homeid.append(int(re.sub('\D', '', home_)))
            roomid.append(int(re.sub('\D', '', room_)))
            roomtype.append(str(re.sub('\d', '', room_)))
            category.append(str(category_))
            subtype.append(str(subtype_[:-7]))

            assert sensor_[:6] == 'sensor'
            sensorid.append(str(sensor_[6:]))

        data = {'homeid': homeid, 'roomid': roomid, 'room_type':roomtype, 'sensorid':sensorid,
                'category': category, 'subtype': subtype, 'filename':filename}
        columns = ['homeid', 'roomid', 'room_type', 'category', 'subtype', 'sensorid', 'filename']

        df = pd.DataFrame(data, columns=columns, dtype=str)
        df.set_index(['homeid', 'roomid', 'room_type', 'category', 'subtype', 'sensorid'], inplace=True)

        #print('Found entries for {} sensor readings.'.format(df.shape[0]))

        return df


    def _filter(self, homeid=None, roomid=None, room_type=None, category=None, subtype=None, sensorid=None):
        def check_input(x):
            """ Assert that the input is a list of strings. """
            if isinstance(x, int):
                x = [str(x), ]
            elif isinstance(x, str):
                x = [x, ]

            if not hasattr(x, '__iter__'):
                raise ValueError('Input {} not understood'.format(x))

            return [str(i) for i in x]

        # Select the matching sensors
        if homeid is None:
            homeid = slice(None)
        else:
            homeid = check_input(homeid)

        if roomid is None:
            roomid = slice(None)
        else:
            roomid = (roomid)

        if room_type is None:
            room_type = slice(None)
        else:
            room_type = check_input(room_type)

        if category is None:
            category = slice(None)
        else:
            category = check_input(category)

        if subtype is None:
            subtype = slice(None)
        else:
            subtype = check_input(subtype)

        if sensorid is None:
            sensorid = slice(None)
        else:
            sensorid = check_input(sensorid)

        filename = slice(None)

        # If homeid, roomid, and room_type are specified, the result will be a Series. This will be converted back
        # to a DataFrame (needs transposing to get it back into the original format)
        #
        # See https://stackoverflow.com/q/30781037 for the axis=0 argument
        try:
            return self.sensorid_mapping.loc(axis=0)[homeid, roomid, room_type, category, subtype, sensorid, filename].to_frame().T
        except AttributeError:
            return self.sensorid_mapping.loc(axis=0)[homeid, roomid, room_type, category, subtype, sensorid, filename]

    def get(self, homeid=None, roomid=None, room_type=None, category=None, subtype=None, sensorid=None):


        df = self._filter(homeid=homeid, roomid=roomid, room_type=room_type, category=category,
                           subtype=subtype, sensorid=sensorid)

        readings = list()
        for (homeid, roomid, room_type, category, subtype, sensorid), row in df.iterrows():
            fname = self.folder_path / Path(row['filename'])

            df = pd.read_csv(fname, header=None, names=['time', 'value'], parse_dates=['time'])

            assert np.issubdtype(df.dtypes['time'], np.datetime64)

            ts = pd.Series(df['value'].values, index=df['time'], name="sensor_{}".format(sensorid))

            #ts /= 10

            readings.append({'homeid': homeid, 'roomid': roomid, 'room_type': room_type,
                             'category': category, 'subtype': subtype, 'sensorid': sensorid,
                             'readings': ts})

        return readings


    def view(self, homeid=None, roomid=None, room_type=None, category=None, subtype=None, sensorid=None):

        df = self._filter(homeid=homeid, roomid=roomid, room_type=room_type, category=category,
                          subtype=subtype, sensorid=sensorid)

        return df.reset_index().loc[:,['homeid', 'roomid', 'room_type', 'category', 'subtype', 'sensorid']]


    def room_types(self):

        return self.sensorid_mapping.reset_index()[['roomid', 'room_type']].groupby('room_type').size().sort_values(ascending=False)

    def categories(self):

        return self.sensorid_mapping.reset_index().loc[:,['category', 'subtype']].drop_duplicates().reset_index(drop=True)

