from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from collections import Counter
from utils.utils import file_exists
from datetime import datetime
from preprocessing.various import downgrade_datatypes
from datetime import timedelta, datetime
from utils.utils import get_granularity_from_minutes


def create_sensing_table(sensor):
    """
    Creates one dataframe from all the sensor data of all users

    dataset raw data should be at dataset/sensing/ in the project folder
    """
    filename = f'pkl/sensing_data/{sensor}.pkl'
    if file_exists(filename):
        print(f'{sensor} data already generated')
    else:
        path = 'dataset/sensing/' + sensor + '/' + sensor + '_u'
        print(path)
        df = pd.read_csv(path + '00' + '.csv', index_col=False)
        df['userId'] = '00'
        for a in range(1, 60):
            userId = '0' + str(a) if a < 10 else str(a)
            try:
                aux = pd.read_csv(path + userId + '.csv', index_col=False)
                aux['userId'] = a
                df = df.append(aux)
            except:
                pass
        df['userId'] = df['userId'].astype('int8')

        #downgrade datatypes

        df = downgrade_datatypes(df)

        df.to_pickle(filename)


def create_sensing_tables():
    sensor_data_files = ['activity', 'audio','gps', 'dark',
                         'phonelock', 'wifi', 'phonecharge', 'bt',
                         'calendar', 'wifi_location', 'conversation']
    for file in sensor_data_files:
        create_sensing_table(file)


def get_sensor_data(sensor) -> pd.DataFrame:
    return downgrade_datatypes(pd.read_pickle(f'./../pkl/sensing_data/{sensor}.pkl'))


def get_studentlife_dataset(nb_min):

    def to_time(df, col='time'):
        df[col] = pd.to_datetime(df[col], unit='s')
        return df

    def floor_time(df, col='time'):
        df[col] = to_time(df,col)[col].dt.floor(freq)
        return df

    def most_common(lst):
        data = Counter(lst)
        return data.most_common(1)[0][0]

    freq = get_granularity_from_minutes(nb_min)
    print(f'Granularaity is {freq}')    
    filename = f'pkl/sedentarismdata_gran{freq}.pkl'
    if not file_exists(filename):
        print(f'{filename} does not exist. This may take a while...')



        ######################################################################
        # TSD
        ######################################################################      
        print('* ' * 10)
        print('Generating TSD features')
        print('* ' * 10)
        # prepare activity data
        sdata = get_sensor_data('activity')
        sdata.columns = ['time', 'activityId', 'userId']
        sdata = sdata.loc[sdata['activityId'] != 3]
        sdata = floor_time(sdata)

        '''
        Set dataset index from the cartesian product between
        the users id and
        the minimun and maximun date found

        '''
        uindex = sdata['userId'].drop_duplicates()

        min_date = sdata.time.min()
        max_date = sdata.time.max()
        dindex = pd.date_range(min_date, max_date, freq=freq)
        index = pd.MultiIndex.from_product(iterables=[uindex, dindex],
                                            names=['userId', 'time'])
        s = pd.DataFrame(index = index)

        sdata = pd.concat([sdata, pd.get_dummies(sdata['activityId'], prefix='act')], axis=1, sort=False)

        # logs per activity
        count_per_activity = sdata.groupby(['userId', 'time'])[['act_0', 'act_1', 'act_2']].sum()

        for col in count_per_activity.columns:
            s[col] = count_per_activity[col].astype('int64')

        # activitymajor
        s['activitymajor'] = sdata.groupby(['userId', 'time'])['activityId'].apply(most_common).astype('object')
        #s.dropna(how='all', inplace=True) #here all or ... is the same as if a columns is nan the other too
        print('activity features generated')

        hour = s.index.get_level_values('time').hour
        hours_in_day = 24
        s['second_sin'] = np.sin(2*np.pi*hour / hours_in_day)
        s['second_cos'] = np.cos(2*np.pi*hour / hours_in_day)


        # dayofweek
        dayofweek = s.index.get_level_values('time').dayofweek
        days_in_week = 7
        s['dayofweek_sin'] = np.sin(2 * np.pi * dayofweek / days_in_week)
        s['dayofweek_cos'] = np.cos(2 * np.pi * dayofweek / days_in_week)

        #weekofyear
        s['weekofyear'] = s.index.get_level_values('time').weekofyear


        # past minutes since the day began and remaining minutes of the day
        s['past_minutes'] = s.index.get_level_values(1).hour * 60 + s.index.get_level_values(1).minute
        s['remaining_minutes'] = 24 * 60 - s['past_minutes']

        s['is_weekend'] = ((s.index.get_level_values(1).dayofweek==0) | \
            (s.index.get_level_values(1).dayofweek==6))
        print('temporal features generated')


        # prepare audio data
        adata = get_sensor_data('audio')
        adata.columns = ['time', 'audioId', 'userId']
        adata = floor_time(adata)
        
        adata = pd.concat([adata, pd.get_dummies(adata['audioId'], prefix='audio')], axis=1, sort=False)

        # logs per audio
        audio_groups = adata.groupby(['userId', 'time'])
        count_per_audio = audio_groups[['audio_0', 'audio_1', 'audio_2']].sum()

        for col in count_per_audio.columns:
            s[col] = count_per_audio[col].astype('int64')

        # audiomajor
        s['audio_major'] = audio_groups['audioId'].apply(most_common)
        # los siguientes usuarios poseen horas completas en las cuales no tienen ningun registro de audio
        #s.loc[s['audiomajor'].isnull(), 'audiomajor'].groupby('userId').size()
        s.loc[:, 'audio_major'].fillna(method='ffill', axis=0, inplace=True) #suponiendo que se deja de grabar cuando no hay ruido
        s.loc[:,'audio_major'] = s.audio_major.astype('object')    
        print('audio features generated')


        # latitude and longitude mean and std
        gpsdata = get_sensor_data('gps')
        #gpsdata = floor_time(gpsdata)

        gpsdata = gpsdata.loc[:,['latitude','longitude', 'time', 'userId']]
        gpsdata['time'] = pd.to_datetime(gpsdata['time'], unit='s')
        gpsdata_shifted = gpsdata.groupby('userId').shift(1)
        gpsdata_shifted.columns = [f's_{col}' for col in gpsdata.columns if col!='userId']
        gpsdata = pd.concat([gpsdata, gpsdata_shifted], axis=1)
        gpsdata.dropna(axis=0, inplace=True)
        gpsdata['diff_date'] = (gpsdata.time - gpsdata.s_time).dt.seconds
        gpsdata['diff_lat'] = gpsdata.latitude - gpsdata.s_latitude
        gpsdata['diff_lon'] = gpsdata.longitude - gpsdata.s_longitude
        gpsdata['instantaneous_speed'] = np.sqrt( np.square(gpsdata.diff_lat / gpsdata.diff_date) + 
                                    np.square(gpsdata.diff_lon / gpsdata.diff_date))
        gpsdata['lat_plus_lon'] = np.sqrt(np.square(gpsdata.diff_lat) + np.square(gpsdata.diff_lon))
        gpsdata.time = gpsdata.time.dt.floor(freq)
        g = gpsdata.groupby(['userId','time'])
        date_features = g.agg({'instantaneous_speed': ['mean','var'], 'lat_plus_lon': 'sum'})
        date_features.columns = ['speed_mean', 'speed_variance','total_distance']
        date_features.fillna(0, inplace=True)
        gps_grouped = gpsdata.groupby(['userId', 'time'])
        s['location_variance'] = gps_grouped['longitude'].var() + gps_grouped['latitude'].var()
        s['location_mean'] = gps_grouped['longitude'].mean() + gps_grouped['latitude'].mean()
        s = s.join(date_features)

        s.loc[:, 'location_mean'] = s.groupby(level=0)['location_mean'].fillna(method='ffill', axis=0). \
            groupby(level=0).fillna(method='bfill', axis=0)
        for col in ['location_variance', 'speed_mean', 'speed_variance','total_distance']:
            s.loc[:, col].fillna(0, axis=0, inplace=True) # if it is NaN I suppose the user did not move and so std=0
        print('gps features generated')

        # hay datos sobre los wifi mas cercano y ademas sobre los que el usuario estuvo
        # dentro del lugar dnd estaba el wifi,
        # hasta elmomento no se utilizan los datos de wifi cercanos
        # se deja el wifi mas cercano, ademas de la cantidad de wifis
        # a los que se conecto cada usuario en una hora, que puede ser un indicador
        # de sedentarismo

        wifidata = get_sensor_data('wifi_location')
        wifidata = floor_time(wifidata)
        wifidataIn = wifidata.loc[wifidata['location'].str.startswith('in')]
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(wifidataIn['location'].values)
        wifidataIn['location'] = integer_encoded

        # s['wifiMajor'] = 0.0
        # s['wifiMajor'] = wifidataIn.groupby(['userId', 'time'])['location'].apply(Most_Common)
        # s.loc[s['wifiMajor'].isna()] = 0
        wifidataIn.reset_index(inplace=True, drop=True)

        wifiChanges = wifidataIn.groupby(['userId', 'time'])['location'].nunique().astype('int')
        s.loc[:, 'wifi_changes'] = wifiChanges
        s.wifi_changes.fillna(0, inplace=True)
        # a = wifidataIn.groupby(['userId', 'time'])['location']
        # wifidataNear = wfidata.loc[wifidata['location'].str.startswith('near')]

        print('wifi features generated')


        ######################################################################
        # TSI
        ######################################################################
        print('* ' * 10)
        print('Generating TSI features')
        print('* ' * 10)
        def add_interval_features(s, df, col, with_number=False):
                    
            def fill_by_interval_percentage(df):
                to_process_index = (df.end.dt.ceil(freq) - df.start.dt.floor(freq)) > timedelta(seconds=60*nb_min)
                while to_process_index.sum() > 0:
                    aux = df.loc[to_process_index]
                    interval = aux.start.dt.floor(freq) + timedelta(seconds = 60 * nb_min)
                    aux1 = aux.copy()
                    aux1['end'] = interval
                    aux2 = aux.copy()
                    aux2['start'] = interval 
                    df = pd.concat([df[~to_process_index], aux1, aux2], axis=0).reset_index(drop=True)
                    to_process_index = (df.end.dt.ceil(freq) - df.start.dt.floor(freq)) > timedelta(seconds=60*nb_min)

                df['percentage'] = (df.end - df.start) / timedelta(seconds=60*nb_min)
                df['time'] = df.start.dt.floor(freq)
                perc = df.groupby(['userId', 'time'], sort=True)['percentage'].sum()
                perc.loc[perc>1] = 1.0
                return perc

            def count_by_interval(df):
                tuples = list()
                for _, t in df.iterrows():
                    if t['start'] != t['end']:
                        r = pd.date_range(start=t['start'], end=t['end'], freq=freq)
                    else:
                        r = [t['start']]
                    tuples +=  [ (t.userId, d) for d in r]
                #drop duplicates cause there are intervals that matches the hour if finish and the hour of start are equal
                ind = pd.MultiIndex.from_tuples(tuples, names = ['userId','time'])
                aux_series = pd.Series(index=ind)
                convs_per_hour = aux_series.groupby(aux_series.index).size().astype('int')
                return convs_per_hour
                

            df = to_time(df, 'start')
            df = to_time(df, 'end')
            col_perc = col + '_percentage'
            int_perc = fill_by_interval_percentage(df.copy())
            s[col_perc] = int_perc
            s[col_perc].fillna(.0, inplace=True)
            
            s.loc[:, col] = True 
            s.loc[s[col_perc]==0.0, col] = False
            
            if with_number:
                df = floor_time(df, 'start')
                df = floor_time(df, 'end')
                col_nb = f'{col}_nb'
                int_nb = count_by_interval(df)
                s[col_nb] = int_nb
                s[col_nb].fillna(0, inplace=True)
            return s


        # is_charging
        chargedata = get_sensor_data('phonecharge')
        s = add_interval_features(s, chargedata, 'is_charging')
        print('charge features generated')

        # isLocked
        lockeddata = get_sensor_data('phonelock')
        s = add_interval_features(s, lockeddata, 'is_locked')
        print('phone lock features generated')
        
        # isInDark
        darkdata = get_sensor_data('dark')
        s = add_interval_features(s, darkdata, 'is_in_dark')
        print('dark data features generated')

        # prepare conversation data
        conversationData = get_sensor_data('conversation')
        conversationData.columns = ['start', 'end', 'userId']
        s = add_interval_features(s, conversationData, 'is_in_conversation', with_number=True)
        print('conversation features generated')

        s.to_pickle(filename)
        print('StudentLife feature generation finished.')

    else:
        print('Prepocessed StudentLife dataset already generated! Loading it...')

    return downgrade_datatypes(pd.read_pickle(filename))



