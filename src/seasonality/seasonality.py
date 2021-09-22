from matplotlib import pyplot as plt
import pandas as pd
from pandas.plotting import autocorrelation_plot
from utils.utils_graphic import plot_by_week
from utils.utils import get_user_data
from preprocessing.studentlife_raw import get_studentlife_dataset, get_sensor_data

def some_f():
    df = get_studentlife_dataset()
    lista = []
    for i in df.index.get_level_values(0).drop_duplicates():
        '''
        Creates 3 plots for each user:
        - The hours in which there is no information about physical activity
        - Autocorrelation plot
        - Plot a week of the user activity (see plot_by_week code)
        
        '''
        min_date = min(df.index.get_level_values(1))
        max_date = max(df.index.get_level_values(1))
        dfu = get_user_data(df, i).droplevel(0).loc[:, 'slevel']
        idx = pd.date_range(min_date, max_date, freq='h')
        d = pd.DataFrame(index=idx)
        d['slevel'] = dfu

        a = d.isna().sum()
        lista.append(a)
        nulls = d.isna()
        plt.close()
        plt.scatter(list(range(len(idx))), nulls, 0.1, marker='x')
        plt.show()

        # Fill nan with previous value
        d.ffill(inplace=True)

        # Autocorrelation Plot
        plt.close()
        plt.rcParams.update({'figure.figsize': (9, 5), 'figure.dpi': 120})
        autocorrelation_plot(d.slevel)
        plt.title('{0},{1}'.format(str(i), a))
        plt.show()

        plot_by_week(i)


def unknown_labels(sensor):
    data = get_sensor_data(sensor)
    data.groupby()

def raw_data_stadistics(freq='1h'):
    sdata = get_sensor_data('activity')
    sdata.columns = ['time', 'activityId', 'userId']
    sdata = sdata.loc[sdata['activityId'] != 3]
    sdata['time'] = pd.to_datetime(sdata['time'], unit='s').dt.floor(freq)
    #return a = sdata.groupby(['userId', 'time']).count()

def generate_MET_stadistics(df):
    '''
    Generates a dataframe with some useful information about all the users
    columns: 'user', 'met', 'std', 'corr', 'nb_nulls'

    '''
    things = list()
    for u in df.index.get_level_values(0).drop_duplicates():
        dfuser = get_user_data(df, u)
        aux = dfuser.droplevel(0).loc[:, 'slevel']
        idx = pd.date_range('2013-03-27 04:00:00', '2013-06-01 3:00:00', freq='h')
        d = pd.DataFrame(index=idx)
        d['slevel'] = aux
        n = d.isna().sum()[0]
        dfuser['hourofday'] = dfuser.index.get_level_values(1).hour
        dfuser['dayofweek'] = dfuser.index.get_level_values(1).dayofweek
        stats = dfuser.groupby(['dayofweek', 'hourofday'])['slevel'].agg(['mean', 'std']).dropna()
        corr = pearsonr(stats['mean'], stats['std'])[0]

        things.append([u, stats['mean'].mean(), stats['std'].mean(), corr, n])
        # corrs.append(corr)
    return pd.DataFrame(columns=['user', 'met', 'std', 'corr', 'nb_nulls'], data=things).sort_values('met')
