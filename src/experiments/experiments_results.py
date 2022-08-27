from os import listdir
import pickle as pkl
import numpy as np
import random 
import pandas as pd
import pickle
import time
import os 
from matplotlib import lines
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.stats import ttest_ind, wilcoxon
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
sns.set_style("whitegrid")

from preprocessing.datasets import get_clean_dataset
from utils.utils import get_experiment_combinations, get_granularity_from_minutes
from experiments.experiment_running import get_closests


def get_classification_results(keywords):
    return [(f[0:-4], pkl.load(open(f'./pkl/results/{f}', 'rb'))) for f in listdir('./pkl/results') if all(f'_{k}' in f for k in keywords)]


def print_classification_results(keywords):
    (names, results) = map(list, zip(*get_classification_results(keywords)))
    show_metric('', 'RMSE', names, results)


def show_metric(title, ylabel, labels, data):
    user_labels = get_list_of_users()
    users_range = np.arange(1, len(user_labels))

    plt.close()
    for d in data:
        plt.scatter(users_range, d, marker='s', c=(random.random(), random.random(), random.random()))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('User')
    plt.legend(labels,
               loc='best')
    plt.xticks(users_range, user_labels, rotation='vertical')
    plt.grid(True)
    plt.show()


def get_experiments_data(with_y_test_pred=False):
    df = pd.read_pickle(f'../pkl/experiments/experiments_df.pkl')
    if not with_y_test_pred:
        return df.loc[:, [col for col in df.columns if col!='y_test_pred']]
    return df


def generate_df_from_experiments():
    """Generate a DataFram from all the available experiment results
    """
    start = time.time()
    rows = []
    combs = get_experiment_combinations()
    closest = get_closests()
    for poi, arch, user, gran, nb_lags, period in combs:
        name = f'_regression_gran{get_granularity_from_minutes(gran)}_period{period}_lags{nb_lags}_model-{arch}_user{user}_{poi}'
        print(name)
        filename = f'../pkl/experiments/{name}.pkl'
        exp_data = pkl.load(open(filename, 'rb'))
        centroid = closest[user]
        new_row = {
            'poi': poi,
            'arch': arch,
            'user': user,
            'gran': gran,
            'nb_lags': nb_lags,
            'period': period,
            'scores': exp_data['scores'],
            'nb_params': exp_data['nb_params'],
            'y_test_pred': exp_data['y_test_pred'],
            'time_to_train': exp_data['time_to_train'],
            'centroid': 'low_met' if centroid == 34 else 'high_met'
        }
        rows.append(new_row)

    df = pd.DataFrame(rows)
    df['mean_score'] = df.scores.apply(lambda x : np.mean(x))
    df['mean_time'] = df.time_to_train.apply(lambda x : np.mean(x))

    df[[f'score_{i}' for i in range(5)]] = pd.DataFrame(df.scores.tolist())
    del df['scores']

    df[[f'time_{i}' for i in range(5)]] = pd.DataFrame(df.time_to_train.tolist()) 
    del df['time_to_train']


    filename = f'../pkl/experiments/experiments_df.pkl'
    df.to_pickle(filename)
    print(f'This took {round((time.time() - start)/60, 3)}')


def rank_results(comp_col='arch', rank_by='score', based_on='user', ix=-1, **kwargs):
    '''
    This function generates a table that ranks the specify comp_col columns
    based on its performance (mean_score col) for all the users
    
    '''
    # TODO implement from agregation function
    df = get_experiments_data()

    assert comp_col in df.columns, f'comp_col must be one of {df.columns}'
    assert comp_col not in kwargs.keys() , f'comp_col cant be a filter keyword'
    assert all(k in df.columns for k in kwargs.keys()) , f'kwargs must be one of {df.columns}'
    
    col_values = list(df[comp_col].drop_duplicates())
    nb_values = len(col_values)
    rank_col_names = [f'Puesto {i}' for i in range(1,nb_values+1)]
    
    if rank_by in ['score','time']:
        if ix>=0:
            rank_by = f'{rank_by}_{ix}'
        else: rank_by = f'mean_{rank_by}'

    for k,v in kwargs.items():
        df = df.loc[df[k]==v]
    
    rows = []
    for bo in df[based_on].drop_duplicates():
        experiments_subset = df.loc[(df[based_on]==bo)]
        
        sorted_scores = experiments_subset[[comp_col,rank_by]].sort_values(rank_by, ascending=True).drop_duplicates(subset=[comp_col])

        best_based_on = sorted_scores.iloc[0:nb_values,0].values
        best_rank_by = np.round(sorted_scores.iloc[0:nb_values,1].values,4)

        row = {'bo': bo, 'best_based_on': best_based_on, 'best_rank_by': best_rank_by }
        rows.append(row)

    results = pd.DataFrame(rows)
    results[rank_col_names] = pd.DataFrame(results.best_based_on.tolist(), index=results.index)
    del results['best_based_on']

    summarize = pd.DataFrame(columns=results.columns, index=col_values)
    for i in summarize.columns:
        for j in summarize.index.values:
            summarize.at[j,i] = sum(results[i]==j)
    del summarize['bo']
    del summarize['best_rank_by']
    return summarize, results


def rank_results_agg_func(comp_col='arch', rank_by='score', based_on='user', ix=-1, agg_func = 'mean', **kwargs):
    '''
    This function generates a table that ranks the specify comp_col columns
    based on its performance (mean_score col) for all the users
    
    '''
    # TODO implement from agregation function
    df = get_experiments_data()

    assert comp_col in df.columns, f'comp_col must be one of {df.columns}'
    assert comp_col not in kwargs.keys() , f'comp_col cant be a filter keyword'
    assert all(k in df.columns for k in kwargs.keys()) , f'kwargs must be one of {df.columns}'
    
    col_values = list(df[comp_col].drop_duplicates())
    nb_values = len(col_values)
    rank_col_names = [f'Puesto {i}' for i in range(1,nb_values+1)]
    
    if rank_by in ['score','time']:
        if ix>=0:
            rank_by = f'{rank_by}_{ix}'
        else: rank_by = f'mean_{rank_by}'

    for k,v in kwargs.items():
        df = df.loc[df[k]==v]
    rows = []
    for bo in df[based_on].drop_duplicates():
        experiments_subset = df.loc[(df[based_on]==bo)]
        
        if agg_func == 'mean':
            agg_func_app = np.mean
        elif agg_func == 'median':
            agg_func_app = np.median
        elif agg_func == 'max':
            agg_func_app = np.max
        elif agg_func == 'min':
            agg_func_app = np.min
        
        grouped_exps = experiments_subset.loc[:,[comp_col,rank_by]].groupby(comp_col)
        agg_func_apply = grouped_exps.agg([agg_func])
        agg_func_apply.columns = [rank_by]

        sorted_results = agg_func_apply.sort_values(by=rank_by, ascending=True)
        #print(sorted_results)

        sorted_results_cleaned = sorted_results.reset_index(drop=False)
        #print(sorted_results_cleaned)

        best_based_on = sorted_results_cleaned.iloc[0:nb_values,0].values
        best_rank_by = np.round(sorted_results_cleaned.iloc[0:nb_values,1].values,4)

        row = {'bo': bo, 'best_based_on': best_based_on, 'best_rank_by': best_rank_by }
        rows.append(row)

    results = pd.DataFrame(rows)
    results[rank_col_names] = pd.DataFrame(results.best_based_on.tolist(), index=results.index)
    del results['best_based_on']
    summarize = pd.DataFrame(columns=results.columns, index=col_values)
    del summarize['bo']
    del summarize['best_rank_by']
    for i in summarize.columns:
        for j in summarize.index.values:
            summarize.at[j,i] = sum(results[i]==j)
    return summarize, results


def filter_exp(**kwargs):
    df = get_experiments_data()
    for k,v in kwargs.items():
        df = df.loc[df[k]==v]
    return df


def order_exp_by(ix=-1, rank_by='score', **kwargs):
    df = filter_exp(**kwargs)
    if ix>=0:
        rank_by_col = f'{rank_by}_{ix}'
    else: rank_by_col = f'mean_{rank_by}'
    return df.sort_values(by=rank_by_col), rank_by_col


def check_results_correctness():
    """Check if experiment results makes sense
    """
    poi = 'per'
    arch = 'mlp'
    nb_lags = 4
    period = 4
    gran = 60
    user = 32
    df = get_experiments_data(with_y_test_pred=True)
    exp = df.loc[((df.poi==poi) & (df.arch==arch) & (df.nb_lags==nb_lags) & (df.period == period) & (df.user==user) & (df.gran==gran))]
    exp = exp.y_test_pred.values[0]
    y_test, y_pred,shapes = get_test_predicted_arrays(exp, return_shapes=True)
    print(y_test.shape, y_pred.shape)
    dataset = get_lagged_dataset(user=user, nb_lags=nb_lags, period=period, nb_min=gran)
    y_test_total = dataset.slevel.values
    y_test_exp = y_test
    nb_cases = y_test_total.shape[0]
    nb_exp_cases = y_test_exp.shape[0]
    diff = nb_cases - nb_exp_cases
    print(f'total nb of cases: {nb_cases}')
    print(f'total nb of cases in the exp: {nb_exp_cases}')
    print(f'diff: {diff}')
    y_test_total_cut = y_test_total[diff:]
    print(y_test_total_cut.shape)
    print(y_test_exp.shape)
    new_df = pd.DataFrame(data={'total': y_test_total_cut, 'exp': y_test_exp})
    shapes = list(np.cumsum(shapes))
    shapes = [0] + shapes[:-1]
    print(shapes)
    for i in range(len(shapes)):
        arr1 = y_test_total_cut[shapes[i]:shapes[i]+10]
        arr2 = exp[i][0][:10]
        new_df = pd.DataFrame(data={'total': arr1, 'exp': arr2})
        print(new_df)        


def get_test_predicted_arrays(exp_data, return_shapes=False):
    zipped = zip(*exp_data)
    l = list(zipped)
    # fails if there is only one example
    l[1] = [np.squeeze(a) for a in l[1]]
    y_test = np.concatenate(l[0]) 
    y_pred = np.concatenate(l[1])
    shapes = [arr.shape[0] for arr in l[0]]
    if return_shapes: 
        return y_test, y_pred, shapes
    else: 
        return y_test, y_pred


def print_results(fromi=1, toi=5, archs=['rnn', 'tcn', 'cnn', 'mlp'], poi='per', user=32, lags=1, period=1, gran=60):
    """Plots the results of a set of experiments 
    """
    df = get_experiments_data(False, with_y_test_pred=True)
    exp = df.loc[((df.poi==poi) & (df.user==user) & (df.nb_lags==lags) & (df.period==period) & (df.gran==gran))]
    plt.close()
    width = 4 + 2*(toi-fromi+1)
    plt.figure(figsize=(width,4))
    first_pass = True
    for arch in archs:
        exp_arch = exp.loc[df.arch==arch,:]
        exp_data = exp_arch.y_test_pred.values[0][fromi-1:toi]
        y_test, y_pred, shapes = get_test_predicted_arrays(exp_data, return_shapes=True)
        lw = .6
        if first_pass:
            plt.plot(y_test, label='Test', lw=lw)
            first_pass = False
            acc_shapes = np.cumsum(np.array(shapes))
            for shape in acc_shapes:
                plt.axvline(shape, color='black', ls='--')
            plt.xlim(0, acc_shapes[-1])
        plt.plot(y_pred, label=f'Predicho ({arch.upper()})', lw=lw)
    plt.axhline(1.5, color='red',ls=':')
    plt.ylim(0,8.5)
    plt.tick_params(
        axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.ylabel('MET')
    plt.tight_layout()
    plt.legend()
    plt.show()


def plot_iterations_time_pattern(max_minutes=None,**kwargs):
    """Plots the time it took for each experiment

    The time is plotted for each iteration and data points are joined with a line.
    There is a different color for each neural network type.

    Args:
        max_minutes (_type_, optional): _description_. Defaults to None.
    """
    df = filter_exp(**kwargs)
    column_names = [f'time_{i}' for i in range(5)]
    plt.close()
    ax = plt.figure(figsize=(6, 6)).gca()

    archs_colors={'rnn': 'b', 'tcn': 'g', 'cnn': 'r', 'mlp': 'm'}
    its = np.arange(1,6)
    first_pass = True
    for arch in archs_colors.keys():
        exp_arch = df.loc[df.arch==arch,:]
        
        to_plot = exp_arch.loc[:, column_names].values.tolist()
        for list in to_plot:
            ax.plot(its, list, lw=.01, color=archs_colors[arch])
            first_pass = False
        first_pass = True

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(1,5)
    if max_minutes != None: 
        ax.set_ylim(0, max_minutes)
    plt.ylabel('Time')
    plt.xlabel('Iteration')

    handles = [lines.Line2D([], [], color=c,
                            markersize=15, label=k.upper()) for k,c in archs_colors.items()]
    plt.legend(loc='upper left', handles=handles)

    plt.show()


def plot_iterations_score_pattern(max_mse=None, **kwargs):
    df = filter_exp(**kwargs)
    column_names = [f'score_{i}' for i in range(5)]
    plt.close()
    ax = plt.figure(figsize=(6, 6)).gca()

    archs_colors={'rnn': 'b', 'tcn': 'g', 'cnn': 'r', 'mlp': 'm'}
    its = np.arange(1,6)
    first_pass = True
    for arch in archs_colors.keys():
        exp_arch = df.loc[df.arch==arch,:]
        
        to_plot = exp_arch.loc[:, column_names].values.tolist()
        for list in to_plot:
            ax.plot(its, list, lw=.01, color=archs_colors[arch])
            first_pass = False
        first_pass = True

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(1,5)
    if max_mse != None: 
        ax.set_ylim(0, max_mse)
    plt.ylabel('MSE')
    plt.xlabel('Iteration')

    handles = [lines.Line2D([], [], color=c,
                            markersize=15, label=k.upper()) for k,c in archs_colors.items()]
    plt.legend(loc='upper left', handles=handles)

    plt.show()



#######################
# Funtions for plotting results where all the users al positioned
# using the logic used to calculate the clusters and centroids
########################

def plot_clusters_performance(rank_by='score', ix=-1, mean=True, **kwargs):
    df = get_clean_dataset()
    d = df.groupby(level=0)['slevel'].agg(['count', 'mean', 'std'])
    nb_kmean = 2
    kmeans = KMeans(n_clusters=nb_kmean).fit(d)


    def get_mse_and_best_arch(x):
        if mean:
            return x.groupby('arch')[rank_by_col].mean().sort_values().reset_index().iloc[0,:]
        else:
            return x.sort_values(by=rank_by_col).loc[:,[rank_by_col,'arch']].iloc[0,:]

    df_exp, rank_by_col = order_exp_by(ix=ix, rank_by=rank_by, **kwargs)
    per_user_best = df_exp.groupby('user').apply(get_mse_and_best_arch) 
    per_user_best = per_user_best[[rank_by_col,'arch']]
    per_user_best.arch = per_user_best.arch.str.upper()
    d = pd.concat([d, per_user_best], axis=1)
    d.columns = ['Cantidad buckets', 'Promedio MET','Desviacion Estándar MET', 'MSE', 'Arquitectura']
    colors={'RNN': 'b', 'TCN': 'g', 'CNN': 'r', 'MLP': 'm'}

    g = sns.relplot(x='Cantidad buckets',
                    y='Promedio MET',
                    hue='Arquitectura',
                    size='MSE',
                    sizes=(100, 500),
                    alpha=.6,
                    data=d,
                    palette=colors)

    g.ax.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='r', marker='x')

    plt.show()

def plot_clusters_performance_both_aggr_func(rank_by='score', ix=-1, mean=True, **kwargs):
    AGGREGATION = 'Función de agregación:'

    df = get_clean_dataset()
    d = df.groupby(level=0)['slevel'].agg(['count', 'mean', 'std'])
    nb_kmean = 2
    kmeans = KMeans(n_clusters=nb_kmean).fit(d)


    def get_mse_and_best(x):
        if mean:
            return x.groupby('arch')[rank_by_col].mean().sort_values().reset_index().iloc[0,:]
        else:
            return x.sort_values(by=rank_by_col).loc[:,[rank_by_col,'arch']].iloc[0,:]

    df_exp, rank_by_col = order_exp_by(ix=ix, rank_by=rank_by, **kwargs)
    per_user_best = df_exp.groupby('user').apply(get_mse_and_best) 
    per_user_best = per_user_best[[rank_by_col,'arch']]
    per_user_best.arch = per_user_best.arch.str.upper()
    d = pd.concat([d, per_user_best], axis=1)
    d.columns = ['Cantidad buckets', 'Promedio MET','Desviacion Estándar MET', 'MSE', 'Arquitectura']
    colors={'RNN': 'b', 'TCN': 'g', 'CNN': 'r', 'MLP': 'm'}

    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12,5), sharey=True)

    g1 = sns.scatterplot(x='Cantidad buckets',
                    y='Promedio MET',
                    hue='Arquitectura',
                    size='MSE',
                    sizes=(100, 500),
                    alpha=.6,
                    data=d,
                    palette=colors,
                    ax=ax1)
    ax1.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='r', marker='x')
    ax1.set_title(AGGREGATION + ' Media')


    mean=False
    per_user_best = df_exp.groupby('user').apply(get_mse_and_best) 
    per_user_best = per_user_best[[rank_by_col,'arch']]
    per_user_best.arch = per_user_best.arch.str.upper()
    d = df.groupby(level=0)['slevel'].agg(['count', 'mean', 'std'])
    d = pd.concat([d, per_user_best], axis=1)
    d.columns = ['Cantidad buckets', 'Promedio MET','Desviacion Estándar MET', 'MSE', 'Arquitectura']


    g = sns.scatterplot(x='Cantidad buckets',
                    y='Promedio MET',
                    hue='Arquitectura',
                    size='MSE',
                    sizes=(100, 500),
                    alpha=.6,
                    data=d,
                    palette=colors,
                    ax=ax2)
    ax2.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='r', marker='x')
    ax2.set_title(AGGREGATION + ' Mínima')

    h,l = g1.get_legend_handles_labels()
    ax1.legend(h[5:], l[5:], bbox_to_anchor=(1, 1), borderpad=1);
    ax2.legend(bbox_to_anchor=(1., 1), borderpad=1);

    fig.tight_layout() 
    plt.show()

def plot_clusters_performance_both_aggr_func_lags(rank_by='score', ix=-1, mean=True, **kwargs):
    AGGREGATION = 'Función de agregación:'

    df = get_clean_dataset()
    d = df.groupby(level=0)['slevel'].agg(['count', 'mean', 'std'])
    nb_kmean = 2
    kmeans = KMeans(n_clusters=nb_kmean).fit(d)


    def get_mse_and_best(x):
        if mean:
            return x.groupby('nb_lags')[rank_by_col].mean().sort_values().reset_index().iloc[0,:]
        else:
            return x.sort_values(by=rank_by_col).loc[:,[rank_by_col,'nb_lags']].iloc[0,:]

    df_exp, rank_by_col = order_exp_by(ix=ix, rank_by=rank_by, **kwargs)
    per_user_best = df_exp.groupby('user').apply(get_mse_and_best) 
    per_user_best = per_user_best[[rank_by_col,'nb_lags']]
    d = pd.concat([d, per_user_best], axis=1)
    d.columns = ['Cantidad buckets', 'Promedio MET','Desviacion Estándar MET', 'MSE', 'Lags']
    colors={1: 'b', 2: 'g', 4: 'r', 8: 'm'}

    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12,5), sharey=True)

    g1 = sns.scatterplot(x='Cantidad buckets',
                    y='Promedio MET',
                    hue='Lags',
                    size='MSE',
                    sizes=(100, 500),
                    alpha=.6,
                    data=d,
                    palette=colors,
                    ax=ax1)
    ax1.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='r', marker='x')
    ax1.set_title(AGGREGATION + ' Media')


    mean=False
    per_user_best = df_exp.groupby('user').apply(get_mse_and_best) 
    per_user_best = per_user_best[[rank_by_col,'nb_lags']]
    d = df.groupby(level=0)['slevel'].agg(['count', 'mean', 'std'])
    d = pd.concat([d, per_user_best], axis=1)
    d.columns = ['Cantidad buckets', 'Promedio MET','Desviacion Estándar MET', 'MSE', 'Lags']


    g = sns.scatterplot(x='Cantidad buckets',
                    y='Promedio MET',
                    hue='Lags',
                    size='MSE',
                    sizes=(100, 500),
                    alpha=.6,
                    data=d,
                    palette=colors,
                    ax=ax2)
    ax2.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='r', marker='x')
    ax2.set_title(AGGREGATION + ' Mínima')

    h,l = g1.get_legend_handles_labels()
    ax1.legend(h[5:], l[5:], bbox_to_anchor=(1, 1), borderpad=1);
    ax2.legend(bbox_to_anchor=(1., 1), borderpad=1);

    fig.tight_layout() 
    plt.show()

def plot_clusters_performance_without_arch(rank_by='score', ix=-1, **kwargs):
    df = get_clean_dataset()
    d = df.groupby(level=0)['slevel'].agg(['count', 'mean', 'std'])
    nb_kmean = 2
    kmeans = KMeans(n_clusters=nb_kmean).fit(d)


    def get_mse_and_best_arch(x):
        return x.sort_values(by=rank_by_col).loc[:,[rank_by_col,'arch']].iloc[0,:]

    df_exp, rank_by_col = order_exp_by(ix=ix, rank_by=rank_by, **kwargs)
    per_user_best = df_exp.groupby('user').apply(get_mse_and_best_arch) 
    per_user_best.arch = per_user_best.arch.str.upper()
    d = pd.concat([d, per_user_best], axis=1)

    df = get_clean_dataset()
    e = df.groupby(level=0)['slevel'].agg(['count', 'mean', 'std'])
    nb_kmean = 2
    kmeans = KMeans(n_clusters=nb_kmean).fit(e)
    y = 'Grupo ' + pd.Series(kmeans.predict(e).astype('str'))
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, e)
    for i in closest:
        y[i] = 'Usuario seleccionado'
    y.index = d.index
    y = y.to_frame('y')
    d = pd.concat([d, y], axis=1)

    d.columns = ['Cantidad buckets', 'Promedio MET','Desviacion Estándar MET', 'MSE', 'Arquitectura', 'Grupo']

    g = sns.relplot(x='Cantidad buckets',
                    y='Promedio MET',
                    size='MSE',
                    hue='Grupo',
                    sizes=(100, 500),
                    alpha=.6,
                    data=d)

    to_annotate = d.loc[(d['Grupo']=='Usuario seleccionado'),:].reset_index(drop=False).iloc[:, :3].values

    style = dict(size=10, color='black')

    for i in range(to_annotate.shape[0]):
        g.ax.annotate(int(to_annotate[i, 0]),
                      xy=(to_annotate[i, 1],
                          to_annotate[i, 2]),
                      ha='center',
                      va='center',
                      **style)
    g.ax.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='r', marker='x')

    plt.show()

def plot_clusters_performance_without_arch_both_aggr_func(rank_by='score', ix=-1, **kwargs):
    AGGREGATION = 'Función de agregación:'
    df = get_clean_dataset()
    d = df.groupby(level=0)['slevel'].agg(['count', 'mean', 'std'])
    nb_kmean = 2
    kmeans = KMeans(n_clusters=nb_kmean).fit(d)


    def get_mse_and_best_arch(x):
        if mean:
            return x.groupby('arch')[rank_by_col].mean().sort_values().reset_index().iloc[0,:]
        else:
            return x.sort_values(by=rank_by_col).loc[:,[rank_by_col,'arch']].iloc[0,:]

    df_exp, rank_by_col = order_exp_by(ix=ix, rank_by=rank_by, **kwargs)
    per_user_best = df_exp.groupby('user')[rank_by_col].mean().to_frame()
    per_user_best.columns = [rank_by_col]
    d = pd.concat([d, per_user_best], axis=1)

    df = get_clean_dataset()
    e = df.groupby(level=0)['slevel'].agg(['count', 'mean', 'std'])
    nb_kmean = 2
    kmeans = KMeans(n_clusters=nb_kmean).fit(e)
    y = 'Grupo ' + pd.Series(kmeans.predict(e).astype('str'))
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, e)
    for i in closest:
        y[i] = 'Usuario modelo'
    y.index = d.index
    y = y.to_frame('y')
    d = pd.concat([d, y], axis=1)
    print(d)
    d.columns = ['Cantidad buckets', 'Promedio MET','Desviacion Estándar MET', 'MSE', 'Grupo']
    print(d.MSE.max())
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12,5), sharey=True)


    g1 = sns.scatterplot(x='Cantidad buckets',
                    y='Promedio MET',
                    size='MSE',
                    hue='Grupo',
                    sizes=(100, 500),
                    alpha=.6,
                    data=d,
                    ax=ax1)

    to_annotate = d.loc[(d['Grupo']=='Usuario modelo'),:].reset_index(drop=False).iloc[:, :3].values

    style = dict(size=10, color='black')

    for i in range(to_annotate.shape[0]):
        ax1.annotate(int(to_annotate[i, 0]),
                        xy=(to_annotate[i, 1],
                            to_annotate[i, 2]),
                        ha='center',
                        va='center',
                        **style)
    ax1.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='r', marker='x')
    ax1.set_title(AGGREGATION + ' Media')

    mean = False
    d = df.groupby(level=0)['slevel'].agg(['count', 'mean', 'std'])
    nb_kmean = 2
    kmeans = KMeans(n_clusters=nb_kmean).fit(d)
    df_exp, rank_by_col = order_exp_by(ix=ix, rank_by=rank_by)
    per_user_best = df_exp.groupby('user')[rank_by_col].min().to_frame()
    per_user_best.columns = [rank_by_col]
    d = pd.concat([d, per_user_best], axis=1)


    df = get_clean_dataset()
    e = df.groupby(level=0)['slevel'].agg(['count', 'mean', 'std'])
    nb_kmean = 2
    kmeans = KMeans(n_clusters=nb_kmean).fit(e)
    y = 'Grupo ' + pd.Series(kmeans.predict(e).astype('str'))
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, e)
    for i in closest:
        y[i] = 'Usuario modelo'
    y.index = d.index
    y = y.to_frame('y')
    d = pd.concat([d, y], axis=1)

    d.columns = ['Cantidad buckets', 'Promedio MET','Desviacion Estándar MET', 'MSE', 'Grupo']
    print(d.MSE.max())

    g = sns.scatterplot(x='Cantidad buckets',
                    y='Promedio MET',
                    size='MSE',
                    hue='Grupo',
                    sizes=(100, 500),
                    alpha=.6,
                    data=d,
                    ax=ax2)

    to_annotate = d.loc[(d['Grupo']=='Usuario modelo'),:].reset_index(drop=False).iloc[:, :3].values

    style = dict(size=10, color='black')

    for i in range(to_annotate.shape[0]):
        ax2.annotate(int(to_annotate[i, 0]),
                        xy=(to_annotate[i, 1],
                            to_annotate[i, 2]),
                        ha='center',
                        va='center',
                        **style)
    ax2.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='r', marker='x')
    ax2.set_title(AGGREGATION + ' Mínima')

    h,l = g1.get_legend_handles_labels()
    ax1.legend(h[4:], l[4:], bbox_to_anchor=(1, 1), borderpad=1);
    ax2.legend(bbox_to_anchor=(1., 1), borderpad=1);
    fig.tight_layout(h_pad=.3) 

    plt.show()

def plot_clusters_performance_without_arch_min_aggr_func(rank_by='score', ix=-1, **kwargs):
    AGGREGATION = 'Función de agregación:'
    df = get_clean_dataset()
    d = df.groupby(level=0)['slevel'].agg(['count', 'mean', 'std'])
    nb_kmean = 2
    kmeans = KMeans(n_clusters=nb_kmean).fit(d)

    df_exp, rank_by_col = order_exp_by(ix=ix, rank_by=rank_by, **kwargs)
    per_user_best = df_exp.groupby('user')[rank_by_col].mean().to_frame()
    per_user_best.columns = [rank_by_col]
    d = pd.concat([d, per_user_best], axis=1)
    fig, (ax2) = plt.subplots(1, 1, figsize=(6,5), sharey=True)
    d = df.groupby(level=0)['slevel'].agg(['count', 'mean', 'std'])
    nb_kmean = 2
    kmeans = KMeans(n_clusters=nb_kmean).fit(d)
    df_exp, rank_by_col = order_exp_by(ix=ix, rank_by=rank_by)
    per_user_best = df_exp.groupby('user')[rank_by_col].min().to_frame()
    per_user_best.columns = [rank_by_col]
    d = pd.concat([d, per_user_best], axis=1)


    df = get_clean_dataset()
    e = df.groupby(level=0)['slevel'].agg(['count', 'mean', 'std'])
    nb_kmean = 2
    kmeans = KMeans(n_clusters=nb_kmean).fit(e)
    y = 'Grupo ' + pd.Series(kmeans.predict(e).astype('str'))
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, e)
    for i in closest:
        y[i] = 'Usuario modelo'
    y.index = d.index
    y = y.to_frame('y')
    d = pd.concat([d, y], axis=1)

    d.columns = ['Cantidad buckets', 'Promedio MET','Desviacion Estándar MET', 'MSE', 'Grupo']
    print(d.MSE.max())

    g = sns.scatterplot(x='Cantidad buckets',
                    y='Promedio MET',
                    size='MSE',
                    hue='Grupo',
                    sizes=(100, 500),
                    alpha=.6,
                    data=d,
                    ax=ax2)

    to_annotate = d.loc[(d['Grupo']=='Usuario modelo'),:].reset_index(drop=False).iloc[:, :3].values

    style = dict(size=10, color='black')

    for i in range(to_annotate.shape[0]):
        ax2.annotate(int(to_annotate[i, 0]),
                        xy=(to_annotate[i, 1],
                            to_annotate[i, 2]),
                        ha='center',
                        va='center',
                        **style)
    ax2.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='r', marker='x')
    ax2.set_title(AGGREGATION + ' Mínima')

    ax2.legend(bbox_to_anchor=(1., 1), borderpad=1);

    plt.show()

def plot_per_cluster_mse_diff_poi():
    
    ix=-1
    rank_by='score'

    df = get_clean_dataset()
    d = df.groupby(level=0)['slevel'].agg(['count', 'mean', 'std'])
    nb_kmean = 2
    kmeans = KMeans(n_clusters=nb_kmean).fit(d)

    s, r = rank_results(comp_col='poi', based_on='user', rank_by='score')    
    l = list(r.best_rank_by.apply(list).values)
    print(l)
    l_zipped = list(zip(*l))
    print(l_zipped)
    best_poi = np.array(l_zipped[0])
    worst_poi = np.array(l_zipped[1])
    diff = worst_poi - best_poi 
    poi = r["Puesto 1"]
    d['MSE'] = diff
    d['POI'] = poi

    d.loc[d.POI=='per','POI'] = 'Personal'
    d.loc[d.POI=='imp','POI'] = 'Impersonal'
    print(diff)
    colors = {'Personal': 'r', 'Impersonal': 'b'}

    d.columns = ['Cantidad buckets', 'Promedio MET','Desviacion Estándar MET', 'Diferencia de MSE', 'POI']

    g = sns.relplot(x='Cantidad buckets',
                    y='Promedio MET',
                    hue='POI',
                    size='Diferencia de MSE',
                    sizes=(50, 300),
                    alpha=.6,
                    data=d,
                    palette=colors)

    g.ax.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='r', marker='x')

    plt.show()

def plot_per_cluster_mse_diff_poi_both_aggr_func():
    AGGREGATION = 'Función de agregación:'
    df = get_clean_dataset()
    d = df.groupby(level=0)['slevel'].agg(['count', 'mean', 'std'])
    nb_kmean = 2
    kmeans = KMeans(n_clusters=nb_kmean).fit(d)
    s, r = rank_results_agg_func(comp_col='poi', based_on='user', rank_by='score', agg_func='mean')    
    l = list(r.best_rank_by.apply(list).values)
    l_zipped = list(zip(*l))
    best_poi = np.array(l_zipped[0])
    worst_poi = np.array(l_zipped[1])
    diff = worst_poi - best_poi 
    poi = r["Puesto 1"]
    d['MSE'] = diff
    d['POI'] = poi

    d.loc[d.POI=='per','POI'] = 'Personal'
    d.loc[d.POI=='imp','POI'] = 'Impersonal'
    colors = {'Personal': 'r', 'Impersonal': 'b'}

    d.columns = ['Cantidad buckets', 'Promedio MET','Desviacion Estándar MET', 'Dif. MSE', 'Naturaleza']


    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12,5), sharey=True)

    g1 = sns.scatterplot(x='Cantidad buckets',
                    y='Promedio MET',
                    hue='Naturaleza',
                    size='Dif. MSE',
                    sizes=(50, 300),
                    alpha=.6,
                    data=d,
                    palette=colors,
                    ax=ax1)

    ax1.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='r', marker='x')
    ax1.set_title(AGGREGATION + ' Media')


    d = df.groupby(level=0)['slevel'].agg(['count', 'mean', 'std'])
    nb_kmean = 2
    kmeans = KMeans(n_clusters=nb_kmean).fit(d)
    s, r = rank_results_agg_func(comp_col='poi', based_on='user', rank_by='score', agg_func='min')    
    l = list(r.best_rank_by.apply(list).values)
    l_zipped = list(zip(*l))
    best_poi = np.array(l_zipped[0])
    worst_poi = np.array(l_zipped[1])
    diff = worst_poi - best_poi 
    poi = r["Puesto 1"]
    d['MSE'] = diff
    d['POI'] = poi

    d.loc[d.POI=='per','POI'] = 'Personal'
    d.loc[d.POI=='imp','POI'] = 'Impersonal'
    colors = {'Personal': 'r', 'Impersonal': 'b'}

    d.columns = ['Cantidad buckets', 'Promedio MET','Desviacion Estándar MET', 'Dif. MSE', 'Naturaleza']

    g = sns.scatterplot(x='Cantidad buckets',
                    y='Promedio MET',
                    hue='Naturaleza',
                    size='Dif. MSE',
                    sizes=(50, 300),
                    alpha=.6,
                    data=d,
                    palette=colors,
                    ax=ax2)
    ax2.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='r', marker='x')
    ax2.set_title(AGGREGATION + ' Mínima')

    h,l = g1.get_legend_handles_labels()
    ax1.legend(h[3:], l[3:], bbox_to_anchor=(1, 1), borderpad=1);
    ax2.legend(bbox_to_anchor=(1., 1), borderpad=1);

    plt.tight_layout(w_pad=1)

    plt.show()

def plot_clusters_performance_by_lags(rank_by='score', ix=-1, **kwargs):
    df = get_clean_dataset()
    d = df.groupby(level=0)['slevel'].agg(['count', 'mean', 'std'])
    nb_kmean = 2
    kmeans = KMeans(n_clusters=nb_kmean).fit(d)

    def get_mse_and_best_arch(x):
        return x.sort_values(by=rank_by_col).loc[:,[rank_by_col,'nb_lags']].iloc[0,:]

    df_exp, rank_by_col = order_exp_by(ix=ix, rank_by=rank_by, **kwargs)
    per_user_best = df_exp.groupby('user').apply(get_mse_and_best_arch) 
    per_user_best.nb_lags = per_user_best.nb_lags.apply(str)
    d = pd.concat([d, per_user_best], axis=1)
    d.columns = ['Cantidad buckets', 'Promedio MET','Desviacion Estándar MET', 'MSE', 'Arquitectura']

    g = sns.relplot(x='Cantidad buckets',
                    y='Promedio MET',
                    hue='Arquitectura',
                    size='MSE',
                    sizes=(100, 500),
                    alpha=.6,
                    data=d)

    g.ax.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='r', marker='x')

    plt.show()

#######################


def compute_from_distribution_division(comp_col, division_values, **kwargs):
    for col_value in filter_exp()[comp_col].unique():
        df = filter_exp(**kwargs)
        df = df.loc[df[comp_col]==col_value]
        a = df.mean_score.quantile(division_values).values
        #print(a)x
        r[col_value] = a
    return r.T


def get_quartiles(comp_col='arch', **kwargs):
    """Get the decil of the results of a set of experiments.

    Return a DataFrame with as many rows as the number of unique values of comp_col

    Example: if comp_col is arch, a DataFrame with 4 rows is returned where each row
    has the deciles of the experiment that correspond to that value of the comp_col

    Args:
        comp_col (str, optional). Defaults to 'arch'.

    Returns:
        _type_: _description_
    """
    quartiles = [0,.25,.5,.75,1.]
    r = pd.DataFrame(index=quartiles)  
    return compute_from_distribution_division(comp_col, quartiles, **kwargs)


def get_delices(comp_col='arch', **kwargs):
    """Get the decil of the results of a set of experiments.

    Return a DataFrame with as many rows as the number of unique values of comp_col

    Example: if comp_col is arch, a DataFrame with 4 rows is returned where each row
    has the deciles of the experiment that correspond to that value of the comp_col

    Args:
        comp_col (str, optional). Defaults to 'arch'.

    Returns:
        _type_: _description_
    """
    deciles = [.1 * i for i in range(11)]
    return compute_from_distribution_division(comp_col, deciles, **kwargs)


def plot_distribution(data1, text1, data2, text2, rank_by, comp_col, poi, user="all"):
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    mu, sigma = stats.norm.fit(data1)
    # Theoretical values for normal distribution in the observed range
    x_hat = np.linspace(min(data1), max(data1), num=100)
    y_hat = stats.norm.pdf(x_hat, mu, sigma)
    # Distribution plot
    axs[0, 0].plot(x_hat, y_hat, linewidth=2, label='normal')
    axs[0, 0].hist(x=data1, density=True, bins=20, color="#3182bd", alpha=0.5)
    axs[0, 0].plot(data1, np.full_like(data1, -0.01), '|k', markeredgewidth=1)
    axs[0, 0].set_title('Distribution for ' + str(text1))
    axs[0, 0].set_xlabel(rank_by)
    axs[0, 0].set_ylabel('Probability density')
    axs[0, 0].legend()
    # qq-plot
    pg.qqplot(data1, dist='norm', ax=axs[0, 1])

    mu, sigma = stats.norm.fit(data2)
    x_hat = np.linspace(min(data2), max(data2), num=100)
    y_hat = stats.norm.pdf(x_hat, mu, sigma)
    axs[1, 0].plot(x_hat, y_hat, linewidth=2, label='normal')
    axs[1, 0].hist(x=data2, density=True, bins=20, color="#3182bd", alpha=0.5)
    axs[1, 0].plot(data2, np.full_like(data2, -0.01), '|k', markeredgewidth=1)
    axs[1, 0].set_title('Distribution for ' + str(text2))
    axs[1, 0].set_xlabel(rank_by)
    axs[1, 0].set_ylabel('Probability density')
    axs[1, 0].legend()

    pg.qqplot(data2, dist='norm', ax=axs[1, 1])

    plt.tight_layout()
    plt.savefig(
        '../figures/' + poi + '/distribution-' + comp_col + '-' + str(text1) + '-' + str(text2) + '-rankedby-' + str(
            rank_by) + '-user-' + str(user) + '.png')
    plt.close(fig)


def plotBoxplot(data1, text1, data2, text2, rank_by, comp_col, poi, user="all"):
    # Plot boxplot
    # ==============================================================================
    data = pd.Series(data1, name=text1).to_frame().join(pd.Series(data2, name=text2))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 3.5))
    sns.boxplot(
        data=data,
        palette='tab10',
        ax=ax
    )
    ax.set_title(f'Distribution for {rank_by} by {comp_col}')
    ax.set_xlabel(comp_col)
    ax.set_ylabel(rank_by)
    plt.savefig('../figures/' + poi + '/boxplot-' + comp_col + '-' + str(text1) + '-' + str(text2) + '-rankedby-' + str(
        rank_by) + '-user-' + str(user) + '.png')
    plt.close(fig)


def get_p_value(comp_col='arch', v1='tcn', v2='mlp', rank_by='score', decil=None, generateFigures=False, **kwargs):
    """Runs statistic tests

    Args:
        comp_col (str, optional): comparison columns. Defaults to 'arch'.
        v1 (str, optional): value 1 of comp_col to compare. Defaults to 'tcn'.
        v2 (str, optional): value 2 of comp_col to compare. Defaults to 'mlp'.
        rank_by (str, optional): the value to be compared and used in the tests. Defaults to 'score'.
        decil (int, optional): the decil to be used to run the test. Defaults to None.
        generateFigures (bool, optional). Defaults to False.

    Returns:
        bool: whether statistical significance was found or not
    """


    if 'user' in kwargs:
        user = kwargs['user']
    else:
        user = "all"

    alpha = 0.05

    print('*' * 48)
    print(f'Comparing values {v1} and {v2} from {comp_col} category, ranked by {rank_by} con **kwargs={kwargs}\n')

    exps = filter_exp(**kwargs)

    if rank_by in ['score', 'time']:
        rank_by = f'mean_{rank_by}'

    v1_exps_df = exps.loc[exps[comp_col] == v1].loc[:, rank_by]
    v1_exps = v1_exps_df.dropna().values
    v2_exps_df = exps.loc[exps[comp_col] == v2].loc[:, rank_by]
    v2_exps = v2_exps_df.dropna().values

    if decil != None:
        decil_per_arch = get_deciles(comp_col)
        decil_value = decil_per_arch.filter(items=[v1], axis=0).iloc[0, decil]
        v1_exps = v1_exps[v1_exps < decil_value]
        decil_value = decil_per_arch.filter(items=[v2], axis=0).iloc[0, decil]
        v2_exps = v2_exps[v2_exps < decil_value]

    print('Sample size of ' + str(v1) + ': ' + str(len(v1_exps)))
    print('Sample size of ' + str(v2) + ': ' + str(len(v2_exps)))

    if len(v1_exps) <= 3 or len(v2_exps) <= 3:
        print("NOT ENOUGH DATA AVAILABLE")
        return False;

    if generateFigures:
        plot_distribution(v1_exps, v1, v2_exps, v2, rank_by, comp_col, kwargs['poi'], user)

    v1_mean = np.mean(v1_exps)
    v2_mean = np.mean(v2_exps)

    v1_median = np.median(v1_exps)
    v2_median = np.median(v2_exps)

    v1_std = np.std(v1_exps)
    v2_std = np.std(v2_exps)

    print(f'mean {v1}: {v1_mean}')
    print(f'mean {v2}: {v2_mean}\n')
    print(f'median {v1}: {v1_median}')
    print(f'median {v2}: {v2_median}\n')
    print(f'std {v1}: {v1_std}')
    print(f'std {v2}: {v2_std}\n')

    print('*' * 31)
    print("Shapiro-Wilk test for normality")
    print('*' * 31)
    shapiro1 = shapiro(v1_exps)[1]
    shapiro2 = shapiro(v2_exps)[1]

    v1_is_normal = shapiro1 > alpha
    v2_is_normal = shapiro2 > alpha

    normality_v1 = 'normal' if v1_is_normal else 'not normal'
    normality_v2 = 'normal' if v2_is_normal else 'not normal'

    print(f'Shapiro-Wilk test result for {v1}: {shapiro1} ({normality_v1})')
    print(f'Shapiro-Wilk test result for {v2}: {shapiro2} ({normality_v2})')
    print('')

    print('*' * 31)
    print("Levene test for equal variances")
    print('*' * 31)
    levene_test = stats.levene(v1_exps, v2_exps, center='median')
    equal_variances = levene_test[1] > alpha
    homoscedasticity = 'Homoscedasticity' if equal_variances else 'Heteroscedasticity'
    print(f'{levene_test} ({homoscedasticity})')
    if generateFigures:
        plotBoxplot(v1_exps, v1, v2_exps, v2, rank_by, comp_col, kwargs['poi'], user)
    print('')

    if len(v1_exps) > 30 or len(v2_exps) > 30:
        print('*' * 13)
        print("T-Test result")
        print('*' * 13)
        test_result = ttest_ind(v1_exps, v2_exps, nan_policy="omit", equal_var=equal_variances)
        print(test_result)
    else:
        print('*' * 13)
        print("U-Test result")
        print('*' * 13)
        test_result = mannwhitneyu(v1_exps, v2_exps, method="exact", nan_policy="omit")
        print(test_result)

    sig_dif = test_result[1] < alpha
    if sig_dif:
        print('Statistical significance found')
    else:
        print('Statistical significance NOT found')

    print('')
    print(f'{v1} NaN values: {v1_exps_df.isna().sum()} de {v1_exps_df.count()}')
    print(f'{v2} NaN values: {v2_exps_df.isna().sum()} de {v2_exps_df.count()}\n')
    print('')
    return sig_dif


#######################################################
# Functions that use get_p_value() to compare different
# aspects of the experiments
########################################################

def compare_prototype_users(generateFigures=False):
    df = get_experiments_data()

    low_met_users = [u for u in df.loc[df.centroid == 'low_met'].user.unique() if u != 34]
    high_met_users = [u for u in df.loc[df.centroid == 'high_met'].user.unique() if u != 32]
    print('x' * 100)
    print('Low MET')
    i = 0
    j = 0
    for u in low_met_users:
        if get_p_value('user', 34, u, poi='imp', generateFigures=generateFigures):
            j += 1
        i += 1
    print(f'{j} of {i} with statistical difference')
    i = 0
    j = 0
    print('x' * 100)
    print('High MET')
    for u in high_met_users:
        if get_p_value('user', 32, u, poi='imp', generateFigures=generateFigures):
            j += 1
        i += 1
    print(f'{j} of {i} with statistical difference')


def compare_architectures(metric='score', poi='imp', generateFigures=False):
    archs = ['cnn', 'mlp', 'rnn', 'tcn']
    if poi == 'imp':
        for i in archs:
            for j in archs:
                if i < j:
                    get_p_value(comp_col='arch', v1=i, v2=j, poi='imp', decil=6, rank_by=metric, generateFigures=generateFigures)
    else:
        users = get_list_of_users()
        for u in users:
            for i in archs:
                for j in archs:
                    if i < j:
                        # get_p_value(comp_col='arch', v1=i, v2=j, poi='imp', verbose=2, force_test="student", decil=6)
                        get_p_value(comp_col='arch', v1=i, v2=j, poi='per', user=u, decil=6, rank_by=metric, generateFigures=generateFigures)


def compare_lags(metric='score', poi='imp', generateFigures=False):
    if poi == 'imp':
        for nb1 in [1, 2, 4, 8]:
            for nb2 in [1, 2, 4, 8]:
                if nb1 < nb2:
                    get_p_value(comp_col='nb_lags', v1=nb1, v2=nb2, poi='imp', rank_by=metric, generateFigures=generateFigures)
    else:
        users = get_list_of_users()
        for u in users:
            for nb1 in [1, 2, 4, 8]:
                for nb2 in [1, 2, 4, 8]:
                    if nb1 < nb2:
                        get_p_value(comp_col='nb_lags', v1=nb1, v2=nb2, poi='per', user=u, rank_by=metric, generateFigures=generateFigures)


def compare_granularity(metric='score', poi='imp', generateFigures=False):
    if poi == 'imp':
        get_p_value(comp_col='gran', v1=30, v2=60, rank_by=metric, poi='imp', generateFigures=generateFigures)
    else:
        users = get_list_of_users()
        for u in get_list_of_users():
            get_p_value(comp_col='gran', v1=30, v2=60, rank_by=metric, poi='per', user=u, generateFigures=generateFigures)
