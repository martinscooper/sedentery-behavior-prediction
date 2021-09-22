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
    # este codigo compara los datos de y_test de los experimentos y 
    # los que estan en el dataset, para ver si concuerdan
    # en algunos hay una diferencia de uno, pero nada mas
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
    # falla si solo hay un caso
    l[1] = [np.squeeze(a) for a in l[1]]
    y_test = np.concatenate(l[0]) 
    y_pred = np.concatenate(l[1])
    shapes = [arr.shape[0] for arr in l[0]]
    if return_shapes: 
        return y_test, y_pred, shapes
    else: 
        return y_test, y_pred

def print_results(fromi=1, toi=5, archs=['rnn', 'tcn', 'cnn', 'mlp'], poi='per', user=32, lags=1, period=1, gran=60):
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
    plt.ylabel('Tiempo')
    plt.xlabel('Iteración')

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
    plt.xlabel('Iteración')

    handles = [lines.Line2D([], [], color=c,
                            markersize=15, label=k.upper()) for k,c in archs_colors.items()]
    plt.legend(loc='upper left', handles=handles)

    plt.show()

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

def get_p_value(comp_col='arch', v1='tcn', v2='mlp', rank_by='score', force_test = None, treshhold=1, verbose=1, decil=None, print_always=True, **kwargs):
    exps = filter_exp(**kwargs)
    if rank_by in ['score','time']:
        rank_by = f'mean_{rank_by}'
    
    v1_exps_df = exps.loc[exps[comp_col]==v1].loc[:,rank_by]
    v1_exps = v1_exps_df.dropna().values
    v2_exps_df = exps.loc[exps[comp_col]==v2].loc[:,rank_by]
    v2_exps = v2_exps_df.dropna().values

    if decil != None: 
        decil_per_arch = quantile_decil(comp_col)
        decil_value = decil_per_arch.filter(items=[v1], axis=0).iloc[0,decil]
        v1_exps = v1_exps[v1_exps < decil_value]
        decil_value = decil_per_arch.filter(items=[v2], axis=0).iloc[0,decil]
        v2_exps = v2_exps[v2_exps < decil_value]

    wilcoxon_diff = (v1_exps_df.values - v2_exps_df.values)
    wilcoxon_diff = wilcoxon_diff[~np.isnan(wilcoxon_diff)]

    v1_mean = np.mean(v1_exps)
    v2_mean = np.mean(v2_exps) 

    v1_median= np.median(v1_exps)
    v2_median = np.median(v2_exps) 

    v1_std = np.std(v1_exps) 
    v2_std = np.std(v2_exps)

        
    alpha = 0.05

    v1_is_normal = shapiro(v1_exps)[1] > alpha
    v2_is_normal = shapiro(v2_exps)[1] > alpha

    if (v1_is_normal and v2_is_normal) or force_test == "student":
        performed_test = 'Performing Student test'
        statistic_diff = ttest_ind(v1_exps,v2_exps, nan_policy="omit")
    else:
        performed_test = 'Performing Wilcoxon test'
        statistic_diff = wilcoxon(wilcoxon_diff, alternative='less')

    if statistic_diff[1] < alpha:
        show_results = True
    else: 
        show_results = False

    if (print_always or show_results) and verbose>0 :
        print('*' * 16)
        print(f'Comparing values {v1} and {v2} from {comp_col} category, ranked by {rank_by} con **kwargs={kwargs}\n')
        if show_results: print('///Diferencia sifnificativa hallada///')
        print(performed_test)
        print(statistic_diff)
        print('')
        
        if verbose>1: 
            print(f'{v1} Valores NaN: {v1_exps_df.isna().sum()} de {v1_exps_df.count()}')
            print(f'{v2} Valores NaN: {v2_exps_df.isna().sum()} de {v2_exps_df.count()}\n')

            print(f'mean {v1}: {v1_mean}')
            print(f'mean {v2}: {v2_mean}\n')
            print(f'median {v1}: {v1_median}')
            print(f'median {v2}: {v2_median}\n')
            print(f'std {v1}: {v1_std}')
            print(f'std {v2}: {v2_std}\n')
            print('\n')
        if verbose > 2: 
            v1_range = (v1_mean - min(2 * v1_std, 1.5), v1_mean + min(2 * v1_std, 1.5))
            v2_range = (v2_mean - min(2 * v2_std, 1.5), v2_mean + min(2 * v2_std, 1.5))
            if v1_mean > v2_mean: 
                range = v2_range
            else: range = v1_range 
            plt.hist(v1_exps, bins=25, color='b', edgecolor='k', alpha=0.5, label=str.upper(v1), range=range)
            plt.hist(v2_exps, bins=25, color='r', edgecolor='k', alpha=0.5, label=str.upper(v2), range=range)
            plt.legend()
            plt.show()
        print('')
    return show_results

def quantile_decil(comp_col='arch', **kwargs):
    deciles = [.1 * i for i in range(11)]
    quartiles = [0,.25,.5,.75,1.]
    r = pd.DataFrame(index=deciles)  

    for col_value in filter_exp()[comp_col].unique():
        df = filter_exp(**kwargs)
        df = df.loc[df[comp_col]==col_value]
        a = df.mean_score.quantile(deciles).values
        #print(a)x
        r[col_value] = a
    return r.T