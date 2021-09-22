import numpy as np
import pandas as pd

from skopt import load
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import seaborn as sns
from preprocessing.datasets import get_clean_dataset
from matplotlib import pyplot as plt
sns.set_style("whitegrid")


def get_tunning_results(arch):
   
    cnn_hypers = ['Nr. filtros', 'Tam. kernel', 'Dropout (conv)', 'Nr. nodos densos', 'Dropout (denso)', 'Nr. épocas', 'Tam. bache', 'MSE']
    mlp_hypers = ['Nr. nodos', 'Nr. capas', 'Usar Batch Normalization', 'Dropout', 'Nr. épocas', 'Tam. bache', 'MSE']
    tcn_hypers = ['Nr. filtros', 'Tam. kernel', 'Dropout', 'Omitir conexiones', 'Usar Batch Normalization', 'Nr. épocas', 'Tam. bache', 'MSE']
    rnn_hypers = ['Nr. capas', 'Nr. unidades LSTM', 'Dropout (lstm)', 'Nr nodos densos', 'Dropout (denso)','Nr. épocas', 'Tam. bache', 'MSE']
    h = {
        'cnn' : cnn_hypers,
        'tcn': tcn_hypers,
        'mlp': mlp_hypers,
        'rnn': rnn_hypers
    }
   
    users = [34,32]
    pois = ['per', 'imp']

    results = []
    for poi in pois:
        for u in users:
            checkpoint_file = f'../../pkl/tunning/checkpoint_{arch}_{u}_{poi}.pkl'
            res = load(checkpoint_file)
            r = sorted(zip(res.func_vals, res.x_iters))[0]
            print(r)
            best = [arch, u, poi, *r[1], r[0]]
            results.append(best)


    df_results = pd.DataFrame(data=results)
    df_results.columns = ['arch', 'Usuario', 'Personal/Impersonal'] + h[arch]
    
    float_df = df_results.select_dtypes(include='float')
    float_cols = float_df.columns
    df_results[float_cols] = np.round(float_df,3)

    int_df = df_results.select_dtypes(include='int64')
    int_cols = int_df.columns
    for col in h[arch]:
        if col in int_df.columns:
            int_df.loc[:,col] = 2 ** int_df[col]

    df_results[int_cols] = int_df
    return  df_results

    
def plot_user_selection(k, min_buckets=0):
    df = get_clean_dataset()
    d = df.groupby(level=0)['slevel'].agg(['count', 'mean', 'std'])
    d = d.loc[d['count']>min_buckets]
    nb_kmean = k
    kmeans = KMeans(n_clusters=nb_kmean).fit(d)

    y = 'Grupo ' + pd.Series(kmeans.predict(d).astype('str'))
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, d)
    for i in closest:
        y[i] = 'Usuario seleccionado'
    print(closest)
    print(kmeans.cluster_centers_)
    y.index = d.index
    y = y.to_frame('y')
    d = pd.concat([d, y], axis=1)
    d.columns = ['Cantidad buckets', 'Promedio MET','Desviacion Estándar MET', 'Grupo']
    g = sns.relplot(x='Cantidad buckets',
                    y='Promedio MET',
                    hue='Grupo',
                    size='Desviacion Estándar MET',
                    sizes=(50, 350),
                    alpha=.6,
                    data=d)

    # array con userid cantbuckets y mean met de cada usuario seleccionado
    to_annotate = d.reset_index(
        drop=False).iloc[:, :3].values
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


