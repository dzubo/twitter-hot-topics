# detect_hot.py - get list of important events on historical data
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn import metrics

import progressbar

input_file = '../data/input/tweets-vec.csv'
output_file = '../data/output/hot-tweets.csv'
MA_WIDTH = 20
HOT_THRESHOLD = 2.5

df = pd.read_csv(input_file)
print('Total {} tweets'.format(len(df)))

# Convert to datetime
df['dt'] = pd.to_datetime(df.created_at)

# Restore list of numbers in vectors
vector_temp = df.avg_vector.apply(lambda x: x.strip('[ ]').split())
vector_float = vector_temp.apply(lambda x: [np.float(elem) for elem in x])
df['avg_vector'] = vector_float


# Running clustering on the window

# metric = 'cosine', 'euclidean', ...
def get_clusters(df, n_clusters=20, metric='cosine'):
    X = np.array(df.avg_vector.tolist(), ndmin=2)
    kmeans = KMeans(n_clusters=n_clusters, init='random').fit(X)
    cluster = kmeans.labels_
    silhouette_values = metrics.silhouette_samples(X, kmeans.labels_,
                                                   metric=metric)
    res = pd.DataFrame({'cluster': cluster,
                        'sil_value': silhouette_values},
                       index=df.index)
    return(res)


def get_cluster_order(df, news_min=10, metric='cosine'):
    cluster = df.cluster
    silhouette_values = df.sil_value
    sil_clusters = pd.DataFrame({'cluster': cluster, 'sil': silhouette_values})

    grouped = sil_clusters.groupby('cluster')
    col_order_sum = grouped.aggregate({'sil': {sum, 'count'}})['sil']. \
        sort_values(by='sum', ascending=False)
    col_order_sum = col_order_sum[col_order_sum['count'] >= news_min]
    scores = list(col_order_sum['sum'])
    col_order_sum = list(col_order_sum.index)

    res = pd.DataFrame({'cluster': col_order_sum,
                        'score': scores})
    return(res)


def get_max_score(df):
    clusters = get_clusters(df)
    cluster_order = get_cluster_order(clusters)
    return(cluster_order.score[0])


# Rolling window on historical data
df_dt = df.set_index('dt')


# 'minute_range' is in munites
def setup_intervals(start, end, freq='10min', minute_range=120):
    dt1 = pd.date_range(start, end, freq=freq)
    dt2 = dt1 - pd.Timedelta(minutes=minute_range)
    intervals = pd.DataFrame({'t_from': dt2, 't_to': dt1})
    return(intervals)


intervals = setup_intervals(start='2017-03-15 00:00:00',
                            end='2017-03-16 02:00:00',
                            freq='10min', minute_range=120)


def get_scores(df, intervals):
    # bar = progressbar.ProgressBar(max_value=len(intervals)-1)
    scores = list()
    hots = list()
    for index, interval in intervals.iterrows():
        t_from = interval.t_from
        t_to = interval.t_to
        df_interval = df[(df.index >= t_from) & (df.index <= t_to)]

        # Clustering, choosing most important cluster
        clusters = get_clusters(df_interval)
        cluster_order = get_cluster_order(clusters)
        score = cluster_order.score[0]
        scores.append(score)

        # Detecting hot cluster
        hot = False
        if(len(scores) > MA_WIDTH):
            ma = np.mean(scores[-MA_WIDTH:])
            std = np.std(scores[-MA_WIDTH:])
            if(score > (ma + HOT_THRESHOLD * std)):
                hot = True
                # Record hot tweet
                df_interval_res = pd.concat([df_interval, clusters], axis=1)
                cluster = cluster_order.head(1)
                print('Score = {:.1f}'.format(cluster.score[0]))
                tweet = df_interval_res[df_interval_res.cluster == cluster.cluster[0]]. \
                        sort_values(by=['sil_value'], ascending=False).head(1)
                print(tweet)
        hots.append(hot)

        # bar.update(index)
    res = pd.DataFrame({'score': scores, 'hot': hots}, index=intervals.index)
    return(res)


scores = get_scores(df_dt, intervals)
print(scores)
