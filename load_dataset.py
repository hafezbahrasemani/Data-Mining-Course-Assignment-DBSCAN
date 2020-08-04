import folium

import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib as plt
from sklearn import metrics
import pandas as pd
import matplotlib.cm as cmx
import matplotlib.colors as colors
import random


def read_csv(path):
    with open(path, newline='') as file:
        X = []
        Y = []
        for line in file:
            tokens = line.split(',')
            X.append(float(tokens[0]))
            Y.append(float(tokens[1]))
    return list(zip(X, Y))


def draw_map(data):
    m = folium.Map(location=[32.427910, 53.688046], zoom_start=5)

    loc = [35.703136, 51.409126]
    folium.Marker(location=loc).add_to(m)

    for i in range(len(data)):
        folium.Circle(location=data[i], radius=0.25, color='red', fill=True).add_to(m)

    m.save('iran.html')


# define a helper function to get the colors for different clusters
def get_cmap(N):
    '''
    Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.
    '''

    color_norm = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='nipy_spectral')

    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color


def compute_dbscan(data):
    # df = pd.read_csv(path)
    # coords = df.(['Latitude', 'Longitude'])
    # print(df)
    db = DBSCAN(eps=0.25, min_samples=50).fit(data)

    print(len(db.core_sample_indices_))
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # clusters = pd.Series([df[labels == n] for n in range(-1, n_clusters_)])

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print(core_samples_mask)
    # plot result

    unique_labels = set(labels)

    c_maps = get_cmap(n_clusters_)
    # print(c_maps)
    # colors = ['blue', 'green', 'yellow', 'brown', 'pink', 'magenta', 'C0', 'C2', 'C4', 'C6', 'olive', 'gold', 'teal']
    colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
              for i in range(len(unique_labels))]

    m = folium.Map(location=[32.427910, 53.688046], zoom_start=5)

    for i in range(len(data)):
        # x, y = m(lons_select, lats_select)
        label = labels[i]
        if label != -1:
            # print(i, label)
            folium.Circle(location=data[i], radius=0.25, color=colors[label], fill=True).add_to(m)
            # m.scatter(data[i][0], data[i][1], 5, marker='o', color=c_maps(i), zorder=10)

    m.save('iran_results.html')

    # for k, col in zip(unique_labels, color_maps):
    #     if k == -1:
    #         # Black used for noise.
    #         col = [0, 0, 0, 1]
    #
    #     class_member_mask = (labels == k)
    #     print(class_member_mask)
        # xy = data[class_member_mask & core_samples_mask]
        # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
        #          markeredgecolor='k', markersize=14)
        #
        # xy = data[class_member_mask & ~core_samples_mask]
        # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
        #          markeredgecolor='k', markersize=6)

    # plt.title('Estimated number of clusters: %d' % n_clusters_)
    # plt.show()
    # print(len(colors))