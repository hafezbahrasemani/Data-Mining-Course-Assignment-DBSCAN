import load_dataset
import dbscan
from guppy import heapy

if __name__ == '__main__':
    path = 'covid.csv'

    data = load_dataset.read_csv(path)

    # print(data)
    # load_dataset.draw_map(data)
    load_dataset.compute_dbscan(data)