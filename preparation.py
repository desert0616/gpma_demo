#!/usr/bin/env python
import os
import random

# download dataset
print('downloading graph dataset...')
os.system('wget https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz')
os.system('gunzip soc-pokec-relationships.txt.gz')

# reformat and shuffle
print('re-formating and shuffling graph dataset...')
node_size = 0
edges = []
with open('soc-pokec-relationships.txt', 'r') as f:
    for line in f.readlines():
        a, b = line.strip().split('\t')
        a, b = int(a), int(b)
        edges.append((a, b))
        node_size = max(node_size, a)
        node_size = max(node_size, b)
edge_size = len(edges)
random.shuffle(edges)
with open('pokec.txt', 'w') as f:
    f.write('{} {}\n'.format(node_size, edge_size))
    for a, b in edges:
        f.write('{} {}\n'.format(a - 1, b - 1))
os.system('rm soc-pokec-relationships.txt')

# download CUB v1.8.0
print('downloading CUB...')
os.system('wget https://github.com/NVlabs/cub/archive/1.8.0.zip')
os.system('unzip 1.8.0.zip')
os.system('cp -rf cub-1.8.0/cub .')
os.system('rm -rf cub-1.8.0')
