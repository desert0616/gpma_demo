# GPMA Demo

Source code for the paper:

[Accelerating Dynamic Graph Analytics on GPUs](http://www.vldb.org/pvldb/vol11/p107-sha.pdf)

GPMA is a data structure to maintain dynamic graphs on GPUs. This repository illustrates a demo to conduct BFS/Connected Component/PageRank on a dynamic graph, which is maintained by GPMA, on the GPU.

## Environment and Dependency
This code is developed and tested on:
* Ubuntu 16.04
* GeForce GTX 1080 Ti with Nvidia Drive 384.111
* CUDA 9.0
* [CUB](https://nvlabs.github.io/cub/) v1.8.0

## Preparation

```preparation.py``` will download, re-format, and shuffle a graph dataset, [__pokec__](https://snap.stanford.edu/data/soc-pokec.html), which can be used in this demo. Meanwhile, it will put CUB in the root folder.

## Build

To build this demo, use ```make```.
You may need to modify the ```Makefile``` with a proper setting, e.g., nvcc path, include path, and GPU architecture.

## Demo
```./gpma_demo [bfs/cc/pr] [graph_path] [bfs_start_node]```

In this demo, first, the first half of edges (the init sliding window) of the given graph will be loaded into GPMA, and then, the chosen graph application is conducted. After that, the sliding window will be moved 100 times to the second half of edges, which means that the current sliding window will not overlap with the original one. Finally, the corresponding graph application is conducted on the updated graph.

The format of the given graph should start with one line including node_size and edge_size, and the following edge_size lines should provide all edges. The edges are directed.

If you have executed ```preparation.py``` to generate a well-formatted pokec graph dataset in advance, and you want to start the BFS from node 0:

```./gpma_demo bfs pokec.txt 0```

The output should be in a similar format as follows:

```
node_num: 1632803, edge_num: 30622564
Graph file is loaded.
start from node 0, number of reachable nodes: 1334862
Graph is updated.
start from node 0, number of reachable nodes: 1334356
```

The number of reachable nodes may be different since the graph is shuffled.

## Reference
```
@article{sha2017gpma,
 title={Accelerating Dynamic Graph Analytics on GPUs},
 author={Sha, Mo and Li, Yuchen and He, Bingsheng and Tan, Kian-Lee},
 journal={Proceedings of the VLDB Endowment},
 volume={11},
 number={1},
 year={2017}
}
```

## License
MIT
