#include <iostream>
#include "containers/pma_dynamic_graph.hpp"
#include "algorithms/bfs.hpp"

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Invalid arguments.\n");
        return -1;
    }

    char *file_path = argv[1];
    int bfs_start_node = std::atoi(argv[2]);

    FILE *fp;
    fp = fopen(file_path, "r");
    if (!fp) {
        printf("open file failed.\n");
        exit(-1);
    }

    int node_size, edge_size;
    fscanf(fp, "%d %d", &node_size, &edge_size);
    printf("node_num: %d, edge_num: %d\n", node_size, edge_size);

    std::vector <edge_type> edges;

    for (int i = 0; i < edge_size; i++) {
        int x, y;
        fscanf(fp, "%d %d", &x, &y);
        edges.emplace_back(edge_type(x, y));
    }

    printf("Graph file is loaded.\n");
    fclose(fp);

    pma_dynamic_graph graph(node_size);
    int half = edge_size / 2;
    graph.init_edge(edges.begin(), edges.begin() + half);

    std::vector<int> bfs_results;
    bfs<pma_dynamic_graph>(graph, bfs_start_node, bfs_results);
    int reach_nodes = 0;
    for (auto &iter : bfs_results) {
        if (iter) reach_nodes++;
    }
    printf("start from node %d, number of reachable nodes: %d\n", bfs_start_node, reach_nodes);

    int num_slide = 100;
    int step = half / num_slide;
    for (int i = 0; i < num_slide; i++) {
        graph.delete_edge(edges.begin() + step * i, edges.begin() + step * i + step);
        graph.insert_edge(edges.begin() + step * i + half, edges.begin() + step * i + step + half);
    }
    printf("Graph file is updated.\n");

    bfs<pma_dynamic_graph>(graph, bfs_start_node, bfs_results);
    reach_nodes = 0;
    for (auto &iter : bfs_results) {
        if (iter) reach_nodes++;
    }
    printf("start from node %d, number of reachable nodes: %d\n", bfs_start_node, reach_nodes);

    return 0;
}