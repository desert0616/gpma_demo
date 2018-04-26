#ifndef CPU_BASELINE_BFS_H
#define CPU_BASELINE_BFS_H

#include <vector>
#include <queue>

template<typename graph_type>
void bfs(graph_type &graph, int start_node, std::vector<int> &results) {
    results.clear();
    results.resize((unsigned long) graph.num_nodes, 0);

    results[start_node] = 1;
    std::queue <vertex_type> q;
    while (!q.empty()) q.pop();
    q.push(start_node);

    while (!q.empty()) {
        auto cur = q.front();
        auto level = results[cur];
        q.pop();
        for (auto iter = graph.get_edge_begin(cur); iter != graph.get_edge_end(cur); ++iter) {
            vertex_type neighbour = (*iter);
            if (results[neighbour] == 0) {
                results[neighbour] = level + 1;
                q.push(neighbour);
            }
        }
    }
}

#endif //CPU_BASELINE_BFS_H
