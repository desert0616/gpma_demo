#ifndef CPU_BASELINE_DYNAMIC_GRAPH_H
#define CPU_BASELINE_DYNAMIC_GRAPH_H

#include <vector>

using vertex_type = int;
using edge_type = std::pair<vertex_type, vertex_type>;
using edge_vector_type = std::vector<edge_type>;

template<typename iterator_type>
class dynamic_graph {
public:
    using iterator = iterator_type;

    int num_nodes;

    dynamic_graph(int _num_nodes) : num_nodes(_num_nodes) {};

    // api for update
    virtual void insert_edge(edge_vector_type::iterator begin_iter, edge_vector_type::iterator end_iter) = 0;

    virtual void delete_edge(edge_vector_type::iterator begin_iter, edge_vector_type::iterator end_iter) = 0;

    virtual void init_edge(edge_vector_type::iterator begin_iter, edge_vector_type::iterator end_iter) {
        insert_edge(begin_iter, end_iter);
    }

    // api for access
    virtual iterator_type get_edge_begin(vertex_type v) = 0;

    virtual iterator_type get_edge_end(vertex_type v) = 0;
};


#endif //CPU_BASELINE_DYNAMIC_GRAPH_H
