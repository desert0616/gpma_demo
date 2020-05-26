#include <iostream>
#include <string>

#include "gpma.cuh"
#include "gpma_bfs.cuh"
#include "gpma_cc.cuh"
#include "gpma_pr.cuh"

void load_data(const char *file_path, thrust::host_vector<int> &host_x, thrust::host_vector<int> &host_y,
        int &node_size, int &edge_size) {

    FILE *fp;
    fp = fopen(file_path, "r");
    if (not fp) {
        printf("Open graph file failed.\n");
        exit(0);
    }

    fscanf(fp, "%d %d", &node_size, &edge_size);
    printf("node_num: %d, edge_num: %d\n", node_size, edge_size);

    host_x.resize(edge_size);
    host_y.resize(edge_size);

    for (int i = 0; i < edge_size; i++) {
        int x, y;
        fscanf(fp, "%d %d", &x, &y);
        host_x[i] = x;
        host_y[i] = y;
    }

    printf("Graph file is loaded.\n");
    fclose(fp);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Invalid arguments.\n");
        printf("<app (bfs/cc/pr)> <graph_path> <node_id (bfs)>\n");
        return -1;
    }

    std::string app = std::string(argv[1]);
    if (app != "bfs" and app != "cc" and app != "pr") {
        printf("Unsupported application.\n");
        printf("<app (bfs/cc/pr)> <graph_path> <node_id (bfs)>\n");
        return -1;
    }

    char* data_path = argv[2];

    int start_node = 0;
    if (app == "bfs") {
    	if (argc != 4) {
            printf("Node id needed.\n");
            printf("<app (bfs/cc/pr)> <graph_path> <node_id (bfs)>\n");
            return -1;
    	}
    	start_node = std::atoi(argv[3]);
    }


    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024ll * 1024 * 1024);
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 5);

    thrust::host_vector<int> host_x;
    thrust::host_vector<int> host_y;
    int node_size;
    int edge_size;
    load_data(data_path, host_x, host_y, node_size, edge_size);

    int half = edge_size / 2;

    thrust::host_vector<KEY_TYPE> h_base_keys(half);
    for (int i = 0; i < half; i++) {
        h_base_keys[i] = ((KEY_TYPE) host_x[i] << 32) + host_y[i];
    }

    DEV_VEC_KEY base_keys = h_base_keys;
    DEV_VEC_VALUE base_values(half, 1);
    cudaDeviceSynchronize();

    int num_slide = 100;
    int step = half / num_slide;

    GPMA gpma;
    init_csr_gpma(gpma, node_size);
    cudaDeviceSynchronize();

    update_gpma(gpma, base_keys, base_values);

    if (app == "bfs") {
        thrust::device_vector<SIZE_TYPE> bfs_result(node_size);
        cudaDeviceSynchronize();
        gpma_bfs(RAW_PTR(gpma.keys), RAW_PTR(gpma.values), RAW_PTR(gpma.row_offset), node_size,
        		gpma.get_size(), start_node, RAW_PTR(bfs_result));
        int reach_nodes = node_size - thrust::count(bfs_result.begin(), bfs_result.end(), 0);
        printf("start from node %d, number of reachable nodes: %d\n", start_node, reach_nodes);
    } else if (app == "cc") {
        thrust::device_vector<SIZE_TYPE> cc_result(node_size);
        cudaDeviceSynchronize();
    	gpma_cc(RAW_PTR(gpma.keys), RAW_PTR(gpma.values), RAW_PTR(gpma.row_offset), node_size, gpma.get_size(), cc_result);

        thrust::host_vector<SIZE_TYPE> host_cc_result(cc_result);
        cudaDeviceSynchronize();
        int cc_count = 0;
        for (int i = 0; i < host_cc_result.size(); i++) {
            if (i == host_cc_result[i]) cc_count++;
        }
        printf("number of connected components: %d\n", cc_count);
    } else if (app == "pr") {
        thrust::host_vector<VALUE_TYPE> pr_result(node_size);
        thrust::fill(pr_result.begin(), pr_result.end(), 1.0);
        int iter_cnt = 0;
        cudaDeviceSynchronize();

        gpma_pr(gpma.keys, gpma.values, gpma.row_offset, node_size, pr_result, gpma.get_size(), iter_cnt);
        cudaDeviceSynchronize();

        printf("pagerank converged in %d iterations\n", iter_cnt);
    }

    for (int i = 0; i < num_slide; i++) {
        thrust::host_vector<KEY_TYPE> hk(step * 2);
        for (int j = 0; j < step; j++) {
            int idx = half + i * step + j;
            hk[j] = ((KEY_TYPE) host_x[idx] << 32) + host_y[idx];
        }
        for (int j = 0; j < step; j++) {
            int idx = i * step + j;
            hk[j + step] = ((KEY_TYPE) host_x[idx] << 32) + host_y[idx];
        }

        DEV_VEC_VALUE update_values(step * 2);
        thrust::fill(update_values.begin(), update_values.begin() + step, 1);
        thrust::fill(update_values.begin() + step, update_values.end(), VALUE_NONE);
        DEV_VEC_KEY update_keys = hk;
        cudaDeviceSynchronize();

        update_gpma(gpma, update_keys, update_values);
        cudaDeviceSynchronize();
    }
    printf("Graph is updated.\n");

    if (app == "bfs") {
        thrust::device_vector<SIZE_TYPE> bfs_result(node_size);
        cudaDeviceSynchronize();
        gpma_bfs(RAW_PTR(gpma.keys), RAW_PTR(gpma.values), RAW_PTR(gpma.row_offset), node_size,
        		gpma.get_size(), start_node, RAW_PTR(bfs_result));
        int reach_nodes = node_size - thrust::count(bfs_result.begin(), bfs_result.end(), 0);
        printf("start from node %d, number of reachable nodes: %d\n", start_node, reach_nodes);
    } else if (app == "cc") {
        thrust::device_vector<SIZE_TYPE> cc_result(node_size);
        cudaDeviceSynchronize();
    	gpma_cc(RAW_PTR(gpma.keys), RAW_PTR(gpma.values), RAW_PTR(gpma.row_offset), node_size, gpma.get_size(), cc_result);

        thrust::host_vector<SIZE_TYPE> host_cc_result(cc_result);
        cudaDeviceSynchronize();
        int cc_count = 0;
        for (int i = 0; i < host_cc_result.size(); i++) {
            if (i == host_cc_result[i]) cc_count++;
        }
        printf("number of connected components: %d\n", cc_count);
    } else if (app == "pr") {
        thrust::host_vector<VALUE_TYPE> pr_result(node_size);
        thrust::fill(pr_result.begin(), pr_result.end(), 1.0);
        int iter_cnt = 0;
        cudaDeviceSynchronize();

        gpma_pr(gpma.keys, gpma.values, gpma.row_offset, node_size, pr_result, gpma.get_size(), iter_cnt);
        cudaDeviceSynchronize();

        printf("pagerank converged in %d iterations\n", iter_cnt);
    }

    return 0;
}
