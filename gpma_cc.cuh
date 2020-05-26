#pragma once

#include "cub/cub.cuh"
#include <thrust/sequence.h>

__global__
void gpma_cc_hook_kernel(KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE edge_size, SIZE_TYPE *parents, bool *edge_flag,
        bool *done_flag, int iter, SIZE_TYPE node_size) {

    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;

    for (int i = global_thread_id; i < edge_size; i += block_offset) {
        if (edge_flag[i]) {
            KEY_TYPE cur_key = keys[i];
            VALUE_TYPE cur_value = values[i];
            SIZE_TYPE u = SIZE_TYPE(cur_key >> 32);
            SIZE_TYPE v = SIZE_TYPE(cur_key & COL_IDX_NONE);

            if (u != COL_IDX_NONE && v != COL_IDX_NONE && cur_value != VALUE_NONE && parents[u] != parents[v]) {
                SIZE_TYPE pu = parents[u];
                SIZE_TYPE pv = parents[v];
                (*done_flag) = false;
                SIZE_TYPE max_parent = max(pu, pv);
                SIZE_TYPE min_parent = min(pu, pv);
                if (iter % 2) parents[max_parent] = min_parent;
                else parents[min_parent] = max_parent;
            } else {
                edge_flag[i] = false;
            }
        }
    }
}

__global__
void gpma_cc_pointer_jumping_kernel(SIZE_TYPE *parents, SIZE_TYPE node_size) {

    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;

    for (int i = global_thread_id; i < node_size; i += block_offset) {
        while (parents[i] != parents[parents[i]]) parents[i] = parents[parents[i]];
    }
}

__host__
void gpma_cc(KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE *row_offset, SIZE_TYPE node_size,
        SIZE_TYPE edge_size, DEV_VEC_SIZE &results) {

    results.resize(node_size);
    thrust::sequence(results.begin(), results.end());
    thrust::device_vector<bool> edge_flag(edge_size, true);
    cudaDeviceSynchronize();

    bool *h_done_flag;
    bool *d_done_flag;
    h_done_flag = (bool*) std::malloc(sizeof(bool));
    cudaMalloc(&d_done_flag, sizeof(bool));

    int iter = 0;

    const SIZE_TYPE THREADS_NUM = 256;
    do {
        iter++;
        h_done_flag[0] = true;
        cudaMemcpy(d_done_flag, h_done_flag, sizeof(bool), cudaMemcpyHostToDevice);

        // hook
        SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, edge_size);
        gpma_cc_hook_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(keys, values, edge_size, RAW_PTR(results), RAW_PTR(edge_flag), d_done_flag, iter, node_size);

        // pointer jumping
        BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, node_size);
        gpma_cc_pointer_jumping_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(RAW_PTR(results), node_size);

        cudaMemcpy(h_done_flag, d_done_flag, sizeof(bool), cudaMemcpyDeviceToHost);
    } while (!h_done_flag[0]);

    free(h_done_flag);
    cudaFree(d_done_flag);
}
