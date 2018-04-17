#pragma once

#include "cub/cub.cuh"

#define FULL_MASK 0xffffffff

template<SIZE_TYPE THREADS_NUM>
__global__
void gpma_bfs_gather_kernel(SIZE_TYPE *node_queue, SIZE_TYPE *node_queue_offset,
        SIZE_TYPE *edge_queue, SIZE_TYPE *edge_queue_offset,
        KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE *row_offsets) {

    typedef cub::BlockScan<SIZE_TYPE, THREADS_NUM> BlockScan;
    __shared__ typename BlockScan::TempStorage block_temp_storage;

    volatile __shared__ SIZE_TYPE comm[THREADS_NUM / 32][3];
    volatile __shared__ SIZE_TYPE comm2[THREADS_NUM];
    volatile __shared__ SIZE_TYPE output_cta_offset;
    volatile __shared__ SIZE_TYPE output_warp_offset[THREADS_NUM / 32];

    typedef cub::WarpScan<SIZE_TYPE> WarpScan;
    __shared__ typename WarpScan::TempStorage temp_storage[THREADS_NUM / 32];

    SIZE_TYPE thread_id = threadIdx.x;
    SIZE_TYPE lane_id = thread_id % 32;
    SIZE_TYPE warp_id = thread_id / 32;

    SIZE_TYPE cta_offset = blockDim.x * blockIdx.x;
    while (cta_offset < node_queue_offset[0]) {
        SIZE_TYPE node, row_begin, row_end;
        if (cta_offset + thread_id < node_queue_offset[0]) {
            node = node_queue[cta_offset + thread_id];
            row_begin = row_offsets[node];
            row_end = row_offsets[node + 1];
        } else
            row_begin = row_end = 0;

        // CTA-based coarse-grained gather
        while (__syncthreads_or(row_end - row_begin >= THREADS_NUM)) {
            // vie for control of block
            if (row_end - row_begin >= THREADS_NUM)
                comm[0][0] = thread_id;
            __syncthreads();

            // winner describes adjlist
            if (comm[0][0] == thread_id) {
                comm[0][1] = row_begin;
                comm[0][2] = row_end;
                row_begin = row_end;
            }
            __syncthreads();

            SIZE_TYPE gather = comm[0][1] + thread_id;
            SIZE_TYPE gather_end = comm[0][2];
            SIZE_TYPE neighbour;
            SIZE_TYPE thread_data_in;
            SIZE_TYPE thread_data_out;
            SIZE_TYPE block_aggregate;
            while (__syncthreads_or(gather < gather_end)) {
                if (gather < gather_end) {
                    KEY_TYPE cur_key = keys[gather];
                    VALUE_TYPE cur_value = values[gather];
                    neighbour = (SIZE_TYPE) (cur_key & COL_IDX_NONE);
                    thread_data_in = (neighbour == COL_IDX_NONE || cur_value == VALUE_NONE) ? 0 : 1;
                } else
                    thread_data_in = 0;

                __syncthreads();
                BlockScan(block_temp_storage).ExclusiveSum(thread_data_in, thread_data_out, block_aggregate);
                __syncthreads();
                if (0 == thread_id) {
                    output_cta_offset = atomicAdd(edge_queue_offset, block_aggregate);
                }
                __syncthreads();
                if (thread_data_in)
                    edge_queue[output_cta_offset + thread_data_out] = neighbour;
                gather += THREADS_NUM;
            }
        }

        // warp-based coarse-grained gather
        while (__any_sync(FULL_MASK, row_end - row_begin >= 32)) {
            // vie for control of warp
            if (row_end - row_begin >= 32)
                comm[warp_id][0] = lane_id;

            // winner describes adjlist
            if (comm[warp_id][0] == lane_id) {
                comm[warp_id][1] = row_begin;
                comm[warp_id][2] = row_end;
                row_begin = row_end;
            }

            SIZE_TYPE gather = comm[warp_id][1] + lane_id;
            SIZE_TYPE gather_end = comm[warp_id][2];
            SIZE_TYPE neighbour;
            SIZE_TYPE thread_data_in;
            SIZE_TYPE thread_data_out;
            SIZE_TYPE warp_aggregate;
            while (__any_sync(FULL_MASK, gather < gather_end)) {
                if (gather < gather_end) {
                    KEY_TYPE cur_key = keys[gather];
                    VALUE_TYPE cur_value = values[gather];
                    neighbour = (SIZE_TYPE) (cur_key & COL_IDX_NONE);
                    thread_data_in = (neighbour == COL_IDX_NONE || cur_value == VALUE_NONE) ? 0 : 1;
                } else
                    thread_data_in = 0;

                WarpScan(temp_storage[warp_id]).ExclusiveSum(thread_data_in, thread_data_out, warp_aggregate);

                if (0 == lane_id) {
                    output_warp_offset[warp_id] = atomicAdd(edge_queue_offset, warp_aggregate);
                }

                if (thread_data_in)
                    edge_queue[output_warp_offset[warp_id] + thread_data_out] = neighbour;
                gather += 32;
            }
        }

        // scan-based fine-grained gather
        SIZE_TYPE thread_data = row_end - row_begin;
        SIZE_TYPE rsv_rank;
        SIZE_TYPE total;
        SIZE_TYPE remain;
        __syncthreads();
        BlockScan(block_temp_storage).ExclusiveSum(thread_data, rsv_rank, total);
        __syncthreads();

        SIZE_TYPE cta_progress = 0;
        while (cta_progress < total) {
            remain = total - cta_progress;

            // share batch of gather offsets
            while ((rsv_rank < cta_progress + THREADS_NUM) && (row_begin < row_end)) {
                comm2[rsv_rank - cta_progress] = row_begin;
                rsv_rank++;
                row_begin++;
            }
            __syncthreads();
            SIZE_TYPE neighbour;
            // gather batch of adjlist
            if (thread_id < min(remain, THREADS_NUM)) {
                KEY_TYPE cur_key = keys[comm2[thread_id]];
                VALUE_TYPE cur_value = values[comm2[thread_id]];
                neighbour = (SIZE_TYPE) (cur_key & COL_IDX_NONE);
                thread_data = (neighbour == COL_IDX_NONE || cur_value == VALUE_NONE) ? 0 : 1;
            } else
                thread_data = 0;
            __syncthreads();

            SIZE_TYPE scatter;
            SIZE_TYPE block_aggregate;

            BlockScan(block_temp_storage).ExclusiveSum(thread_data, scatter, block_aggregate);
            __syncthreads();

            if (0 == thread_id) {
                output_cta_offset = atomicAdd(edge_queue_offset, block_aggregate);
            }
            __syncthreads();

            if (thread_data)
                edge_queue[output_cta_offset + scatter] = neighbour;
            cta_progress += THREADS_NUM;
            __syncthreads();
        }

        cta_offset += blockDim.x * gridDim.x;
    }
}

template<SIZE_TYPE THREADS_NUM>
__global__
void gpma_bfs_contract_kernel(SIZE_TYPE *edge_queue, SIZE_TYPE *edge_queue_offset,
        SIZE_TYPE *node_queue, SIZE_TYPE *node_queue_offset,
        SIZE_TYPE level, SIZE_TYPE *label, SIZE_TYPE *bitmap) {

    typedef cub::BlockScan<SIZE_TYPE, THREADS_NUM> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    volatile __shared__ SIZE_TYPE output_cta_offset;

    volatile __shared__ SIZE_TYPE warp_cache[THREADS_NUM / 32][128];
    const SIZE_TYPE HASH_KEY1 = 1097;
    const SIZE_TYPE HASH_KEY2 = 1103;
    volatile __shared__ SIZE_TYPE cta1_cache[HASH_KEY1];
    volatile __shared__ SIZE_TYPE cta2_cache[HASH_KEY2];

    // init cta-level cache
    for (int i = threadIdx.x; i < HASH_KEY1; i += blockDim.x)
        cta1_cache[i] = SIZE_NONE;
    for (int i = threadIdx.x; i < HASH_KEY2; i += blockDim.x)
        cta2_cache[i] = SIZE_NONE;
    __syncthreads();

    SIZE_TYPE thread_id = threadIdx.x;
    SIZE_TYPE warp_id = thread_id / 32;
    SIZE_TYPE cta_offset = blockDim.x * blockIdx.x;

    while (cta_offset < edge_queue_offset[0]) {
        SIZE_TYPE neighbour;
        SIZE_TYPE valid = 0;

        do {
            if (cta_offset + thread_id >= edge_queue_offset[0]) break;
            neighbour = edge_queue[cta_offset + thread_id];

            // warp cull
            SIZE_TYPE hash = neighbour & 127;
            warp_cache[warp_id][hash] = neighbour;
            SIZE_TYPE retrieved = warp_cache[warp_id][hash];
            if (retrieved == neighbour) {
                warp_cache[warp_id][hash] = thread_id;
                if (warp_cache[warp_id][hash] != thread_id)
                    break;
            }

            // history cull
            if (cta1_cache[neighbour % HASH_KEY1] == neighbour) break;
            if (cta2_cache[neighbour % HASH_KEY2] == neighbour) break;
            cta1_cache[neighbour % HASH_KEY1] = neighbour;
            cta2_cache[neighbour % HASH_KEY2] = neighbour;

            // bitmap check
            SIZE_TYPE bit_loc = 1 << (neighbour % 32);
            SIZE_TYPE bit_chunk = bitmap[neighbour / 32];
            if (bit_chunk & bit_loc) break;
            bitmap[neighbour / 32] = bit_chunk + bit_loc;

            SIZE_TYPE ret = atomicCAS(label + neighbour, 0, level);
            valid = ret ? 0 : 1;
        } while (false);
        __syncthreads();

        SIZE_TYPE scatter;
        SIZE_TYPE total;
        BlockScan(temp_storage).ExclusiveSum(valid, scatter, total);
        __syncthreads();

        if (0 == thread_id) {
            output_cta_offset = atomicAdd(node_queue_offset, total);
        }
        __syncthreads();

        if (valid)
            node_queue[output_cta_offset + scatter] = neighbour;

        cta_offset += blockDim.x * gridDim.x;
    }
}

__host__
void gpma_bfs(KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE *row_offsets,
        SIZE_TYPE node_size, SIZE_TYPE edge_size, SIZE_TYPE start_node, SIZE_TYPE *results) {

    cudaMemset(results, 0, sizeof(SIZE_TYPE) * node_size);
    SIZE_TYPE *bitmap;
    cudaMalloc(&bitmap, sizeof(SIZE_TYPE) * ((node_size - 1) / 32 + 1));
    cudaMemset(bitmap, 0, sizeof(SIZE_TYPE) * ((node_size - 1) / 32 + 1));
    SIZE_TYPE *node_queue;
    cudaMalloc(&node_queue, sizeof(SIZE_TYPE) * node_size);
    SIZE_TYPE *node_queue_offset;
    cudaMalloc(&node_queue_offset, sizeof(SIZE_TYPE));
    SIZE_TYPE *edge_queue;
    cudaMalloc(&edge_queue, sizeof(SIZE_TYPE) * edge_size);
    SIZE_TYPE *edge_queue_offset;
    cudaMalloc(&edge_queue_offset, sizeof(SIZE_TYPE));

    // init
    SIZE_TYPE host_num[1];
    host_num[0] = start_node;
    cudaMemcpy(node_queue, host_num, sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
    host_num[0] = 1 << (start_node % 32);
    cudaMemcpy(&bitmap[start_node / 32], host_num, sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
    host_num[0] = 1;
    cudaMemcpy(node_queue_offset, host_num, sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(&results[start_node], host_num, sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);

    SIZE_TYPE level = 1;
    const SIZE_TYPE THREADS_NUM = 256;
    while (true) {
        // gather
        SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, host_num[0]);
        host_num[0] = 0;
        cudaMemcpy(edge_queue_offset, host_num, sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
        gpma_bfs_gather_kernel<THREADS_NUM> <<<BLOCKS_NUM, THREADS_NUM>>>(node_queue, node_queue_offset,
                edge_queue, edge_queue_offset, keys, values, row_offsets);

        // contract
        level++;
        cudaMemcpy(node_queue_offset, host_num, sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
        cudaMemcpy(host_num, edge_queue_offset, sizeof(SIZE_TYPE), cudaMemcpyDeviceToHost);
        BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, host_num[0]);

        gpma_bfs_contract_kernel<THREADS_NUM> <<<BLOCKS_NUM, THREADS_NUM>>>(edge_queue, edge_queue_offset,
                node_queue, node_queue_offset, level, results, bitmap);
        cudaMemcpy(host_num, node_queue_offset, sizeof(SIZE_TYPE), cudaMemcpyDeviceToHost);

        if (0 == host_num[0]) break;
    }

    cudaFree(bitmap);
    cudaFree(node_queue);
    cudaFree(node_queue_offset);
    cudaFree(edge_queue);
    cudaFree(edge_queue_offset);
}
