#pragma once

#define PR_THRES 0.001

template<SIZE_TYPE VECTORS_PER_BLOCK, SIZE_TYPE THREADS_PER_VECTOR>
__global__
void gpma_csr_spmv_pr_kernel(SIZE_TYPE *row_offset, KEY_TYPE *keys,
        VALUE_TYPE *values, SIZE_TYPE row_num, VALUE_TYPE *x, VALUE_TYPE *y,
        VALUE_TYPE *lp, VALUE_TYPE q) {

    __shared__  volatile VALUE_TYPE reduce_data[VECTORS_PER_BLOCK
            * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];
    __shared__  volatile SIZE_TYPE ptrs[VECTORS_PER_BLOCK][2];

    const SIZE_TYPE THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;

    const SIZE_TYPE thread_id = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;
    const SIZE_TYPE thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);
    const SIZE_TYPE vector_id = thread_id / THREADS_PER_VECTOR;
    const SIZE_TYPE vector_lane = threadIdx.x / THREADS_PER_VECTOR;
    const SIZE_TYPE num_vectors = VECTORS_PER_BLOCK * gridDim.x;

    for (SIZE_TYPE row = vector_id; row < row_num; row += num_vectors) {
        // use two threads to fetch row pointer
        if (thread_lane < 2) {
            ptrs[vector_lane][thread_lane] = row_offset[row + thread_lane];
        }

        const SIZE_TYPE row_start = ptrs[vector_lane][0];
        const SIZE_TYPE row_end = ptrs[vector_lane][1];

        VALUE_TYPE sum = 0.0;

        for (SIZE_TYPE i = row_start + thread_lane; i < row_end; i +=
                THREADS_PER_VECTOR) {
            SIZE_TYPE col_idx = keys[i] & COL_IDX_NONE;
            VALUE_TYPE value = values[i];
            if (COL_IDX_NONE != col_idx && value != VALUE_NONE)
                sum += (value / lp[col_idx]) * x[col_idx];
        }

        reduce_data[threadIdx.x] = sum;

        // reduce the sum of threads
        VALUE_TYPE temp;
        if (THREADS_PER_VECTOR > 16) {
            temp = reduce_data[threadIdx.x + 16];
            reduce_data[threadIdx.x] = sum = sum + temp;
        }
        if (THREADS_PER_VECTOR > 8) {
            temp = reduce_data[threadIdx.x + 8];
            reduce_data[threadIdx.x] = sum = sum + temp;
        }
        if (THREADS_PER_VECTOR > 4) {
            temp = reduce_data[threadIdx.x + 4];
            reduce_data[threadIdx.x] = sum = sum + temp;
        }
        if (THREADS_PER_VECTOR > 2) {
            temp = reduce_data[threadIdx.x + 2];
            reduce_data[threadIdx.x] = sum = sum + temp;
        }
        if (THREADS_PER_VECTOR > 1) {
            temp = reduce_data[threadIdx.x + 1];
            reduce_data[threadIdx.x] = sum = sum + temp;
        }

        // write back answer
        if (0 == thread_lane) {
            y[row] = q * reduce_data[threadIdx.x] + (1 - q);
        }
    }
}

// y = p * A * x + (1 - q)
template<SIZE_TYPE THREADS_PER_VECTOR>
__host__
void _pagerank_one(SIZE_TYPE *row_offset, KEY_TYPE *keys, VALUE_TYPE *values,
        SIZE_TYPE row_num, VALUE_TYPE *lp, VALUE_TYPE q, VALUE_TYPE *x,
        VALUE_TYPE *y) {
    const SIZE_TYPE THREADS_NUM = 128;
    const SIZE_TYPE VECTORS_PER_BLOCK = THREADS_NUM / THREADS_PER_VECTOR;
    const SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(VECTORS_PER_BLOCK, row_num);

    gpma_csr_spmv_pr_kernel<VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<BLOCKS_NUM, THREADS_NUM>>>(
            row_offset, keys, values, row_num, x, y, lp, q);
}
__host__
void pagerank_one_iteration(SIZE_TYPE *row_offset, KEY_TYPE *keys,
        VALUE_TYPE *values, SIZE_TYPE row_num, VALUE_TYPE *lp, VALUE_TYPE q,
        VALUE_TYPE *x, VALUE_TYPE *y, SIZE_TYPE avg_nnz_per_row) {

    if (avg_nnz_per_row <= 2) {
        _pagerank_one<2>(row_offset, keys, values, row_num, lp, q, x, y);
        return;
    }
    if (avg_nnz_per_row <= 4) {
        _pagerank_one<4>(row_offset, keys, values, row_num, lp, q, x, y);
        return;
    }
    if (avg_nnz_per_row <= 8) {
        _pagerank_one<8>(row_offset, keys, values, row_num, lp, q, x, y);
        return;
    }
    if (avg_nnz_per_row <= 16) {
        _pagerank_one<16>(row_offset, keys, values, row_num, lp, q, x, y);
        return;
    }
    _pagerank_one<32>(row_offset, keys, values, row_num, lp, q, x, y);
}

template<typename T>
struct square {
    __host__  __device__ T operator()(const T& x) const {
        return x * x;
    }
};
template<typename T>
struct ABS {
    __host__  __device__ T operator()(const T& x) const {
        return abs(x);
    }
};

void gpma_pr(DEV_VEC_KEY &keys, DEV_VEC_VALUE &values, DEV_VEC_SIZE &row_offset, SIZE_TYPE row_num,
        thrust::host_vector<VALUE_TYPE> &result, SIZE_TYPE nnz_num, int &iter_cnt) {
    DEV_VEC_VALUE pr[2], norm_pr[2];
    pr[0].resize(row_num, 1);
    pr[1].resize(row_num, 1);
    norm_pr[0].resize(row_num);
    norm_pr[1].resize(row_num);
    DEV_VEC_VALUE dist(row_num);
    pr[0] = result;

    int cnt = 0;
    cudaDeviceSynchronize();

    // generate lp array
    thrust::host_vector<SIZE_TYPE> h_row_offset = row_offset;
    thrust::host_vector<KEY_TYPE> h_keys = keys;
    thrust::host_vector<VALUE_TYPE> h_values = values;
    thrust::host_vector<VALUE_TYPE> h_lp(row_num, 0.0);
    cudaDeviceSynchronize();

    for (int i = 0; i < row_num; i++) {
        for (int j = h_row_offset[i]; j < h_row_offset[i + 1]; j++) {
            KEY_TYPE key = h_keys[j];
            VALUE_TYPE value = h_values[j];
            SIZE_TYPE col_idx = key & COL_IDX_NONE;
            if (COL_IDX_NONE != col_idx && value != VALUE_NONE) {
                h_lp[col_idx] += 1.0;
            }
        }
    }
    DEV_VEC_VALUE d_lp;
    d_lp = h_lp;
    cudaDeviceSynchronize();

    SIZE_TYPE avg_nnz_per_row = nnz_num / row_num;

    double norm2;
    do {
        pagerank_one_iteration(RAW_PTR(row_offset), RAW_PTR(keys),
                RAW_PTR(values), row_num, RAW_PTR(d_lp), 0.85,
                RAW_PTR(pr[cnt % 2]), RAW_PTR(pr[(cnt + 1) % 2]),
                avg_nnz_per_row);
        cnt++;

        double mod1 = sqrt(
                thrust::transform_reduce(pr[0].begin(), pr[0].end(),
                        square<double>(), 0.0, thrust::plus<double>()));
        double mod2 = sqrt(
                thrust::transform_reduce(pr[1].begin(), pr[1].end(),
                        square<double>(), 0.0, thrust::plus<double>()));

        thrust::constant_iterator<double> ci_mod1(mod1);
        thrust::constant_iterator<double> ci_mod2(mod2);

        thrust::transform(pr[0].begin(), pr[0].end(), ci_mod1,
                norm_pr[0].begin(), thrust::divides<double>());
        thrust::transform(pr[1].begin(), pr[1].end(), ci_mod2,
                norm_pr[1].begin(), thrust::divides<double>());

        thrust::transform(norm_pr[0].begin(), norm_pr[0].end(),
                norm_pr[1].begin(), dist.begin(), thrust::minus<double>());
        norm2 = sqrt(
                thrust::transform_reduce(dist.begin(), dist.end(),
                        square<double>(), 0.0, thrust::plus<double>()));
        printf("PR iter: %d\tnorm2: %f\n", cnt, norm2);
    } while (norm2 > PR_THRES);

    result = pr[cnt % 2];
    cudaDeviceSynchronize();
    iter_cnt = cnt;
}
