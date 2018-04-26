#ifndef CPU_BASELINE_PMA_DYNAMIC_GRAPH_H
#define CPU_BASELINE_PMA_DYNAMIC_GRAPH_H

#include "dynamic_graph.h"
#include <cassert>
#include <cmath>
#include <algorithm>

using KEY_TYPE = edge_type;
const vertex_type COL_IDX_NONE = -1;

class pma_iterator {
    int real_idx_;
    std::vector <KEY_TYPE> &data_;
    std::vector<bool> &element_exist_;

public:
    pma_iterator(int real_idx, std::vector <KEY_TYPE> &data, std::vector<bool> &element_exist)
            : real_idx_(real_idx), data_(data), element_exist_(element_exist) {
        while (!element_exist_[real_idx_]) real_idx_++;
    }

    vertex_type &operator*() {
        return data_[real_idx_].second;
    }

    pma_iterator &operator++() {
        real_idx_++;
        while (!element_exist_[real_idx_]) real_idx_++;
        return *this;
    }

    bool operator!=(pma_iterator const &comp) {
        return this->real_idx_ != comp.real_idx_;
    }

};


class pma_dynamic_graph : public dynamic_graph<pma_iterator> {
public:
    // density threshold for lower_leaf, lower_root, upper_root, upper_leaf
    // these four threshold should be monotonically increasing
    double density_lower_thres_leaf_ = 0.08;
    double density_lower_thres_root_ = 0.30;
    double density_upper_thres_root_ = 0.70;
    double density_upper_thres_leaf_ = 0.92;

    std::vector <KEY_TYPE> data_;
    std::vector<bool> element_exist_;
    std::vector <KEY_TYPE> buffer_;

    // csr offset array
    std::vector<int> row_offset_;

    int segment_length_;
    int tree_height_;
    std::vector<int> lower_element_;
    std::vector<int> upper_element_;

    int element_cnt_;

    void recalculate_density() {
        lower_element_.resize((size_t) tree_height_ + 1);
        upper_element_.resize((size_t) tree_height_ + 1);
        int level_length = segment_length_;

        for (int i = 0; i <= tree_height_; i++) {
            double density_lower = density_lower_thres_root_ +
                                   (density_lower_thres_leaf_ - density_lower_thres_root_) * (tree_height_ - i) /
                                   tree_height_;
            double density_upper = density_upper_thres_root_ +
                                   (density_upper_thres_leaf_ - density_upper_thres_root_) * (tree_height_ - i) /
                                   tree_height_;

            lower_element_[i] = (int) ceil(density_lower * level_length);
            upper_element_[i] = (int) floor(density_upper * level_length);

            //special trim for wrong threshold introduced by float-integer conversion
            if (0 < i) {
                lower_element_[i] = std::max(lower_element_[i], 2 * lower_element_[i - 1]);
                upper_element_[i] = std::min(upper_element_[i], 2 * upper_element_[i - 1]);
            }
            level_length <<= 1;
        }
    }

    void init_pma(int num_nodes) {
        // these four density threshold are the monotonically increasing
        assert(density_lower_thres_leaf_ < density_lower_thres_root_);
        assert(density_lower_thres_root_ < density_upper_thres_root_);
        assert(density_upper_thres_root_ < density_upper_thres_leaf_);

        // 2 * lower should be not greater than upper
        assert(2 * density_lower_thres_root_ <= density_upper_thres_root_);

        // the minimal tree structure has 2 levels with 4 elements' space, and the leaf segment's length is 2
        // even if the current density doesn't satisfy the minimum, a halving shouldn't be triggered
        this->segment_length_ = 2;
        this->tree_height_ = 1;
        this->recalculate_density();
        this->data_.resize(4);
        this->element_exist_.resize(4);
        this->element_cnt_ = 0;

        this->row_offset_.resize((unsigned long) (num_nodes + 1));
    }

    int locate_segment(KEY_TYPE value) {
        // when current tree structure is minimal, the lower density is not guaranteed
        // special judgement is required
        if (4 == data_.size()) {
            if (element_exist_[2] && data_[2] <= value) return 2;
            else return 0;
        }

        // binary search the appropriate segment for current value
        int prefix = 0;
        int current_bit = segment_length_ << tree_height_ >> 1;
        while (segment_length_ <= current_bit) {
            if (data_[prefix | current_bit] <= value) prefix |= current_bit;
            current_bit >>= 1;
        }
        return prefix;
    }

    void project_buffer(int head, int rear) {
        buffer_.clear();
        for (int i = head; i != rear; i++) {
            if (element_exist_[i]) buffer_.push_back(data_[i]);
        }
    }

    inline bool is_guard(KEY_TYPE value) { return value.second == COL_IDX_NONE; }

    void evenly_dispatch_buffer(int head, int rear) {
        // reset exist flags
        fill(element_exist_.begin() + head, element_exist_.begin() + rear, false);

        int node_length = rear - head;
        int element_per_segment = (int) buffer_.size() * segment_length_ / node_length;
        int remainder = (int) buffer_.size() * segment_length_ % node_length;

        // use remainder to handle the aliquant part
        // for each segment, if (cur_remainder + remainder) > node_length then assign one more to this segment
        // and update the new cur_remainder
        int cur_remainder = 0;
        int element_assigned = 0;
        int cur_assigned_ptr = 0;
        for (int i = head; i != rear; i += segment_length_) {
            cur_remainder += remainder;
            element_assigned = cur_remainder < node_length ? element_per_segment : element_per_segment + 1;
            cur_remainder %= node_length;
            for (int j = i; j != i + element_assigned; j++) {
                element_exist_[j] = true;
                data_[j] = buffer_[cur_assigned_ptr++];
                if (is_guard(data_[j])) {
                    row_offset_[data_[j].first] = j;
                }
            }
        }
    }

    inline int get_parent(int left_location, int level) { return left_location & ~(segment_length_ << level); };

    void resize(int size) {
        project_buffer(0, (int) data_.size());

        // fls is a builtin func for most x86 amd amd architecture
        // fls(int x) -> floor(log2(x)) + 1, which means the higher bit's index
        // segment length should be a pow of 2
        segment_length_ = 1 << (fls(fls(size)) - 1);
        tree_height_ = fls(size / segment_length_) - 1;

        // rebuild PMA
        recalculate_density();
        data_.resize((unsigned long) size);
        element_exist_.resize((unsigned long) size);
        evenly_dispatch_buffer(0, size);
    }

    void rebalance(int left_location, int level) {
        int node_length = segment_length_ << level;
        int node_element_cnt = (int) count(element_exist_.begin() + left_location,
                                           element_exist_.begin() + left_location + node_length, true);
        if (lower_element_[level] <= node_element_cnt && node_element_cnt <= upper_element_[level]) {
            // this node satisfy the desnity threshold, do the rebalance
            int right_location = left_location + node_length;
            project_buffer(left_location, right_location);
            evenly_dispatch_buffer(left_location, right_location);
        } else {
            if (level == tree_height_) {
                // root imbalance, double or halve PMA
                if (node_element_cnt < lower_element_[level]) resize((int) data_.size() >> 1);
                else resize((int) data_.size() << 1);
            } else {
                // unsatisfied density, to rebalance its parent
                rebalance(get_parent(left_location, level), level + 1);
            }
        }
    }

    void insert_pma(KEY_TYPE value) {
        int segment_head = locate_segment(value);
        int segment_size = (int) count(element_exist_.begin() + segment_head,
                                       element_exist_.begin() + segment_head + segment_length_, true);
        int segment_rear = segment_head + segment_size;

        for (int i = segment_head; i != segment_rear; i++) {
            if (value < data_[i]) swap(value, data_[i]);
        }
        element_exist_[segment_rear] = true;
        data_[segment_rear] = value;

        ++element_cnt_;
        ++segment_rear;
        ++segment_size;
        if (segment_size > upper_element_[0]) rebalance(get_parent(segment_head, 0), 1);
        else {
            for (int i = segment_head; i != segment_rear; i++) {
                if (is_guard(data_[i])) {
                    row_offset_[data_[i].first] = i;
                }
            }
        }
    }

    void delete_pma(KEY_TYPE value) {
        int segment_head = locate_segment(value);
        int segment_size = (int) count(element_exist_.begin() + segment_head,
                                       element_exist_.begin() + segment_head + segment_length_, true);
        int segment_rear = segment_head + segment_size;
        for (int i = segment_head; i != segment_rear; i++) {
            if (value == data_[i]) {
                for (int j = i + 1; j != segment_rear; j++) data_[j - 1] = data_[j];
                --element_cnt_;
                --segment_rear;
                --segment_size;
                element_exist_[segment_rear] = false;
                break;
            }
        }
        if (segment_length_ > 2 && segment_size < lower_element_[0]) {
            rebalance(get_parent(segment_head, 0), 1);
        } else {
            for (int i = segment_head; i != segment_rear; i++) {
                if (is_guard(data_[i])) {
                    row_offset_[data_[i].first] = i;
                }
            }
        }
    }

    static inline int fls(int x) {
        int r = 32;
        if (!x) return 0;
        if (!(x & 0xffff0000u)) {
            x <<= 16;
            r -= 16;
        }
        if (!(x & 0xff000000u)) {
            x <<= 8;
            r -= 8;
        }
        if (!(x & 0xf0000000u)) {
            x <<= 4;
            r -= 4;
        }
        if (!(x & 0xc0000000u)) {
            x <<= 2;
            r -= 2;
        }
        if (!(x & 0x80000000u)) {
            x <<= 1;
            r -= 1;
        }
        return r;
    }

public:
    pma_dynamic_graph(int _num_nodes) : dynamic_graph(_num_nodes) {
        init_pma(_num_nodes);

        // init csr offset array
        for (int i = 0; i <= _num_nodes; i++) {
            auto gurad = edge_type(i, COL_IDX_NONE);
            insert_pma(gurad);
        }
    }

    void init_edge(edge_vector_type::iterator begin_iter, edge_vector_type::iterator end_iter) override {
        buffer_.clear();
        for (int i = 0; i != (int) data_.size(); i++) {
            if (element_exist_[i]) buffer_.push_back(data_[i]);
        }
        for (auto iter = begin_iter; iter != end_iter; ++iter) {
            buffer_.emplace_back(*iter);
        }
        std::sort(buffer_.begin(), buffer_.end());

        int size = 1;
        while (size * density_upper_thres_root_ < buffer_.size()) size <<= 1;

        segment_length_ = 1 << (fls(fls(size)) - 1);
        tree_height_ = fls(size / segment_length_) - 1;

        // rebuild PMA
        recalculate_density();
        data_.resize((unsigned long) size);
        element_exist_.resize((unsigned long) size);
        evenly_dispatch_buffer(0, size);
    }

    void insert_edge(edge_vector_type::iterator begin_iter, edge_vector_type::iterator end_iter) override {
        for (auto iter = begin_iter; iter != end_iter; iter++) {
            insert_pma(*iter);
        }
    }

    void delete_edge(edge_vector_type::iterator begin_iter, edge_vector_type::iterator end_iter) override {
        for (auto iter = begin_iter; iter != end_iter; iter++) {
            delete_pma(*iter);
        }
    }

    pma_iterator get_edge_begin(vertex_type v) override {
        return pma_iterator(row_offset_[v] + 1, data_, element_exist_);
    }

    pma_iterator get_edge_end(vertex_type v) override {
        return pma_iterator(row_offset_[v + 1], data_, element_exist_);
    }
};

#endif //CPU_BASELINE_PMA_DYNAMIC_GRAPH_H
