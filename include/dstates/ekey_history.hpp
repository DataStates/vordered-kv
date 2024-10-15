#ifndef __EKEY_HISTORY_T
#define __EKEY_HISTORY_T

#include "marker.hpp"
#include "key_info.hpp"
#include <atomic>
#include <stdexcept>
#include <vector>
#include <memory>
#include <shared_mutex>

template <class V>
class ekey_history_t {
    static const size_t BLOCK_SIZE = 128;

    struct entry_t {
        int ts;
        V val;
        bool marked = false;
    };
    // these are related to a particular block
    struct block_t {
        std::vector<entry_t> entries;
        std::atomic<int> first_ts;
        std::atomic<int> last_ts;
        std::atomic<size_t> size;
        std::shared_ptr<block_t> next;

        block_t() : entries(BLOCK_SIZE), first_ts(std::numeric_limits<int>::max()),
                    last_ts(std::numeric_limits<int>::min()), size(0), next(nullptr) {}
    };
    // these are related to all of ekey_history
    // head is not atomic as it doesn't keep getting updated.
    std::shared_ptr<block_t> head;
    std::atomic<std::shared_ptr<block_t>> tail;
    std::atomic<size_t> total_size;

    // Vector to store the starting timestamp and block pointer for efficient access.
    std::vector<std::pair<int, std::shared_ptr<block_t>>> block_index;
    mutable std::shared_mutex block_index_mutex;


public:
    key_info_t info;

    ekey_history_t() : head(std::make_shared<block_t>()), tail(head), total_size(0) {
        std::unique_lock lock(block_index_mutex);
        block_index.emplace_back(std::numeric_limits<int>::min(), head);

    }

    void insert(int t, const V &v) {
        while (true) {
            auto current_tail = tail.load();
            size_t slot = current_tail->size.fetch_add(1);
            
            if (slot < BLOCK_SIZE) {
                current_tail->entries[slot].ts = t;
                current_tail->entries[slot].val = v;
                current_tail->entries[slot].marked = true;

                int expected_first = current_tail->first_ts.load();
                current_tail->first_ts.compare_exchange_strong(expected_first, std::min(expected_first, t));

                int expected_last = current_tail->last_ts.load();
                current_tail->last_ts.compare_exchange_strong(expected_last, std::max(expected_last, t));

                total_size.fetch_add(1);
                // info.update(t, v == marker_t<V>::low_marker);
                return;
            } else {
                auto new_block = std::make_shared<block_t>();
                std::shared_ptr<block_t> expected_next = nullptr;
                if (std::atomic_compare_exchange_strong(&current_tail->next, &expected_next, new_block)) {
                    std::shared_ptr<block_t> expected_tail = current_tail;
                    tail.compare_exchange_strong(expected_tail, new_block);

                    std::unique_lock lock(block_index_mutex);
                    block_index.emplace_back(t, new_block);
                }
            }
        }
    }

    void remove(int t) {
        insert(t, marker_t<V>::low_marker);                                                                                 
    }

    V find(int t) {

        std::shared_ptr<block_t> target_block;
        {
            std::shared_lock lock(block_index_mutex);
            
            if (block_index.empty()) {
                return marker_t<V>::low_marker;
            }

            int left = 0;
            int right = block_index.size() - 1;

            while (left <= right) {
                int mid = left + (right - left) / 2;

                if (block_index[mid].first <= t) {
                    // if we have reached the last block, or the starting value of next block is > t, then this is the block
                    if (mid == block_index.size() - 1 || block_index[mid + 1].first > t) {
                        target_block = block_index[mid].second;
                        break;
                    }
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }   
            // if after search target_block is not set
            if (!target_block) {
                // right points to the value just less than t
                if (right >= 0) {
                    target_block = block_index[right].second;
                } else {
                    return marker_t<V>::low_marker;
                }
            }
        }

        size_t left = 0, right = target_block->size.load() - 1;
        // std::cout << "Block size : " << right << "\n";
        while (left <= right) {
            size_t middle = left + (right - left) / 2;
            if (t < target_block->entries[middle].ts)
                right = middle - 1;
            else if (t > target_block->entries[middle].ts)
                left = middle + 1;
                // if t == middle timestamp and it is marked
            else if (target_block->entries[middle].marked){
                std::cout << "Value returned after finding exactly : " << target_block->entries[middle].val << "\n";
                return target_block->entries[middle].val;
            }
            else
                right = middle - 1;  // If not marked, continue searching for the previous valid entry
        }

        // If no exact match, return the most recent value before t
        while (right >= 0) {
            if (target_block->entries[right].marked && target_block->entries[right].ts <= t) {
                std::cout << "Value returned after not finding a match : " << target_block->entries[right].val << "\n";
                return target_block->entries[right].val;
            }
            --right;
        }
        std::cout << "Returning low marker : " << marker_t<V>::low_marker << "\n";
        return marker_t<V>::low_marker;
    }
    

    void copy_to(std::vector<std::pair<int, V>> &result) {
        result.clear();
        auto current = head;
        // std::cout << "Inside copy function.\n";
        while (current) {
            std::cout << "Going through new block.\n";
            std::cout << "size of the current block : " << current->size.load() - 1 << "\n";
            for (size_t i = 0; i < current->size.load() - 1; ++i) {
                // std::cout << "Going through a particular value.  " << i << " \n";
                if (current->entries[i].marked) {
                    // std::cout<< "Inside the if block.\n";
                    result.emplace_back(current->entries[i].ts, current->entries[i].val);
                    // std::cout << "After the emplacing.\n";
                }

            }
            current = current->next;
        }
        // std::cout << "End of copy function.\n";
    }

    void cleanup() {
        auto current = head;
        while (current) {
            auto next = current->next;
            current.reset();
            current = next;
        }
        head.reset();
        tail.store(nullptr);
        total_size.store(0);
    }
    
    size_t size() {
        return total_size.load();
    }
};

#endif // __EKEY_HISTORY_T