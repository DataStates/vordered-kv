#ifndef __VORDERED_KV
#define __VORDERED_KV

#include "marker.hpp"
#include "emem_history.hpp"
#include "pmem_history.hpp"

#include <atomic>
#include <functional>

template <typename K, typename V, typename P = pmem_history_t <K, V>, bool use_shortcuts = true> class vordered_kv_t {
    static const int MAX_LEVEL = 24;

    struct node_t {
        typedef std::atomic<node_t *> next_t;

        K key;
        typename P::plog_t history{nullptr};
        std::vector<next_t> next, shortcut;

        node_t(const K &k, int levels = MAX_LEVEL) : key(k), next(levels), shortcut(levels) { }
    };

    node_t head, tail;
    std::atomic<int> version{0};
    P pool;
    unsigned int rand_state = 0x123;
    std::mutex rand_mutex;

public:
    inline static const V low_marker = marker_t<V>::low_marker;
    inline static const V high_marker = marker_t<V>::high_marker;

    vordered_kv_t(const std::string &db) : head(low_marker), tail(high_marker), pool(db) {
        for (int i = 0; i < MAX_LEVEL; i++)
            head.next[i].store(&tail);
	using namespace std::placeholders;
	version.store(pool.restore(std::bind(&vordered_kv_t::insert, this, _1, _2, _3)));
    }

    ~vordered_kv_t() {
        node_t *curr = head.next[0].load();
        while (curr != &tail) {
            node_t *next = curr->next[0].load();
	    pool.deallocate(curr->history, true);
            delete curr;
            curr = next;
        }
    }

    void scrub() {
	for (int level = 0; level < MAX_LEVEL; level++) {
	    node_t *valid_pred = &head, *curr = valid_pred->next[level];
	    while (curr != &tail) {
		bool curr_removed = curr->history->info.latest_removed();
		if (!curr_removed) {
		    valid_pred->shortcut[level].store(curr);
		    valid_pred = curr;
		}
		curr = curr->next[level];
	    }
	}
    }

    node_t *find_node(const K &key, node_t **preds, node_t **succs, bool adjustment = false) {
        int level = head.next.size() - 1;
        node_t *pred = &head, *valid_pred = pred, *curr;
	bool pred_removed = false;

        while (true) {
	    curr = pred->next[level].load();
            if (curr->key < key) {
		if constexpr(use_shortcuts) {
		    node_t *scut = valid_pred->shortcut[level].load();
		    if (scut != nullptr && scut->key > curr->key && scut->key < key)
			curr = scut;
		    if (adjustment) {
			bool curr_removed = curr->history->info.latest_removed();
			if (!curr_removed) {
			    if (pred_removed && curr != scut)
				valid_pred->shortcut[level].store(curr);
			    valid_pred = curr;
			}
			pred_removed = curr_removed;
		    }
		}
                pred = curr;
		continue;
	    }
	    preds[level] = pred;
	    succs[level] = curr;
	    if (level == 0)
		break;
	    level--;
        }
        auto ret = curr->key == key ? curr : nullptr;
	return ret;
    }

    bool insert(const K &key, const V &value, typename P::plog_t plog = nullptr) {
        node_t *preds[MAX_LEVEL], *succs[MAX_LEVEL];
        node_t *pred, *succ, *node = nullptr;
        // while insertion is not successful
        while(true) {
            // to find the appropriate node for the insertion, if the node with same key is found it returns that
            // std::cout << "Inside while loop vordered_kv.hpp\n";
            node_t *found = find_node(key, preds, succs); // else returns nullptr
            if (found) { // if node found
                if (node) {
                    // somebody else was faster at inserting the same key
                    if (node->history != plog)
			pool.deallocate(node->history);
                    delete node;
                }
                node = found;
            } else if (node == nullptr) { // if node not found, create a new node
            // std::cout << "If node not found.\n";
		std::unique_lock<std::mutex> lock(rand_mutex);
		int rand_int = rand_r(&rand_state);
		lock.unlock();
                int levels = ffs(rand_int | (1 << (MAX_LEVEL - 1)));
                node = new node_t(key, levels);
                // std::cout << "New node created.\n";
            }
            if (plog == nullptr) { // 
                if (node->history == nullptr)
                // allocate just creates a new ekey_history object log_t, node->history points to that object
		    node->history = pool.allocate();
            // std::cout << "New history object created.\n";
            // version is common for the whole of vordered_kv, it updates for the whole skip list.
                node->history->insert(version, value);
                // std::cout << "New value in history block inserted\n";
            } else
                node->history = plog;
            succ = succs[0];
            if (succ == node)
                return true;
            for (size_t level = 0; level < node->next.size(); level++)
                node->next[level].store(succs[level]);
            pred = preds[0];
            if (pred->next[0].compare_exchange_weak(succ, node)) {
                if (plog == nullptr)
                    pool.append(key, node->history);
                break;
            }
        }
        size_t level = 1;
        while (level < node->next.size()) {
            pred = preds[level];
            succ = succs[level];
            if (!pred->next[level].compare_exchange_weak(succ, node)) {
                find_node(key, preds, succs);
                continue;
            }
            level++;
        }
        return true;
    }

    bool remove(const K &key) {
        node_t *preds[MAX_LEVEL], *succs[MAX_LEVEL];
        node_t *node = find_node(key, preds, succs);
        if (node == nullptr)
            return false;
        node->history->remove(version);
        return true;
    }

    V find(int v, const K &key) {
	node_t *preds[MAX_LEVEL], *succs[MAX_LEVEL];
        node_t *node = find_node(key, preds, succs, false);
        if (node == nullptr)
            return low_marker;
        else
        {
            // std::cout << "Node found.\nVersion : " << v << "\n";
            V val = node->history->find(v);
            // std::cout << "Value found : " << val << "\n";
            return val;
            // return node->history->find(v);
        }  
    }

    void get_snapshot(int v, std::vector<std::pair<K, V>> &result) {
        result.clear();
        node_t *curr = head.next[0].load();
        while (curr != &tail) {
            auto p = std::make_pair(curr->key, curr->history->find(v));
            if (p.second != low_marker)
                result.push_back(p);
            curr = curr->next[0].load();
        }
    }

    void get_key_history(const K &key, std::vector<std::pair<int, V>> &result) {
        result.clear();
	node_t *preds[MAX_LEVEL], *succs[MAX_LEVEL];
        node_t *node = find_node(key, preds, succs, false);
        if (node == nullptr)
            return;
        node->history->copy_to(result);
    }

    size_t get_total_size() const {
        size_t total = 0;
        node_t* curr = head.next[0].load();
        while (curr != &tail) {
            if (curr->history) {
                total += curr->history->size();
            }
            curr = curr->next[0].load();
        }
        return total;
    }

    void print_all_key_values() const {
    // std::cout << "Inside print function.\n";
    node_t *curr = head.next[0].load();
    // std::cout << "Loaded the head.\n";
    while (curr != &tail) {
        // std::cout << "Key: " << curr->key << std::endl << std::endl << std::endl;
        std::vector<std::pair<int, V>> history;
        curr->history->copy_to(history);
        // std::cout << "History copied.\n";
        int count = 0;
        for (const auto& entry : history) {
            count++;
            std::cout << count << "  Timestamp: " << entry.first << ", Value: " << entry.second << std::endl;
        }
        curr = curr->next[0].load();
    }
    // std::cout << "End of the print function.\n";
}

    int latest() {
        return version;
    }

    int tag() {
        return version++;
    }

    void clear_stats() {
    }

    const std::string get_stats() {
        return "not yet implemented";
    }
};

#endif // __VORDERED_KV
