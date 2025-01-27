#ifndef __PMEM_HISTORY
#define __PMEM_HISTORY

#include "pkey_history.hpp"
#include "pkey_chain.hpp"

#include <omp.h>
#include <thread>
#include <unistd.h>

#define __DEBUG
#include "debug.hpp"

template <typename K, typename V> class pmem_history_t {
public:
    typedef pkey_history_t<V> log_t;
    typedef pmem::obj::persistent_ptr<log_t> plog_t;

private:
    static const size_t BLOCK_SIZE = 1024;
    inline static const std::string POOL_NAME = "vordered_map_pool";

    typedef typename std::conditional<std::is_same<K, std::string>::value, pmem::obj::string, K>::type PK;
    typedef std::pair<PK, plog_t> entry_t;
    typedef pkey_chain_t<entry_t, BLOCK_SIZE> keymap_t;
    typedef pmem::obj::persistent_ptr<keymap_t> pkeymap_t;
    struct root_t {
	pkeymap_t keymap;
    };
    typedef pmem::obj::pool<root_t> pool_t;

    pool_t pool;

public:
    pmem_history_t(const std::string &db_name, size_t db_size) {
	if (access(db_name.c_str(), F_OK) != 0) {
            pool = pmem::obj::pool<root_t>::create(db_name, POOL_NAME, db_size);
	    pmem::obj::transaction::run(pool, [&] {
		pool.root()->keymap = pmem::obj::make_persistent<keymap_t>();
	    });
            DBG("created a new pmembobj pool, path = " << db_name);
        } else {
            pool = pmem::obj::pool<root_t>::open(db_name, POOL_NAME);
            DBG("opened an existing pmemobj pool, path = " << db_name);
	}
    }
    ~pmem_history_t() {
	pool.close();
    }

    int restore(std::function<bool (const K &, const V &, plog_t)> inserter) {
	TIMER_START(restore_index);
	std::atomic<int> count{0}, version{0};
	int thread_no = std::thread::hardware_concurrency();
        #pragma omp parallel num_threads(thread_no)
	{
	    auto head = pool.root()->keymap->get_head();
	    int block_id = 0;
	    while (head) {
		if (block_id % thread_no == omp_get_thread_num()) {
		    for (size_t i = 0; i < BLOCK_SIZE; i++) {
			plog_t log = head->block[i].second;
			if (log) {
			    int prev, curr = log->info.latest_version();
			    do {
				prev = version.load();
			    } while (prev < curr && !version.compare_exchange_weak(prev, curr));
			    inserter(get_volatile(head->block[i].first), V(), log);
			    count++;
			}
		    }
		}
		head = head->next;
		block_id++;
	    }
	}
	TIMER_STOP(restore_index, "restored keys = " << count);
	return version.load();
    }

    plog_t allocate() {
	plog_t ptr;
	pmem::obj::transaction::run(pool, [&] {
	    ptr = pmem::obj::make_persistent<log_t>();
	});
	return ptr;
    }
    void deallocate(plog_t ptr, bool cleanup = false) {
	if (cleanup)
	    return;
	pmem::obj::transaction::run(pool, [&] {
	    pmem::obj::delete_persistent<log_t>(ptr);
	});
    }
    void append(const K &key, plog_t kh) {
	pool.root()->keymap->append(key, kh);
    }
};

#endif //__PMEM_HISTORY
