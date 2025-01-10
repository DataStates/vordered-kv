#include <vector>
#include <map>
#include <thread>
#include <limits>
#include <random>
#include <algorithm>
#include <filesystem>

#include "dstates/marker.hpp"
#include "dstates/vordered_kv.hpp"
#include "dstates/lockedmap.hpp"
#include "dstates/sqlite_wrapper.hpp"
#include "dstates/rocksdb_wrapper.hpp"

#define __DEBUG
#include "dstates/debug.hpp"

typedef std::pair<int, int> intp_t;

static std::vector<intp_t> ref_vals;
static std::map<int, int> sorted_vals;
static const int N = 1000000;
static const int INIT = 0, RESTART = 1, SHORTCUT = 2;

void create_reference(int n) {
    std::mt19937 rng(112233L);
    ref_vals.resize(n);
    sorted_vals.clear();
    int i = 0;
    while (i < n) {
        int val = rng();
        auto it = sorted_vals.find(val);
        if (it != sorted_vals.end())
            continue;
        auto p = std::make_pair(val, val + 1);
        sorted_vals.insert(it, p);
        ref_vals[i] = p;
        i++;
    }
    std::cout << "Init complete, max element: " << sorted_vals.rbegin()->first + 1 << std::endl;
}

template <class Map> void run_extract_find(Map &vmap, int n, int r, int t) {
    std::atomic<int> extracted{0};
    int latest = vmap.latest();

    TIMER_START(t_extract);
    #pragma omp parallel num_threads(t)
    {
        std::mt19937 rng(334455 + omp_get_thread_num());
        std::vector<std::pair<int, int>> result;
        #pragma omp for
        for (int j = 0; j < t; j++) {
            result.clear();
            int v = rng() % r;
            vmap.get_snapshot(latest - v, result);
            extracted += result.size();
        }
    }
    TIMER_STOP(t_extract, "extract snapshot " << t << " times, level = " << r << ", items = " << extracted);

    std::atomic<int> non_empty{0};
    TIMER_START(t_item);
    #pragma omp parallel num_threads(t)
    {
        std::vector<std::pair<int, int>> result;
        #pragma omp for
        for (int i = 0; i < n; i++) {
            result.clear();
            vmap.get_key_history(ref_vals[i].first, result);
            if (result.size() != 0)
                non_empty++;
            else
                DBG("empty history for key: " << ref_vals[i].first);
        }
    }
    TIMER_STOP(t_item, "extract key history for " << n << " keys, non empty history = " << non_empty);

    std::atomic<int> found{0};
    TIMER_START(t_find);
    #pragma omp parallel num_threads(t)
    {
        std::mt19937 rng(445566 + omp_get_thread_num());
        #pragma omp for
        for (int i = 0; i < n; i++) {
            int v = rng() % r;
            if (vmap.find(latest - v, ref_vals[i].first) != marker_t<int>::low_marker)
                found++;
        }
    }
    TIMER_STOP(t_find, "find " << n << " KV pairs, level = " << r << ", items = " << found);
}

template <class Map> void run_insert(Map &vmap, const int n, const int t, const int ref_start = 0) {
    TIMER_START(t_insert);
    #pragma omp parallel num_threads(t)
    {
        #pragma omp for
        for (int i = 0; i < n; i++) {
            vmap.insert(ref_vals[ref_start + i].first, ref_vals[ref_start + i].second);
        }
    }
    TIMER_STOP(t_insert, "insert " << n << " KV pairs, ref_start = " << ref_start);
}

template <class Map> void run_remove(Map &vmap, const int n, const int r, const int t, const int ref_start = 0) {
    TIMER_START(t_remove);
    #pragma omp parallel num_threads(t)
    {
        #pragma omp for
        for (int i = 0; i < r; i++)
            vmap.remove(ref_vals[ref_start + i].first);
    }
    TIMER_STOP(t_remove, "remove " << r << " KV pairs, ref_start = " << ref_start);
}

template <class Map> void run_bench(Map &vmap, bool ephemeral, int bench_id, int N, int t) {
    if (bench_id == INIT) {
	DBG("initial bench: insert, remove, insert");
	run_insert(vmap, N, t);
	run_remove(vmap, N, N, t);
	run_insert(vmap, N, t, N);
	DBG("initial bench: get_snapshot, get_key_history, find");
	run_extract_find(vmap, N, vmap.latest(), t);
    } else if (bench_id == RESTART) {
	DBG("restart bench: get_snapshot, get_key_history, find");
	if (ephemeral)
	    DBG("skipped for ephemeral approach");
	else
	    run_extract_find(vmap, N, vmap.latest(), t);
    } else if (bench_id == SHORTCUT) {
	DBG("shortcut bench: insert, find, remove, find");
	run_insert(vmap, N, t);
	run_extract_find(vmap, N, 1, t);
	run_remove(vmap, N, int(std::trunc(N * 0.9)), t); // remove 90% of elements
	if constexpr(std::is_same<Map, vordered_kv_t<int, int, pmem_history_t<int, int>, true>>::value
		     || std::is_same<Map, vordered_kv_t<int, int, pmem_history_t<int, int>, false>>::value) {
	    vmap.scrub();
	    vmap.clear_stats();
	}
	run_extract_find(vmap, N, 1, t);
    } else
	FATAL("no valid bench ID specified: INIT, RESTART, SHORTCUT");
}

void run_for_approach(const std::string &approach, int bench_id, int N, int t, const std::string &db, bool shared) {
    if (approach == "e_vordered_kv_t") {
        vordered_kv_t<int, int, emem_history_t<int, int>, true, true> map(db);
	run_bench(map, true, bench_id, N, t);
    } else if (approach == "locked_map_t") {
        locked_map_t<int, int> map;
        run_bench(map, true, bench_id, N, t);
    } else if (approach == "p_vordered_kv_t") {
        vordered_kv_t<int, int, pmem_history_t<int, int>, false, true> map(db);
	run_bench(map, false, bench_id, N, t);
	DBG("stats: " << map.get_stats());
    } else if (approach == "ps_vordered_kv_t") {
        vordered_kv_t<int, int, pmem_history_t<int, int>, true, true> map(db);
	run_bench(map, false, bench_id, N, t);
	DBG("stats: " << map.get_stats());
    } else if (approach == "sqlite_wrapper_t") {
        sqlite_wrapper_t map(db, t, shared);
	run_bench(map, false, bench_id, N, t);
    } else if (approach == "rocksdb_wrapper_t") {
	rocksdb_wrapper_t<int, int> map(db);
	run_bench(map, false, bench_id, N, t);
    } else
        FATAL("no valid approach selected: skiptlist_t, locked_map_t, pskiplist_t, sqlitewrapper_t, rocksdb_wrapper_t");
}

int main(int argc, char **argv) {
    if (argc != 7) {
        std::cout << "Usage: " << argv[0] << " <min_threads> <max_threads> <approach> <bench> <db_path> <shared>" << std::endl;
        return -1;
    }
    int t_min = std::stoi(argv[1]), t_max = std::stoi(argv[2]);
    std::string approach(argv[3]), suite(argv[4]), db(argv[5]);
    bool shared = std::string(argv[6]).compare("true") == 0;

    DBG("Multi-threaded ordered map bench, threads: " << t_min << "/" << t_max);
    create_reference(2 * N);
    for (int t = t_min; t <= t_max; t <<= 1) {
        DBG("starting for " << approach << " with " << t << " threads");
	if (suite.compare("standard") == 0) {
	    run_for_approach(approach, INIT, N, t, db, shared);
	    run_for_approach(approach, RESTART, N, t, db, shared);
	} else if (suite.compare("shortcut") == 0)
	    run_for_approach(approach, SHORTCUT, N, t, db, shared);
	else
	    FATAL("no valid suite selected: standard, shortcut");
        std::filesystem::remove_all(db);
    }

    return 0;
}
