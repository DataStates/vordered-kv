#include <omp.h>
#include <mpi.h>

#include <vector>
#include <random>
#include <thread>

#include "dstates/marker.hpp"
#include "dstates/vordered_kv.hpp"
#include "dstates/lockedmap.hpp"
#include "dstates/sqlite_wrapper.hpp"

#define __DEBUG
#include "dstates/debug.hpp"

typedef std::pair<int, int> intp_t;
typedef std::vector<intp_t> result_t;

static result_t ref_vals;
static std::map<int, int> sorted_vals;
static const int N = 1000000;
static int rank = 0, no_ranks = 0;

void create_reference(int n) {
    std::mt19937 rng(rank);
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
}

template <class Map> void run_insert(Map &vmap, int n, int t) {
    TIMER_START(t_insert);
    #pragma omp parallel num_threads(t)
    {
        #pragma omp for
        for (int i = 0; i < n; i++)
            vmap.insert(ref_vals[i].first, ref_vals[i].second);
    }
    TIMER_STOP(t_insert, "inserted " << n << " KV pairs, rank = " << rank);
}

template <class Map> void run_find(Map &vmap, int n) {
    std::mt19937 rng(rank + 112233L);
    int query[2], count = 0;
    TIMER_START(t_find);
    for (int j = 0; j < n; j++) {
        if (rank == 0) {
            query[0] = rng();
            query[1] = rng() % vmap.latest();
        }
        MPI_Bcast(query, 2, MPI_INT, 0, MPI_COMM_WORLD);
        int local = vmap.find(query[1], query[0]), global;
        MPI_Reduce(&local, &global, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (global != marker_t<int>::low_marker)
            count++;
    }
    if (rank == 0)
        TIMER_STOP(t_find, "collective find " << n << " KV pairs, count = " << count);
}

template <class Map> void run_extract(Map &vmap, int v) {
    int sizes[no_ranks], offsets[no_ranks + 1];
    int query;

    TIMER_START(t_snap);
    if (rank == 0)
        query = v;
    MPI_Bcast(&query, 1, MPI_INT, 0, MPI_COMM_WORLD);
    result_t local_snap;
    vmap.get_snapshot(query, local_snap);
    int local_size = 2 * local_snap.size();
    MPI_Gather(&local_size, 1, MPI_INT, sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int global_size = 0, *global = nullptr;
    if (rank == 0) {
        offsets[0] = 0;
        for (int i = 1; i <= no_ranks; i++)
            offsets[i] = offsets[i - 1] + sizes[i - 1];
        global_size = offsets[no_ranks];
        global = new int[global_size];
    }
    MPI_Gatherv(&local_snap[0], local_size, MPI_INT, global, sizes, offsets, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        result_t temp;
        temp.reserve(global_size);
        for (int i = 1; i < no_ranks; i++) {
            int left = 0, right = offsets[i];
            temp.clear();
            while (left < (int)local_snap.size() && right < offsets[i + 1]) {
                if (local_snap[left].first <= global[right]) {
                    temp.emplace_back(local_snap[left]);
                    left++;
                } else {
                    temp.emplace_back(global[right], global[right + 1]);
                    right += 2;
                }
            }
            while (left < (int)local_snap.size()) {
                temp.emplace_back(local_snap[left]);
                left++;
            }
            while (right < offsets[i + 1]) {
                temp.emplace_back(global[right], global[right + 1]);
                right += 2;
            }
            local_snap.swap(temp);
        }
    }
    if (rank == 0) {
        delete []global;
        TIMER_STOP(t_snap, "collective extract, items = " << local_snap.size());
    }
}

template <class Map> void run_tests(Map &map, int n) {
    create_reference(N);
    run_insert(map, N, std::thread::hardware_concurrency());
    run_find(map, N);
    for (int j = 0; j < map.latest(); j+= map.latest() / 5)
        run_extract(map, j);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <db_path> <approach>" << std::endl;
        return -1;
    }
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &no_ranks);
    std::string db = std::string(argv[1]) + "/test.db." + std::to_string(rank);
    std::string approach(argv[2]);
    bool shared = argc > 3;

    DBG("Starting approach " << approach);
    if (approach == "skiplist_t") {
        vordered_kv_t<int, int, true, emem_history_t<int, int>> map(db);
        run_tests(map, N);
    } else if (approach =="pskiplist_t") {
        vordered_kv_t<int, int, true, pmem_history_t<int, int>> map(db);
        run_tests(map, N);
    } else if (approach == "locked_map_t") {
        locked_map_t<int, int> map;
        run_tests(map, N);
    } else if (approach == "sqlite_wrapper_t") {
        sqlite_wrapper_t map(db, std::thread::hardware_concurrency(), shared);
        run_tests(map, N);
    } else
        ERROR("no valid approach selected");

    MPI_Finalize();
    return 0;
}
