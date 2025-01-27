#ifndef __EMEM_HISTORY
#define __EMEM_HISTORY

#include "ekey_history.hpp"

#include <memory>
#include <functional>

template <typename K, typename V> class emem_history_t {
public:
    typedef ekey_history_t<V> log_t;
    typedef log_t* plog_t;

    emem_history_t(const std::string &db_name, size_t db_size) { }
    ~emem_history_t() { }
    int restore(std::function<bool (const K &, const V &, plog_t)> inserter) {
	return 0;
    }
    plog_t allocate() {
	return new log_t();
    }
    void deallocate(plog_t ptr, bool cleanup = false) {
	delete ptr;
    }
    void append(const K &key, plog_t kh) { }
};

#endif // __EMEM_HISTORY
