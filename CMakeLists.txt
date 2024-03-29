cmake_minimum_required (VERSION 3.10)
project (DSTATES)

##### Global settings
include(GNUInstallDirs)
add_definitions(-O3 -g -Wall -D__BENCHMARK -D__ASSERT -D__INFO)
list(APPEND CMAKE_MODULE_PATH "${DSTATES_SOURCE_DIR}/cmake")
set(CMAKE_CXX_STANDARD 17)
option(BUILD_BENCHMARKS "Build benchmarks?")

# OpenMP needed for multi-threaded restart
find_package(OpenMP REQUIRED)
list(APPEND DSTATES_LIBS OpenMP::OpenMP_CXX)

# Needs LIBPMEMOBJ LIBPMEMOBJ++
find_package(LIBPMEMOBJ REQUIRED)
find_package(LIBPMEMOBJ++ REQUIRED)
list(APPEND DSTATES_LIBS ${LIBPMEMOBJ++_LIBRARIES})
include_directories(${LIBPMEMOBJ++_INCLUDE_DIRS})

# Optional packages (mostly needed for comparisons)
find_package(MPI)
find_package(SQLite3)
find_package(RocksDB)

# -----------------------------------------------------------------------------------
include_directories(BEFORE ${DSTATES_SOURCE_DIR}/include)
add_subdirectory(include)
add_subdirectory(tests)
if (BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif()
