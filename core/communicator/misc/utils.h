// Copyright (c) 2025, TensorCast Team.

#ifndef CORE_COMMUNICATOR_MISC_UTILS_H_
#define CORE_COMMUNICATOR_MISC_UTILS_H_

extern "C" {
#include <assert.h>
#include <strings.h>
};
#include <algorithm>
#include <cstring>
#include <string>

#include "absl/log/absl_check.h"
#include "core/communicator/misc/common.h"

namespace tensorcast::communicator::misc {

template <class T>
inline void CLEAR(T& obj) {
  bzero(&obj, sizeof(obj));
}

template <class T>
inline void FREE_PTR(T*& ptr) {
  if (ptr != nullptr) {
    free(ptr);
    ptr = nullptr;
  }
}

template <class T>
void CLEAR_PTR(T* t) {
  bzero(t, sizeof(T));
}

template <class T>
void CLEAR_PTR(T* t, size_t s) {
  bzero(t, s);
}

template <class T>
void ALLOC(T** t) {
  *t = reinterpret_cast<T*>(malloc(sizeof(T)));
}

template <class T>
void ALLOC(T** t, int C) {
  *t = reinterpret_cast<T*>(malloc(sizeof(T) * C));
}

template <class T>
void CALLOC(T** t) {
  *t = reinterpret_cast<T*>(malloc(sizeof(T)));
  CLEAR_PTR(*t);
}

template <class T>
void CALLOC(T** t, uint64_t size) {
  *t = reinterpret_cast<T*>(malloc(size));
  CLEAR_PTR(*t);
}

template <class T>
void CALLOC_ALIGN(T** t) {
  int ret = posix_memalign(reinterpret_cast<void**>(t), 4096, sizeof(T));
  ABSL_CHECK(ret == 0) << "posix_memalign failed";
  CLEAR_PTR(*t);
}

template <class T>
void CALLOC_ALIGN(T** t, uint64_t size) {
  int ret = posix_memalign(reinterpret_cast<void**>(t), 4096, size);
  ABSL_CHECK(ret == 0) << "posix_memalign failed";
  CLEAR_PTR(*t);
}

static inline void STRCPY(uint8_t* t, const std::string& str) {
  strncpy(reinterpret_cast<char*>(t), str.c_str(), str.size());
  t[str.size()] = '\0';
}

static inline void STRCPY(char* t, const std::string& str) {
  strncpy(t, str.c_str(), str.size());
  t[str.size()] = '\0';
}

static inline void STRNCPY(char* t, const std::string& str, size_t size) {
  if (size == 0)
    return;
  const size_t n = std::min(str.size(), size - 1);
  if (n > 0) {
    std::memcpy(t, str.data(), n);
  }
  t[n] = '\0';
}

inline bool NEED_HUGEPAGE(size_t size) {
  return size > MB;
}

std::string ip2str(uint32_t ip);

std::string gid2str(uint8_t* gid);

uint32_t get_intf_ip();

std::string get_default_ip();

uint64_t get_us();

} // namespace tensorcast::communicator::misc

#endif // CORE_COMMUNICATOR_MISC_UTILS_H_
