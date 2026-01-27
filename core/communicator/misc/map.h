
// Copyright (c) 2025-2026, TensorCast Team.

#ifndef CORE_COMMUNICATOR_MISC_MAP_H_
#define CORE_COMMUNICATOR_MISC_MAP_H_

#include <mutex>
#include <unordered_map>
#include <vector>
#include "absl/log/log.h"

namespace tensorcast::communicator::misc {

template <class K, class V>
class Map {
 public:
  void put(K key, V value) {
    std::unique_lock<std::mutex> lock(mu_);
    map_.insert(std::make_pair(key, value));
  }

  bool exist(K key) {
    std::unique_lock<std::mutex> lock(mu_);
    return map_.find(key) != map_.end();
  }

  V get(K key) {
    std::unique_lock<std::mutex> lock(mu_);
    auto itr = map_.find(key);
    if (itr == map_.end()) {
      return V();
    }
    return itr->second;
  }

  void del(K key) {
    std::unique_lock<std::mutex> lock(mu_);
    auto itr = map_.find(key);
    if (itr != map_.end()) {
      map_.erase(itr);
    } else {
      LOG(FATAL) << "failed to delete key " << key << " from map";
    }
  }

  bool erase_if(const K& key, const V& expected) {
    std::unique_lock<std::mutex> lock(mu_);
    auto itr = map_.find(key);
    if (itr == map_.end()) {
      return false;
    }
    if (itr->second != expected) {
      return false;
    }
    map_.erase(itr);
    return true;
  }

  bool erase_if_present(const K& key) {
    std::unique_lock<std::mutex> lock(mu_);
    return map_.erase(key) > 0;
  }

  std::vector<std::pair<K, V>> pairs() {
    std::vector<std::pair<K, V>> pairs;
    std::unique_lock<std::mutex> lock(mu_);
    for (auto itr : map_) {
      pairs.push_back(std::make_pair(itr.first, itr.second));
    }
    return pairs;
  }

  void clear() {
    std::unique_lock<std::mutex> lock(mu_);
    map_.clear();
  }

 private:
  mutable std::mutex mu_;
  std::unordered_map<K, V> map_;
};

} // namespace tensorcast::communicator::misc

#endif // COMMUNICATOR_MISC_MAP_H_
