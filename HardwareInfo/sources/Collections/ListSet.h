#pragma once

#include <vector>
#include <algorithm>
#include <stdexcept>

namespace OpenHardwareMonitor 
{
    namespace Collections 
    {

        template <typename T>
        class ListSet {
        private:
            std::vector<T> list;

        public:
            // Add an item to the set (if it doesn't already exist)
            bool Add(const T& item) {
                if (std::find(list.begin(), list.end(), item) != list.end()) {
                    return false; // item already exists
                }
                list.push_back(item);
                return true;
            }

            // Remove an item from the set
            bool Remove(const T& item) {
                auto it = std::find(list.begin(), list.end(), item);
                if (it == list.end()) {
                    return false; // item not found
                }
                list.erase(it);
                return true;
            }

            // Check if the set contains the item
            bool Contains(const T& item) const {
                return std::find(list.begin(), list.end(), item) != list.end();
            }

            // Convert the set to a vector (array-like)
            std::vector<T> ToArray() const {
                return list; // simply return a copy of the vector
            }

            // Get the count of elements in the set
            int Count() const {
                return static_cast<int>(list.size());
            }

            // Iterator support
            class Iterator {
            private:
                typename std::vector<T>::const_iterator it;
            public:
                Iterator(typename std::vector<T>::const_iterator iter) : it(iter) {}

                const T& operator*() const {
                    return *it;
                }

                Iterator& operator++() {
                    ++it;
                    return *this;
                }

                bool operator==(const Iterator& other) const {
                    return it == other.it;
                }

                bool operator!=(const Iterator& other) const {
                    return it != other.it;
                }
            };

            Iterator begin() const {
                return Iterator(list.begin());
            }

            Iterator end() const {
                return Iterator(list.end());
            }
        };

    } // namespace Collections
} // namespace OpenHardwareMonitor
