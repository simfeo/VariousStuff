#pragma once

#include <vector>
#include <stdexcept>
#include <iterator>

namespace OpenHardwareMonitor 
{
    namespace Collections 
    {

        template <typename T>
        class RingCollection 
        {
        private:
            std::vector<T> array;
            int head;
            int tail;
            int size;

            void clearReferences() {
                // Replace with default constructed values to clear references
                if (head < tail) {
                    for (int i = head; i < tail; ++i)
                        array[i] = T();
                }
                else {
                    for (int i = 0; i < tail; ++i)
                        array[i] = T();
                    for (int i = head; i < static_cast<int>(array.size()); ++i)
                        array[i] = T();
                }
            }

        public:
            RingCollection() : RingCollection(0) {}

            RingCollection(int capacity) : array(capacity), head(0), tail(0), size(0) {
                if (capacity < 0)
                    throw std::out_of_range("capacity must be non-negative");
            }

            int Capacity() const {
                return static_cast<int>(array.size());
            }

            void SetCapacity(int value) {
                if (value < 0)
                    throw std::out_of_range("capacity must be non-negative");

                std::vector<T> newArray(value);
                if (size > 0) {
                    if (head < tail) {
                        std::copy(array.begin() + head, array.begin() + tail, newArray.begin());
                    }
                    else {
                        int frontSize = static_cast<int>(array.size()) - head;
                        std::copy(array.begin() + head, array.end(), newArray.begin());
                        std::copy(array.begin(), array.begin() + tail, newArray.begin() + frontSize);
                    }
                }

                array = std::move(newArray);
                head = 0;
                tail = (size == value) ? 0 : size;
            }

            void Clear() {
                clearReferences();
                head = tail = size = 0;
            }

            void Append(const T& item) {
                if (size == static_cast<int>(array.size())) {
                    int newCapacity = static_cast<int>(array.size()) * 3 / 2;
                    if (newCapacity < static_cast<int>(array.size()) + 8)
                        newCapacity = static_cast<int>(array.size()) + 8;
                    SetCapacity(newCapacity);
                }

                array[tail] = item;
                tail = (tail + 1 == static_cast<int>(array.size())) ? 0 : tail + 1;
                ++size;
            }

            T Remove() {
                if (size == 0)
                    throw std::runtime_error("Collection is empty");

                T result = array[head];
                array[head] = T();
                head = (head + 1 == static_cast<int>(array.size())) ? 0 : head + 1;
                --size;
                return result;
            }

            int Count() const {
                return size;
            }

            T& operator[](int index) {
                if (index < 0 || index >= size)
                    throw std::out_of_range("index out of range");

                int i = head + index;
                if (i >= static_cast<int>(array.size()))
                    i -= static_cast<int>(array.size());
                return array[i];
            }

            const T& operator[](int index) const {
                if (index < 0 || index >= size)
                    throw std::out_of_range("index out of range");

                int i = head + index;
                if (i >= static_cast<int>(array.size()))
                    i -= static_cast<int>(array.size());
                return array[i];
            }

            T& First() {
                if (size == 0)
                    throw std::runtime_error("Collection is empty");
                return array[head];
            }

            const T& First() const {
                if (size == 0)
                    throw std::runtime_error("Collection is empty");
                return array[head];
            }

            T& Last() {
                if (size == 0)
                    throw std::runtime_error("Collection is empty");
                return array[(tail == 0) ? static_cast<int>(array.size()) - 1 : tail - 1];
            }

            const T& Last() const {
                if (size == 0)
                    throw std::runtime_error("Collection is empty");
                return array[(tail == 0) ? static_cast<int>(array.size()) - 1 : tail - 1];
            }

            // Iterator support
            class Iterator {
            private:
                const RingCollection& collection;
                int index;
            public:
                using iterator_category = std::forward_iterator_tag;
                using value_type = T;
                using difference_type = int;
                using pointer = const T*;
                using reference = const T&;

                Iterator(const RingCollection& collection, int index)
                    : collection(collection), index(index) {
                }

                reference operator*() const {
                    return collection[index];
                }

                pointer operator->() const {
                    return &collection[index];
                }

                Iterator& operator++() {
                    ++index;
                    return *this;
                }

                bool operator==(const Iterator& other) const {
                    return &collection == &other.collection && index == other.index;
                }

                bool operator!=(const Iterator& other) const {
                    return !(*this == other);
                }
            };

            Iterator begin() const {
                return Iterator(*this, 0);
            }

            Iterator end() const {
                return Iterator(*this, size);
            }
        };

    } // namespace Collections
} // namespace OpenHardwareMonitor
