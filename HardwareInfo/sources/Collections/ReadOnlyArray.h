#ifndef READONLYARRAY_H
#define READONLYARRAY_H

#include <vector>
#include <stdexcept>
#include <iterator>
#include <algorithm>

namespace OpenHardwareMonitor {
    namespace Collections {

        template <typename T>
        class ReadOnlyArray {
        private:
            const std::vector<T>& array;

        public:
            ReadOnlyArray(const std::vector<T>& arr) : array(arr) {}

            const T& operator[](int index) const {
                if (index < 0 || index >= static_cast<int>(array.size()))
                    throw std::out_of_range("index out of range");
                return array[index];
            }

            int Length() const {
                return static_cast<int>(array.size());
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
                return Iterator(array.begin());
            }

            Iterator end() const {
                return Iterator(array.end());
            }

            // Convert to a normal array
            std::vector<T> ToArray() const {
                return array; // No need to clone, just return the vector
            }

            // Implicit conversion to ReadOnlyArray from a std::vector
            static ReadOnlyArray<T> FromArray(const std::vector<T>& arr) {
                return ReadOnlyArray<T>(arr);
            }
        };

    } // namespace Collections
} // namespace OpenHardwareMonitor

#endif // READONLYARRAY_H
