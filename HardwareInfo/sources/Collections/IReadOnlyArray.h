#pragma once

#include <vector>
#include <stdexcept>

namespace OpenHardwareMonitor {
    namespace Collections {

        template <typename T>
        class IReadOnlyArray {
        public:
            virtual ~IReadOnlyArray() {}

            // Pure virtual function to access elements
            virtual const T& operator[](int index) const = 0;

            // Pure virtual function to get the length of the array
            virtual int Length() const = 0;

            // Optional: function to get a const iterator to begin and end
            virtual typename std::vector<T>::const_iterator begin() const = 0;
            virtual typename std::vector<T>::const_iterator end() const = 0;
        };

    } // namespace Collections
} // namespace OpenHardwareMonitor
