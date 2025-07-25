// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2009-2010 Michael Möller <mmoeller@openhardwaremonitor.org>
//

#ifndef IREADONLYARRAY_H
#define IREADONLYARRAY_H

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

#endif // IREADONLYARRAY_H