#pragma once

#include <cstdint>
#include <iostream>

namespace OpenHardwareMonitor
{
    namespace Hardware
    {

        struct IOControlCode {
            uint32_t code;

            // Method enumeration
            enum class Method : uint32_t {
                Buffered = 0,
                InDirect = 1,
                OutDirect = 2,
                Neither = 3
            };

            // Access enumeration
            enum class Access : uint32_t {
                Any = 0,
                Read = 1,
                Write = 2
            };

            // Constructor with default method Buffered
            IOControlCode(uint32_t deviceType, uint32_t function, Access access)
                : IOControlCode(deviceType, function, Method::Buffered, access) {
            }

            // Constructor with specific method and access
            IOControlCode(uint32_t deviceType, uint32_t function, Method method, Access access) {
                code = (deviceType << 16) |
                    (static_cast<uint32_t>(access) << 14) |
                    (function << 2) |
                    static_cast<uint32_t>(method);
            }


        };


    } // namespace Hardware
} // namespace OpenHardwareMonitor
