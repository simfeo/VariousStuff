#pragma once

#include <string>

namespace OpenHardwareMonitor 
{
    namespace Hardware 
    {

        class ISettings 
        {
        public:
            virtual ~ISettings() = default;

            virtual bool Contains(const std::string& name) = 0;
            virtual void SetValue(const std::string& name, const std::string& value) = 0;
            virtual std::string GetValue(const std::string& name, const std::string& defaultValue) = 0;
            virtual void Remove(const std::string& name) = 0;
        };

    } // namespace Hardware
} // namespace OpenHardwareMonitor