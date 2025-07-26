
#ifndef IGROUP_H
#define IGROUP_H

#include <Interfaces/IHardware.h>

#include <string>
#include <vector>

namespace OpenHardwareMonitor {
    namespace Hardware {
        class IGroup {
        public:
            virtual ~IGroup() = default;

            // Pure virtual function to get all hardware in the group
            virtual std::vector<IHardware*> GetHardware() const = 0;

            // Pure virtual function to get a report of the group
            virtual std::string GetReport() const = 0;

            // Pure virtual function to close the group
            virtual void Close() = 0;
        };

    } // namespace Hardware
} // namespace OpenHardwareMonitor

#endif // GROUP_H