#ifndef RAMGROUP_H
#define RAMGROUP_H

#include <Include/Hardware.h>
#include <Include/SMBIOS.h>

#include <Interfaces/IGroup.h>

#include <vector>
#include <string>

namespace OpenHardwareMonitor {
    namespace Hardware {
        namespace RAM {

            // RAMGroup class
            class RAMGroup :public IGroup {
            private:
                std::vector<Hardware*> hardware;

            public:
                RAMGroup(SMBIOS* smbios, ISettings* settings);
                ~RAMGroup();

                virtual std::vector<IHardware*> GetHardware() const override;
                virtual std::string GetReport() const override;
                virtual void Close() override;
                
            };

        }  // namespace RAM
    }  // namespace Hardware
}  // namespace OpenHardwareMonitor

#endif // RAMGROUP_H