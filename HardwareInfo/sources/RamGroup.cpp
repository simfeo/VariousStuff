#include <Include/RAMGroup.h>
#include <Include/GenericRam.h>
#include <iostream>

namespace OpenHardwareMonitor {
    namespace Hardware {
        namespace RAM {
            // RAMGroup class implementation
            RAMGroup::RAMGroup(SMBIOS* smbios, ISettings* settings) {
                // No implementation for RAM on Unix systems
#ifdef __unix__
                hardware.clear();
#else
                hardware.push_back(new GenericRAM("Generic Memory"));
#endif
            }

            void RAMGroup::Close() {
                for (auto* ram : hardware) {
                    ram->Close();
                }
            }

            RAMGroup::~RAMGroup() {
                for (auto* ram : hardware) {
                    delete ram;
                }
            }

            std::vector<IHardware*> RAMGroup::GetHardware() const
            {
                return std::vector<IHardware*>(hardware.begin(), hardware.end());
            }

            std::string RAMGroup::GetReport() const
            {
                return "";
            }

        }  // namespace RAM
    }  // namespace Hardware
}  // namespace OpenHardwareMonitor
