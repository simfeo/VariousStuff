#include <Include/GenericRam.h>
#include <iostream>
namespace OpenHardwareMonitor {
    namespace Hardware {
        namespace RAM {
            GenericRAM::GenericRAM(const std::string& name)
                : loadSensor("Memory"), usedMemory("Used Memory"), availableMemory("Available Memory") {

                // Set default values for sensors
                loadSensor.value = 0.0f;
                usedMemory.value = 0.0f;
                availableMemory.value = 0.0f;
            }

            GenericRAM::~GenericRAM() {}

            void GenericRAM::Update() {
                MEMORYSTATUSEX status;
                status.dwLength = sizeof(status);

                if (!getMemoryStatus(status)) {
                    std::cerr << "Failed to retrieve memory status!" << std::endl;
                    return;
                }

                // Update load sensor
                loadSensor.value = 100.0f - (100.0f * status.ullAvailPhys) / status.ullTotalPhys;

                // Update used memory sensor (in GB)
                usedMemory.value = static_cast<float>(status.ullTotalPhys - status.ullAvailPhys) / (1024 * 1024 * 1024);

                // Update available memory sensor (in GB)
                availableMemory.value = static_cast<float>(status.ullAvailPhys) / (1024 * 1024 * 1024);
            }

            bool GenericRAM::getMemoryStatus(MEMORYSTATUSEX& status) {
                return GlobalMemoryStatusEx(&status);
            }
        }
    }
}