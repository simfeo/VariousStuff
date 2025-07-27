#include <Include/GenericRam.h>
#include <iostream>

namespace OpenHardwareMonitor 
{
    namespace Hardware 
    {
        namespace RAM 
        {
            GenericRAM::GenericRAM(const std::string& name, ISettings* settings)
                : Hardware(name, Identifier({ "ram" }), settings)
                , loadSensor ("Memory", 0, SensorType::Load, this, settings)
                , usedMemory ("Used Memory", 0, SensorType::Data, this, settings)
                , availableMemory ("Available Memory", 1, SensorType::Data, this, settings)
            {
                
                ActivateSensor(&loadSensor);

                ActivateSensor(&usedMemory);

                ActivateSensor(&availableMemory);
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
                loadSensor.SetValue( 100.0f - (100.0f * status.ullAvailPhys) / status.ullTotalPhys);

                // Update used memory sensor (in GB)
                usedMemory.SetValue( static_cast<float>(status.ullTotalPhys - status.ullAvailPhys) / (1024.0f * 1024.0f * 1024.0f));

                // Update available memory sensor (in GB)
                availableMemory.SetValue(static_cast<float>(status.ullAvailPhys) / (1024.0f * 1024.0f * 1024.0f));
            }

            bool GenericRAM::getMemoryStatus(MEMORYSTATUSEX& status) {
                return GlobalMemoryStatusEx(&status);
            }
        }
    }
}