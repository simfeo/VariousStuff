#pragma once

#include <Include/Hardware.h>
#include <Include/Sensor.h>

#include <string>
#include <Windows.h>


namespace OpenHardwareMonitor 
{
    namespace Hardware 
    {
        namespace RAM 
        {
            class GenericRAM : public Hardware 
            {
            public:
                // Constructor
                GenericRAM(const std::string& name, ISettings* settings);

                // Destructor
                virtual ~GenericRAM();

                // Get HardwareType
                virtual HardwareType GetHardwareType() const override
                {
                    return HardwareType::RAM;
                }

                // Update sensor values
                virtual void Update() override;

            private:

                Sensor loadSensor;
                Sensor usedMemory;
                Sensor availableMemory;

                static bool getMemoryStatus(MEMORYSTATUSEX& status);
            };
        }
    }
}
