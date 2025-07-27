#pragma once

#include <Interfaces/IElement.h>

#include <string>
#include <vector>
#include <memory>
#include <functional>


namespace OpenHardwareMonitor 
{
    namespace Hardware 
    {

        // Define the type for HardwareEventHandler delegate as a function type
        using HardwareEventHandler = std::function<void(IHardware*)>;

        // Abstract base class representing the IComputer interface
        class IComputer : public IElement {
        public:
            virtual ~IComputer() = default;

            // Getter for the Hardware array (as a vector of shared pointers in C++)
            virtual std::vector<std::shared_ptr<IHardware>> getHardware() const = 0;

            // Getter for MainboardEnabled
            virtual bool getMainboardEnabled() const = 0;

            // Getter for CPUEnabled
            virtual bool getCPUEnabled() const = 0;

            // Getter for RAMEnabled
            virtual bool getRAMEnabled() const = 0;

            // Getter for GPUEnabled
            virtual bool getGPUEnabled() const = 0;

            // Getter for FanControllerEnabled
            virtual bool getFanControllerEnabled() const = 0;

            // Getter for HDDEnabled
            virtual bool getHDDEnabled() const = 0;

            // Method to generate a report
            virtual std::string getReport() const = 0;

            // Event handler methods
            virtual void addHardwareAddedHandler(const HardwareEventHandler& handler) = 0;
            virtual void addHardwareRemovedHandler(const HardwareEventHandler& handler) = 0;
        };

    }  // namespace Hardware
}  // namespace OpenHardwareMonitor