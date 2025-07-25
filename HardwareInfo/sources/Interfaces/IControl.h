#ifndef ICONTROL_H
#define ICONTROL_H

#include "IIdentifier.h"

namespace OpenHardwareMonitor {
    namespace Hardware {

        // Enum for Control Modes
        enum class ControlMode {
            Undefined,
            Software,
            Default
        };

        // Abstract base class representing the IControl interface
        class IControl {
        public:
            virtual ~IControl() = default;

            // Getter for the Identifier
            virtual Identifier* getIdentifier() const = 0;

            // Getter for ControlMode
            virtual ControlMode getControlMode() const = 0;

            // Getter for SoftwareValue
            virtual float getSoftwareValue() const = 0;

            // Method to set Default
            virtual void setDefault() = 0;

            // Getter for MinSoftwareValue
            virtual float getMinSoftwareValue() const = 0;

            // Getter for MaxSoftwareValue
            virtual float getMaxSoftwareValue() const = 0;

            // Method to set Software value
            virtual void setSoftware(float value) = 0;
        };

    }  // namespace Hardware
}  // namespace OpenHardwareMonitor
#endif // !ICONTROL_H
