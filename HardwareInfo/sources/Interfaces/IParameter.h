#ifndef IPARAMETER_H
#define IPARAMETER_H

#include "ISensor.h"
#include <Include/Identifier.h>

#include <string>

namespace OpenHardwareMonitor {
    namespace Hardware {

        // Abstract base class to act like an interface.
        class IParameter {
        public:
            virtual ~IParameter() = default;

            // Getter for the Sensor.
            virtual ISensor* getSensor() const = 0;

            // Getter for the Identifier.
            virtual Identifier* getIdentifier() const = 0;

            // Getter for the Name.
            virtual std::string getName() const = 0;

            // Getter for the Description.
            virtual std::string getDescription() const = 0;

            // Getter for the Value.
            virtual float getValue() const = 0;
            virtual void setValue(float value) = 0;

            // Getter for DefaultValue.
            virtual float getDefaultValue() const = 0;

            // Getter for IsDefault.
            virtual bool getIsDefault() const = 0;
            virtual void setIsDefault(bool isDefault) = 0;
        };

    }  // namespace Hardware
}  // namespace OpenHardwareMonitor

#endif // !IPARAMETER_H
