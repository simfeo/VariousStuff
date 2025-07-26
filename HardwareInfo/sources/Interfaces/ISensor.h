#ifndef ISENSOR_H
#define ISENSOR_H

#include "IHardware.h"
#include "IParameter.h"
#include "IControl.h"

#include <Include/Identifier.h>

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <chrono>
#include <optional>

namespace OpenHardwareMonitor {
    namespace Hardware {

        // Enum for Sensor types
        enum class SensorType {
            Voltage,      // V
            Clock,        // MHz
            Temperature,  // °C
            Load,         // %
            Fan,          // RPM
            Flow,         // L/h
            Control,      // %
            Level,        // %
            Factor,       // 1
            Power,        // W
            Data,         // GB = 2^30 Bytes
            SmallData,    // MB = 2^20 Bytes
            Throughput    // MB/s = 2^20 Bytes/s
        };

        // Struct to represent the SensorValue
        struct SensorValue {
        private:
            float value;
            std::chrono::system_clock::time_point time;

        public:
            SensorValue(float value, std::chrono::system_clock::time_point time)
                : value(value), time(time) {
            }

            float getValue() const { return value; }
            std::chrono::system_clock::time_point getTime() const { return time; }
        };

        // Abstract base class representing the ISensor interface
        class ISensor : public IElement {
        public:
            virtual ~ISensor() = default;

            // Getter for the Hardware
            virtual IHardware* getHardware() const = 0;

            // Getter for SensorType
            virtual SensorType getSensorType() const = 0;

            // Getter for Identifier
            virtual Identifier* getIdentifier() const = 0;

            // Getter and Setter for Name
            virtual std::string getName() const = 0;
            virtual void setName(const std::string& name) = 0;

            // Getter for Index
            virtual int getIndex() const = 0;

            // Getter for IsDefaultHidden
            virtual bool getIsDefaultHidden() const = 0;

            // Getter for Parameters (Read-only)
            virtual const std::vector<std::shared_ptr<IParameter>>& getParameters() const = 0;

            // Getter for Value
            virtual std::optional<float> getValue() const = 0;

            // Getter for Min
            virtual std::optional<float> getMin() const = 0;

            // Getter for Max
            virtual std::optional<float> getMax() const = 0;

            // Reset Min and Max values
            virtual void resetMin() = 0;
            virtual void resetMax() = 0;

            // Getter for Sensor Values
            virtual std::vector<SensorValue> getValues() const = 0;

            // Getter for Control
            virtual IControl* getControl() const = 0;
        };

    }  // namespace Hardware
}  // namespace OpenHardwareMonitor

#endif