#ifndef IHARDWARE_H
#define IHARDWARE_H

#include "IElement.h"
#include "ISensor.h"

#include <Include/Identifier.h>

#include <string>
#include <functional>

namespace OpenHardwareMonitor {
	namespace Hardware {

        using SensorEventHandler = std::function<void(ISensor*)>;

        // HardwareType enumeration
        enum class HardwareType {
            Mainboard,
            SuperIO,
            CPU,
            RAM,
            GpuNvidia,
            GpuAti,
            TBalancer,
            Heatmaster,
            HDD
        };

        // IHardware interface
        class IHardware : public IElement {
        public:
            virtual ~IHardware() = default;

            // Accessors
            virtual const std::string& GetName() const = 0;
            virtual void SetName(const std::string& name) = 0;

            virtual const Identifier& GetIdentifier() const = 0;
            virtual HardwareType GetHardwareType() const = 0;

            // Actions
            virtual std::string GetReport() const = 0;
            virtual void Update() = 0;

            // Hierarchy
            virtual const std::vector<IHardware*>& GetSubHardware() const = 0;
            virtual IHardware* GetParent() const = 0;

            // Sensors
            virtual const std::vector<ISensor*>& GetSensors() const = 0;

            // Event-like mechanism (delegate registration)
            virtual void AddSensorAddedHandler(SensorEventHandler handler) = 0;
            virtual void RemoveSensorAddedHandler(SensorEventHandler handler) = 0;

            virtual void AddSensorRemovedHandler(SensorEventHandler handler) = 0;
            virtual void RemoveSensorRemovedHandler(SensorEventHandler handler) = 0;
        };
	}
}

#endif // !IHARDWARE_H
