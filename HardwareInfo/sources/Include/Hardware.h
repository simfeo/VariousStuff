#ifndef HARDWARE_H
#define HARDWARE_H

#include <Interfaces/IHardware.h>
#include <Interfaces/ISensor.h>
#include <Interfaces/IVisitor.h>
#include <Interfaces/ISettings.h>

#include <Include/Identifier.h>


#include <string>
#include <vector>
#include <functional>

namespace OpenHardwareMonitor {
    namespace Hardware {
        class Hardware : public IHardware {
        protected:
            Identifier identifier;
            std::string name;
            std::string customName;
            ISettings* settings;
            std::vector<ISensor*> active;

        public:
            Hardware(const std::string& name, const Identifier& identifier, ISettings* settings);

            // Override methods from IHardware
            virtual IHardware* GetParent() const override;
            virtual const std::vector<ISensor*>& GetSensors() const override;

            // Custom Methods
            virtual void ActivateSensor(ISensor* sensor);
            virtual void DeactivateSensor(ISensor* sensor);

            std::string Name() const;
            void Name(const std::string& value);

            virtual const Identifier& GetIdentifier() const;

            // Events
            std::function<void(ISensor*)> SensorAdded;
            std::function<void(ISensor*)> SensorRemoved;

            virtual void Close();

            virtual void Accept(const IVisitor* visitor) override;
            virtual void Traverse(const IVisitor* visitor) override;

            virtual std::string GetReport() const { return ""; };

            // Event Handlers
            std::function<void(Hardware*)> Closing;
        };
    }
}

#endif // HARDWARE_H
