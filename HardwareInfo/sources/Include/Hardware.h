#pragma once

#include <Interfaces/IHardware.h>
#include <Interfaces/ISensor.h>
#include <Interfaces/IVisitor.h>
#include <Interfaces/ISettings.h>
#include <Interfaces/Delegate.h>

#include <Include/Identifier.h>


#include <string>
#include <vector>
#include <functional>

namespace OpenHardwareMonitor
{
    namespace Hardware
    {
        class Hardware : public IHardware 
        {
        protected:
            Identifier identifier;
            std::string name;
            std::string customName;
            ISettings* settings;
            std::vector<ISensor*> active;

            Delegate<void(ISensor*)> SensorAdded;
            Delegate<void(ISensor*)> SensorRemoved;

            Delegate<void(Hardware*)> Closing;


        public:
            Hardware(const std::string& name, const Identifier& identifier, ISettings* settings);

            //IEelement
            virtual void Accept(const IVisitor* visitor) override;
            virtual void Traverse(const IVisitor* visitor) override;

            // Override methods from IHardware
            virtual IHardware* GetParent() const override;
            virtual const std::vector<ISensor*>& GetSensors() const override;

            virtual const Identifier GetIdentifier() const override;

            virtual void AddSensorAddedHandler(const SensorEventHandler& handler) override;
            virtual void RemoveSensorAddedHandler(const SensorEventHandler& handler) override;

            virtual void AddSensorRemovedHandler(const SensorEventHandler& handler) override;
            virtual void RemoveSensorRemovedHandler(const SensorEventHandler& handler) override;

            // Custom Methods
            virtual void ActivateSensor(ISensor* sensor);
            virtual void DeactivateSensor(ISensor* sensor);

            virtual const std::string& GetName() const override;
            virtual void SetName(const std::string& name) override;

            virtual std::string GetReport() const { return ""; };

            void AddClosingHandler(const std::function<void(Hardware*)>& handler);
            virtual void Close();
            // Event Handlers
        };
    }
}