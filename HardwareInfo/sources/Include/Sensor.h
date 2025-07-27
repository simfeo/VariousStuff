#pragma once

#include <Interfaces/ISensor.h>
#include <Include/Hardware.h>

#include <string>
#include <vector>
#include <ctime>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <chrono>

namespace OpenHardwareMonitor 
{
    namespace Hardware 
    {

        class Sensor : public ISensor 
        {
        private:
            std::string defaultName;
            std::string name;
            int index;
            bool defaultHidden;
            SensorType sensorType;
            Hardware* hardware;
            std::vector<std::shared_ptr<IParameter>> parameters;
            std::optional<float> currentValue{ std::nullopt };
            std::optional<float> minValue{ std::nullopt };
            std::optional<float> maxValue{ std::nullopt };
            std::vector<SensorValue> values;
            ISettings* settings;
            IControl* control;
            float sum;
            int count;

        public:
            Sensor(const std::string& name, int index, SensorType sensorType, Hardware* hardware, ISettings* settings);
            Sensor(const std::string& name, int index, SensorType sensorType, Hardware* hardware, std::vector<IParameter*> parameterDescriptions, ISettings* settings);
            Sensor(const std::string& name, int index, bool defaultHidden, SensorType sensorType, Hardware* hardware, std::vector<IParameter*> parameterDescriptions, ISettings* settings);

            // IElement
            virtual void Accept(const IVisitor* visitor) override;
            virtual void Traverse(const IVisitor* visitor) override;

            //ISensor
            virtual IHardware* getHardware() const override;
            virtual SensorType getSensorType() const override;
            virtual Identifier getIdentifier() const override;
            virtual std::string getName() const override;
            virtual void setName(const std::string& name) override;
            virtual int getIndex() const override;
            virtual bool getIsDefaultHidden() const override;
            virtual const std::vector<std::shared_ptr<IParameter>>& getParameters() const override;
            virtual std::optional<float> getValue() const override;
            virtual std::optional<float> getMin() const override;
            virtual std::optional<float> getMax() const override;
            virtual void resetMin() override;
            virtual void resetMax() override;
            virtual std::vector<SensorValue> getValues() const override;
            virtual IControl* getControl() const override;

            // own functions
            void SetSensorValuesToSettings();
            void GetSensorValuesFromSettings();
            void AppendValue(float value, std::chrono::system_clock::time_point time);
            void SetValue(float value);
            void SetControl(IControl* control);
        };
    }
}