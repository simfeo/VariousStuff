#include <Interfaces/ISettings.h>
#include <Interfaces/IControl.h>
#include <Interfaces/IVisitor.h>
#include <Interfaces/IParameter.h>


#include <Include/Sensor.h>
#include <Include/Hardware.h>

#include <Include/HelpersCommon.h>


#include <iostream>
#include <sstream>
#include <fstream>
#include <iterator>
#include <iomanip>

namespace OpenHardwareMonitor 
{
    namespace Hardware 
    {

        Sensor::Sensor(const std::string& name_, int index, SensorType sensorType, Hardware* hardware, ISettings* settings)
            : Sensor(name, index, false, sensorType, hardware, {}, settings) {
        }

        Sensor::Sensor(const std::string& name_, int index, SensorType sensorType, Hardware* hardware, std::vector<IParameter*> parameterDescriptions, ISettings* settings)
            : Sensor(name_, index, false, sensorType, hardware, parameterDescriptions, settings) {
        }

        Sensor::Sensor(const std::string& name_, int index, bool defaultHidden, SensorType sensorType, Hardware* hardware,
            std::vector<IParameter*> parameterDescriptions, ISettings* settings)
            : defaultName(name_), index(index), defaultHidden(defaultHidden), sensorType(sensorType), hardware(hardware),
            settings(settings), currentValue(std::make_optional(0.0f)), minValue(std::make_optional(0.0f)), maxValue(std::make_optional(0.0f)), sum(0), count(0)
        {
            // Initialize parameters
            for (auto& param : parameterDescriptions) {
                parameters.push_back(std::shared_ptr<IParameter>(param));
            }

            //name = settings->GetValue("Sensor:" + std::to_string(index) + ":name", name);
            name = settings->GetValue(Identifier(getIdentifier(), {"name"}).ToString(), name);
            GetSensorValuesFromSettings();

            // Hardware closing event handler
            std::function<void(Hardware*)> closing = [this](Hardware*) {
                this->SetSensorValuesToSettings();
                };
            hardware->AddClosingHandler(closing);
        }

        void Sensor::SetSensorValuesToSettings() {
            std::stringstream ss;
            std::chrono::system_clock::time_point t  = std::chrono::system_clock::now();
            for (const auto& sensorValue : values) {
                std::chrono::system_clock::time_point v = sensorValue.getTime();
                ss << (v - t) << " " << sensorValue.getValue() << "\n";
                t = v;
            }

            std::string valuesStr = ss.str();
            //settings.SetValue(new Identifier(Identifier, "values").ToString(),
            //    Convert.ToBase64String(m.ToArray()));
            settings->SetValue(Identifier(getIdentifier(), { "values" }).ToString(), valuesStr);
        }

        void Sensor::GetSensorValuesFromSettings() {
            //std::string valuesStr = settings->GetValue("Sensor:" + std::to_string(index) + ":values", "");
            std::string valuesStr = settings->GetValue(Identifier(getIdentifier(), { "values" }).ToString(), "");


            std::istringstream ss(valuesStr);
            long t = 0;
            float value;
            std::time_t time_t_val;
            while (ss >> time_t_val >> value) {
                std::chrono::system_clock::time_point time = std::chrono::system_clock::from_time_t(time_t_val);
                AppendValue(value, time);
            }
        }

        void Sensor::AppendValue(float value, std::chrono::system_clock::time_point time) {
            if (values.size() >= 2 && values.back().getValue() == value && values[values.size() - 1].getValue() == value) {
                values.back() = SensorValue(value, time);
                return;
            }
            values.push_back(SensorValue(value, time));
        }

        void Sensor::Accept(const IVisitor* visitor) {
            if (!visitor) throw std::invalid_argument("Visitor cannot be null");
            visitor->VisitSensor(this);
        }

        void Sensor::Traverse(const IVisitor* visitor) {
            for (const auto& param : parameters) {
                param->Accept(visitor);
            }
        }

        IHardware* Sensor::getHardware() const
        {
            return hardware;
        }

        SensorType Sensor::getSensorType() const {
            return sensorType;
        }

        Identifier Sensor::getIdentifier() const {
            //return hardware->GetIdentifier() + ":" + std::to_string(static_cast<int>(sensorType)) + ":" + std::to_string(index);
            return Identifier(hardware->GetIdentifier(), { OHM_H::ToLowerString(SensorTypeToString(sensorType)), std::to_string(index) });
        }

        std::string Sensor::getName() const {
            return name;
        }

        void Sensor::setName(const std::string& value) {
            name = !value.empty() ? value : defaultName;
            //settings->SetValue("Sensor:" + std::to_string(index) + ":name", name);

            settings->SetValue(Identifier(getIdentifier(), {"name"}).ToString(), name);
        }

        int Sensor::getIndex() const {
            return index;
        }

        bool Sensor::getIsDefaultHidden() const {
            return defaultHidden;
        }

        const std::vector<std::shared_ptr<IParameter>>& Sensor::getParameters() const {
            return parameters;
        }

        std::optional<float> Sensor::getValue() const {
            return currentValue;
        }

        void Sensor::SetValue(float value) {
            std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
            while (!values.empty() 
                && std::chrono::duration_cast<std::chrono::seconds>(now - values.front().getTime()).count() > 86400)
            { // 86400 seconds = 1 day
                values.erase(values.begin());
            }

            if (!std::isnan(value) || !std::isinf(value))
            {
                sum += value;
                count++;
                if (count == 4) {
                    AppendValue(sum / count, now);
                    sum = 0;
                    count = 0;
                }
            }

            currentValue = value;
            if (!minValue.has_value() || minValue > value) 
                minValue = value;
            if (!maxValue.has_value() || maxValue < value)
                maxValue = value;
        }

        std::optional<float> Sensor::getMin() const {
            return minValue;
        }

        std::optional<float> Sensor::getMax() const {
            return maxValue;
        }

        void Sensor::resetMin() {
            minValue = std::make_optional(0.0f);
        }

        void Sensor::resetMax() {
            maxValue = std::make_optional(0.0f);
        }

        std::vector<SensorValue> Sensor::getValues() const {
            std::vector<SensorValue> return_val;
            return_val.reserve(values.size());
            std::copy(values.begin(), values.end(), return_val.begin());
            return return_val;
        }

        IControl* Sensor::getControl() const {
            return control;
        }

        void Sensor::SetControl(IControl* control) {
            this->control = control;
        }

    }
}