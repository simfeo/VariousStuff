#include <Include/Hardware.h>
#include <iostream>

namespace OpenHardwareMonitor {
	namespace Hardware {

		Hardware::Hardware(const std::string& name, const Identifier& _identifier, ISettings* settings)
			: name(name), identifier(identifier), settings(settings) {
			customName = settings->GetValue(Identifier(identifier, { "name" }).ToString(), name);
		}

		IHardware* Hardware::GetParent() const {
			return nullptr;  // No parent by default
		}

		const std::vector<ISensor*>& Hardware::GetSensors() const {
			return active;
		}

		void Hardware::ActivateSensor(ISensor* sensor) {
			active.push_back(sensor);
			if (SensorAdded) {
				SensorAdded(sensor);
			}
		}

		void Hardware::DeactivateSensor(ISensor* sensor) {
			auto it = std::find(active.begin(), active.end(), sensor);
			if (it != active.end()) {
				active.erase(it);
				if (SensorRemoved) {
					SensorRemoved(sensor);
				}
			}
		}

		std::string Hardware::Name() const {
			return customName;
		}

		void Hardware::Name(const std::string& value) {
			if (!value.empty()) {
				customName = value;
			}
			else {
				customName = name;
			}
			settings->SetValue(Identifier(identifier, { "name" }).ToString(), customName);
		}

		const Identifier& Hardware::GetIdentifier() const {
			return identifier;
		}

		void Hardware::Close() {
			if (Closing) {
				Closing(this);
			}
		}

		void Hardware::Accept(const IVisitor* visitor) {
			if (!visitor) {
				throw std::invalid_argument("Visitor cannot be null");
			}
			visitor->VisitHardware(this);
		}

		void Hardware::Traverse(const IVisitor* visitor) {
			for (auto* sensor : active) {
				sensor->Accept(visitor);
			}
		}
	}
}