#ifndef IVISITOR_H
#define IVISITOR_H

#include "IComputer.h"
#include "IHardware.h"
#include "IParameter.h"
#include "ISensor.h"

namespace OpenHardwareMonitor {
	namespace Hardware {

		class IVisitor {
		public:
			virtual ~IVisitor() = default;

			virtual void VisitComputer(const IComputer* computer) const = 0;
			virtual void VisitHardware(const IHardware* hardware) const = 0;
			virtual void VisitSensor(const ISensor* sensor) const = 0;
			virtual void VisitParameter(const IParameter* parameter) const = 0;
		};

	} // namespace Hardware
} // namespace OpenHardwareMonitor
#endif