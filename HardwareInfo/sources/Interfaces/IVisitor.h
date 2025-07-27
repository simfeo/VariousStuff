#pragma once

namespace OpenHardwareMonitor 
{
	namespace Hardware 
	{
		class IComputer;
		class IHardware;
		class ISensor;
		class IParameter;

		class IVisitor 
		{
		public:
			virtual ~IVisitor() = default;

			virtual void VisitComputer(const IComputer* computer) const = 0;
			virtual void VisitHardware(const IHardware* hardware) const = 0;
			virtual void VisitSensor(const ISensor* sensor) const = 0;
			virtual void VisitParameter(const IParameter* parameter) const = 0;
		};

	} // namespace Hardware
} // namespace OpenHardwareMonitor
