#pragma once

#include <Interfaces/IVisitor.h>

namespace OpenHardwareMonitor
{
	namespace Hardware
	{

		class IElement {
		public:
			virtual ~IElement() = default;

			// accept visitor on this element
			virtual void Accept(const IVisitor* visitor) = 0;

			// call accept(visitor) on all child elements (called only from visitors)
			virtual void Traverse(const IVisitor* visitor) = 0;
		};
	}
}
