#ifndef GROUPAFFINITY_H
#define GROUPAFFINITY_H

#include <limits>
#include <cstdint>

namespace OpenHardwareMonitor {
	namespace Hardware {
		struct GroupAffinity {
			uint16_t Group;
			uint64_t Mask;

			const static GroupAffinity Undefined;

			GroupAffinity(uint16_t group, uint64_t mask);

			static GroupAffinity Single(uint16_t group, int index);

			bool Equals(const GroupAffinity& a1);

			int GetHashCode();

			bool operator ==(const GroupAffinity& a2);
			bool operator !=(const GroupAffinity& a2);
		};
	}
}


#endif // !GROUPAFFINITY_H


