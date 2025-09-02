#include <Include/GroupAffinity.h>

namespace OpenHardwareMonitor
{
	namespace Hardware
	{
		GroupAffinity Undefined = GroupAffinity(std::numeric_limits<uint16_t>::max(), 0);

		inline GroupAffinity::GroupAffinity(uint16_t group, uint64_t mask) {
			Group = group;
			Mask = mask;
		}

		inline GroupAffinity GroupAffinity::Single(uint16_t group, int index) {
			return GroupAffinity(group, 1ULL << index);
		}

		inline bool GroupAffinity::Equals(const GroupAffinity& a1) {
			return (Group == a1.Group) && (Mask == a1.Mask);
		}

		inline int GroupAffinity::GetHashCode() {
			return (int)Group ^ ((int)Mask ^ (int)(Mask >> 32));
		}

		bool GroupAffinity::operator ==(const GroupAffinity& a2) {
			return (Group == a2.Group) && (Mask == a2.Mask);
		}

		bool GroupAffinity::operator !=(const GroupAffinity& a2) {
			return (Group != a2.Group) || (Mask != a2.Mask);
		}
	}
}
