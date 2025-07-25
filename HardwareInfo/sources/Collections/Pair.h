#ifndef IPAIR_H
#define IPAIR_H

#include <functional> // for std::hash
#include <cstddef>    // for std::size_t

namespace OpenHardwareMonitor {
	namespace Collections {
		template <typename F, typename S>
		class Pair {
		private:
			F first;
			S second;

		public:
			// Constructors
			Pair() = default;
			Pair(const F& first, const S& second) : first(first), second(second) {}

			// Accessors
			const F& First() const { return first; }
			void First(const F& value) { first = value; }

			const S& Second() const { return second; }
			void Second(const S& value) { second = value; }

			// Hash function similar to GetHashCode
			std::size_t GetHashCode() const {
				std::size_t h1 = firstHash();
				std::size_t h2 = secondHash();
				return h1 ^ h2;
			}

		private:
			std::size_t firstHash() const {
				return std::hash<F>{}(first);
			}

			std::size_t secondHash() const {
				return std::hash<S>{}(second);
			}
		};
	} // namespace Collections
} // namespace OpenHardwareMonitor

#endif