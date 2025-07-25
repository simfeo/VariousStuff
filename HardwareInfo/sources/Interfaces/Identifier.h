#ifndef IDENTIFIER_H
#define IDENTIFIER_H
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace OpenHardwareMonitor {
	namespace Hardware {

		class Identifier {
		private:
			std::string identifier;
			static constexpr char Separator = '/';

			// Helper: Validates identifier parts
			static void CheckIdentifiers(const std::vector<std::string>& identifiers) {
				for (const auto& s : identifiers) {
					if (s.find(' ') != std::string::npos || s.find(Separator) != std::string::npos) {
						throw std::invalid_argument("Invalid identifier: contains space or separator");
					}
				}
			}

			// Helper: Joins identifier parts
			static std::string BuildIdentifier(const std::vector<std::string>& parts) {
				std::ostringstream oss;
				for (const auto& part : parts) {
					oss << Separator << part;
				}
				return oss.str();
			}

		public:
			// Constructors
			Identifier(const std::vector<std::string>& identifiers) {
				CheckIdentifiers(identifiers);
				identifier = BuildIdentifier(identifiers);
			}

			Identifier(const Identifier& base, const std::vector<std::string>& extensions) {
				CheckIdentifiers(extensions);
				std::ostringstream oss;
				oss << base.ToString();
				for (const auto& ext : extensions) {
					oss << Separator << ext;
				}
				identifier = oss.str();
			}

			// ToString equivalent
			std::string ToString() const {
				return identifier;
			}

			// Comparison
			bool operator==(const Identifier& other) const {
				return identifier == other.identifier;
			}

			bool operator!=(const Identifier& other) const {
				return !(*this == other);
			}

			bool operator<(const Identifier& other) const {
				return identifier < other.identifier;
			}

			bool operator>(const Identifier& other) const {
				return identifier > other.identifier;
			}

			// For use in associative containers
			std::size_t GetHashCode() const {
				return std::hash<std::string>{}(identifier);
			}

			// For sorting (like IComparable)
			int CompareTo(const Identifier& other) const {
				return identifier.compare(other.identifier);
			}
		};
	}
}


#endif // !IDENTIFIER_H
