#include <Include/Identifier.h>
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace OpenHardwareMonitor {
    namespace Hardware {

        // Helper: Validates identifier parts
        void Identifier::CheckIdentifiers(const std::vector<std::string>& identifiers) {
            for (const auto& s : identifiers) {
                if (s.find(' ') != std::string::npos || s.find(Separator) != std::string::npos) {
                    throw std::invalid_argument("Invalid identifier: contains space or separator");
                }
            }
        }

        // Helper: Joins identifier parts
        std::string Identifier::BuildIdentifier(const std::vector<std::string>& parts) {
            std::ostringstream oss;
            for (const auto& part : parts) {
                oss << Separator << part;
            }
            return oss.str();
        }

        // Constructors
        Identifier::Identifier(const std::vector<std::string>& identifiers) {
            CheckIdentifiers(identifiers);
            identifier = BuildIdentifier(identifiers);
        }

        Identifier::Identifier(const Identifier& base, const std::vector<std::string>& extensions) {
            CheckIdentifiers(extensions);
            std::ostringstream oss;
            oss << base.ToString();
            for (const auto& ext : extensions) {
                oss << Separator << ext;
            }
            identifier = oss.str();
        }

        // ToString equivalent
        std::string Identifier::ToString() const {
            return identifier;
        }

        // Comparison
        bool Identifier::operator==(const Identifier& other) const {
            return identifier == other.identifier;
        }

        bool Identifier::operator!=(const Identifier& other) const {
            return !(*this == other);
        }

        bool Identifier::operator<(const Identifier& other) const {
            return identifier < other.identifier;
        }

        bool Identifier::operator>(const Identifier& other) const {
            return identifier > other.identifier;
        }

        // For use in associative containers
        std::size_t Identifier::GetHashCode() const {
            return std::hash<std::string>{}(identifier);
        }

        // For sorting (like IComparable)
        int Identifier::CompareTo(const Identifier& other) const {
            return identifier.compare(other.identifier);
        }

    } // namespace Hardware
} // namespace OpenHardwareMonitor
