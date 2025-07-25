#ifndef IDENTIFIER_H
#define IDENTIFIER_H

#include <string>
#include <vector>

namespace OpenHardwareMonitor {
    namespace Hardware {

        class Identifier {
        private:
            std::string identifier;
            static constexpr char Separator = '/';

            // Helper: Validates identifier parts
            static void CheckIdentifiers(const std::vector<std::string>& identifiers);

            // Helper: Joins identifier parts
            static std::string BuildIdentifier(const std::vector<std::string>& parts);

        public:
            // Constructors
            Identifier(const std::vector<std::string>& identifiers);
            Identifier(const Identifier& base, const std::vector<std::string>& extensions);

            // ToString equivalent
            std::string ToString() const;

            // Comparison
            bool operator==(const Identifier& other) const;
            bool operator!=(const Identifier& other) const;
            bool operator<(const Identifier& other) const;
            bool operator>(const Identifier& other) const;

            // For use in associative containers
            std::size_t GetHashCode() const;

            // For sorting (like IComparable)
            int CompareTo(const Identifier& other) const;
        };

    } // namespace Hardware
} // namespace OpenHardwareMonitor

#endif // !IDENTIFIER_H