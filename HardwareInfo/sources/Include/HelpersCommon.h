#pragma once

#include <string>
#include <vector>

namespace OHM_H
{
    std::string TrimString(const std::string& inStr);

    std::string ToLowerString(std::string s);

    std::string ToBase64STring(const std::vector<unsigned char>& data, bool insertLineBreaks = false);

    std::vector<std::string> SplitString(const std::string& fieldName, char delimiter, bool ignoreEmpty = true);
}