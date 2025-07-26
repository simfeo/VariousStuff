#ifndef HELPERSCOMMON_H
#define HELPERSCOMMON_H

#include <string>

namespace OHM_H
{
    std::string TrimString(const std::string& inStr)
    {
        if (!inStr.length())
        {
            return "";
        }
        size_t num = inStr.length() - 1;
        size_t i = 0;

        for (i = 0; i < inStr.length() && (std::isspace(inStr.at(i))); i++)
        {
        }

        while (num >= i && (std::isspace(inStr.at(num))))
        {
            num--;
        }
        return inStr.substr(i, num - i);
    }
}
#endif // !HELPERSCOMMON_H
