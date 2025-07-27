#include <Include/HelpersCommon.h>

#include <array>
#include <algorithm>

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
        return inStr.substr(i, num - i + 1);
    }


    std::string ToLowerString(std::string s)
    {
        std::transform(s.begin(), s.end(), s.begin(),
            [](unsigned char c) { return std::tolower(c); } // correct
        );
        return s;
    }



    static std::array<char,65> base64Table{
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
            'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
            'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
            'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7',
            '8', '9', '+', '/', '='
    };

    static size_t CalculateBase64OutputLength(size_t inputLength, bool insertLineBreaks)
    {
        size_t num = inputLength / 3ULL * 4;
        num += ((inputLength % 3 != 0) ? 4 : 0);
        if (num == 0L)
        {
            return 0;
        }

        if (insertLineBreaks)
        {
            size_t num2 = num / 76;
            if (num % 76 == 0ULL)
            {
                num2--;
            }

            num += num2 * 2;
        }


        return num;
    }

    static int _ConvertToBase64Array(std::string& outText, const std::vector<unsigned char>& inData, size_t offset, size_t length, bool insertLineBreaks)
    {
        size_t num = length % 3;
        size_t num2 = offset + (length - num);
        size_t num3 = 0;
        size_t num4 = 0;
        char* base64AlphabetPtr = &base64Table[0];
        {
            size_t i;
            for (i = offset; i < num2; i += 3)
            {
                if (insertLineBreaks)
                {
                    if (num4 == 76)
                    {
                        outText[num3++] = '\r';
                        outText[num3++] = '\n';
                        num4 = 0;
                    }

                    num4 += 4;
                }

                outText[num3] = base64AlphabetPtr[(inData[i] & 0xFC) >> 2];
                outText[num3 + 1] = base64AlphabetPtr[((inData[i] & 3) << 4) | ((inData[i + 1] & 0xF0) >> 4)];
                outText[num3 + 2] = base64AlphabetPtr[((inData[i + 1] & 0xF) << 2) | ((inData[i + 2] & 0xC0) >> 6)];
                outText[num3 + 3] = base64AlphabetPtr[inData[i + 2] & 0x3F];
                num3 += 4;
            }

            i = num2;
            if (insertLineBreaks && num != 0 && num4 == 76)
            {
                outText[num3++] = '\r';
                outText[num3++] = '\n';
            }

            switch (num)
            {
            case 2:
                outText[num3] = base64AlphabetPtr[(inData[i] & 0xFC) >> 2];
                outText[num3 + 1] = base64AlphabetPtr[((inData[i] & 3) << 4) | ((inData[i + 1] & 0xF0) >> 4)];
                outText[num3 + 2] = base64AlphabetPtr[(inData[i + 1] & 0xF) << 2];
                outText[num3 + 3] = base64AlphabetPtr[64];
                num3 += 4;
                break;
            case 1:
                outText[num3] = base64AlphabetPtr[(inData[i] & 0xFC) >> 2];
                outText[num3 + 1] = base64AlphabetPtr[(inData[i] & 3) << 4];
                outText[num3 + 2] = base64AlphabetPtr[64];
                outText[num3 + 3] = base64AlphabetPtr[64];
                num3 += 4;
                break;
            }
        }

        return num3;
    }

    std::string ToBase64STring(const std::vector<unsigned char>& data, bool insertLineBreaks)
    {
        size_t lenght = CalculateBase64OutputLength(data.size(), insertLineBreaks);
        std::string out = "";
        out.resize(lenght);
        _ConvertToBase64Array(out, data, 0, data.size(), insertLineBreaks);

        return out;
    }


    std::vector<std::string> SplitString(const std::string& fieldName, char delimiter, bool ignoreEmpty)
    {
        std::vector<std::string> parts;
        size_t lastDelim = 0;

        for (size_t i = 0; i < fieldName.size(); ++i)
        {
            if (fieldName[i] == delimiter)
            {
                parts.push_back(fieldName.substr(lastDelim, i - lastDelim));
                lastDelim = i + 1;
            }
            else if (i == (fieldName.size() - 1))
            {
                parts.push_back(fieldName.substr(lastDelim));
            }

            if (ignoreEmpty && parts.back() == "")
            {
                parts.pop_back();
            }
        }

        return parts;
    }
}