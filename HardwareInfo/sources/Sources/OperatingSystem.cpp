#include <Include/OperatingSystem.h>

#include <tchar.h>
#include <iostream>

namespace OpenHardwareMonitor {
    namespace Hardware {

        bool OperatingSystem::IsUnix = false;
        bool OperatingSystem::Is64BitOperatingSystem = false;

        void OperatingSystem::Initialize() {
#ifdef __unix__
            IsUnix = true;
#else
            Is64BitOperatingSystem = GetIs64BitOperatingSystem();
#endif
        }

        bool OperatingSystem::GetIs64BitOperatingSystem() {
            SYSTEM_INFO systemInfo;
            GetNativeSystemInfo(&systemInfo);

            // If pointer size is 8 bytes, it is a 64-bit OS.
            if (systemInfo.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_AMD64) {
                return true;
            }

            try {
                BOOL isWow64 = FALSE;
                BOOL result = IsWow64Process(GetCurrentProcess(), &isWow64);
                return (result && isWow64);
            }
            catch (...) {
                return false;
            }
        }


        BOOL OperatingSystem::IsWow64Process(HANDLE hProcess, PBOOL Wow64Process)
        {
            BOOL bIs64BitOS = FALSE;

            // We check if the OS is 64 Bit
            typedef BOOL(WINAPI* LPFN_ISWOW64PROCESS) (HANDLE, PBOOL);

            LPFN_ISWOW64PROCESS fnIsWow64Process = (LPFN_ISWOW64PROCESS)GetProcAddress(GetModuleHandle((TEXT("kernel32"))), "IsWow64Process");

            if (NULL != fnIsWow64Process)
            {
                if (!fnIsWow64Process(GetCurrentProcess(), &bIs64BitOS))
                {
                    std::cout << "Cannot obtain 32 or 64 bit" << std::endl;
                }
            }
            return bIs64BitOS;
        }

    } // namespace Hardware
} // namespace OpenHardwareMonitor