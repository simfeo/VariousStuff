#pragma once
#ifndef OPERATINGSYSTEM_H
#define OPERATINGSYSTEM_H
#include <Windows.h>


namespace OpenHardwareMonitor {
    namespace Hardware {

        class OperatingSystem {
        public:
            static bool IsUnix;
            static bool Is64BitOperatingSystem;

            static void Initialize();

        private:
            static bool GetIs64BitOperatingSystem();
            static DWORD getPlatform();
            static BOOL IsWow64Process(HANDLE hProcess, PBOOL Wow64Process);
        };

    } // namespace Hardware
} // namespace OpenHardwareMonitor

#endif // OPERATINGSYSTEM_H