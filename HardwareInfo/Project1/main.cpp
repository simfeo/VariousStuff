#include <iostream>
#include <windows.h>
#include <psapi.h>
#include <winioctl.h>
#include <iphlpapi.h>
#include <comdef.h> // For COM error handling

// Function to get CPU temperature (requires additional libraries or hardware-specific methods)
// This is a placeholder; replace with your actual temperature retrieval method.
double getCpuTemperature() {

    return -1; // Placeholder value
}

// Function to get CPU utilization
double getCpuUtilization() {
    FILETIME idleTime, kernelTime, userTime;
    GetSystemTimes(&idleTime, &kernelTime, &userTime);

    ULARGE_INTEGER idle, kernel, user;
    idle.LowPart = idleTime.dwLowDateTime;
    idle.HighPart = idleTime.dwHighDateTime;
    kernel.LowPart = kernelTime.dwLowDateTime;
    kernel.HighPart = kernelTime.dwHighDateTime;
    user.LowPart = userTime.dwLowDateTime;
    user.HighPart = userTime.dwHighDateTime;

    ULARGE_INTEGER totalTime = idle;
    totalTime.QuadPart += kernel.QuadPart + user.QuadPart;

    if (totalTime.QuadPart == 0) return 0.0; // Avoid division by zero

    return (double)(totalTime.QuadPart - idle.QuadPart) / totalTime.QuadPart * 100.0;
}


// Function to get RAM utilization (simplified)
double getRamUtilization() {
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    return (double)(memInfo.ullTotalPhys - memInfo.ullAvailPhys) / memInfo.ullTotalPhys * 100.0;
}

// Function to get disk drive utilization (simplified - only shows free space)
void getDiskUtilization() {
    DWORD sectorsPerCluster, bytesPerSector, numberOfFreeClusters, totalNumberOfClusters;
    ULARGE_INTEGER freeBytesAvailable, totalNumberOfBytes, totalNumberOfBytesAvailable;

    for (int i = 0; i < 26; i++) { // Check drives A-Z
        std::wstring driveLetter = L"\\\\.\\" + std::wstring(1, (wchar_t)('A' + i));
        if (GetDiskFreeSpaceEx(driveLetter.c_str(), &freeBytesAvailable, &totalNumberOfBytes, &totalNumberOfBytesAvailable)) {
            std::wcout << L"Drive " << driveLetter << L": Free space: " << freeBytesAvailable.QuadPart << L" bytes" << std::endl;
        }
    }
}


// Function to get network utilization (requires more advanced techniques)
// This is a placeholder;  Network monitoring is complex and requires more advanced techniques.
void getNetworkUtilization() {
    std::cout << "Network utilization:  (Implementation needed)" << std::endl;
}


int main() {
    std::cout << "CPU Temperature: " << getCpuTemperature() << " °C" << std::endl;
    std::cout << "CPU Utilization: " << getCpuUtilization() << "%" << std::endl;
    std::cout << "RAM Utilization: " << getRamUtilization() << "%" << std::endl;
    getDiskUtilization();
    getNetworkUtilization();
    return 0;
}