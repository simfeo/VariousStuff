#include <Include/SMBIOS.h>
#include <iostream>

int main(int argc, char** argv)
{
    //should be enableing computer
    OpenHardwareMonitor::Hardware::SMBIOS bios;
    std::cout << bios.GetReport()<<std::endl;

    return 0;
}