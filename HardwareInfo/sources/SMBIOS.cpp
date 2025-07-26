#include <Include/SMBIOS.h>
#include <Include/HelpersCommon.h>

#include <sstream>

#include <Windows.h>
#include <wbemidl.h>
#include <iostream>
#include <vector>
#include <comdef.h>

// Link with wbemuuid.lib for WMI support
#pragma comment(lib, "wbemuuid.lib")


namespace OpenHardwareMonitor {
    namespace Hardware {

        // Constructor
        SMBIOS::SMBIOS() : version(nullptr), biosInformation(nullptr), systemInformation(nullptr), baseBoardInformation(nullptr), processorInformation(nullptr) {
#ifdef __unix__
//later
#else
            ProcessRawSMBios();
#endif
        }

        // Process SMBIOS raw data (Windows specific)
        void SMBIOS::ProcessRawSMBios() {
            GetRawData();
            if (raw.size())
            {
                FillStructs();
            }
        }

        void SMBIOS::GetRawData()
        {
            raw.clear();
            BYTE majorVersion = 0;
            BYTE minorVersion = 0;

            // COM initialization
            HRESULT hres = CoInitializeEx(0, COINIT_MULTITHREADED);
            if (FAILED(hres)) {
                std::cerr << "COM library initialization failed." << std::endl;
                return;
            }

            // Set security levels
            hres = CoInitializeSecurity(
                NULL, -1, NULL, NULL, RPC_C_AUTHN_LEVEL_DEFAULT, RPC_C_IMP_LEVEL_IMPERSONATE,
                NULL, EOAC_NONE, NULL);
            if (FAILED(hres)) {
                std::cerr << "Security initialization failed." << std::endl;
                CoUninitialize();
                return;
            }

            IWbemLocator* pLoc = NULL;
            IWbemServices* pSvc = NULL;

            // Create WMI locator
            hres = CoCreateInstance(CLSID_WbemLocator, 0, CLSCTX_INPROC_SERVER, IID_IWbemLocator, (LPVOID*)&pLoc);
            if (FAILED(hres)) {
                std::cerr << "WMI locator creation failed." << std::endl;
                CoUninitialize();
                return;
            }

            // Connect to WMI
            hres = pLoc->ConnectServer(
                _bstr_t(L"ROOT\\WMI"), // WMI namespace
                NULL, NULL, 0, NULL, 0, 0, &pSvc);
            if (FAILED(hres)) {
                std::cerr << "WMI connection failed." << std::endl;
                pLoc->Release();
                CoUninitialize();
                return;
            }

            // Set security levels
            hres = CoSetProxyBlanket(
                pSvc, RPC_C_AUTHN_WINNT, RPC_C_AUTHZ_NONE, NULL, RPC_C_AUTHN_LEVEL_CALL,
                RPC_C_IMP_LEVEL_IMPERSONATE, NULL, EOAC_NONE);
            if (FAILED(hres)) {
                std::cerr << "Failed to set proxy blanket." << std::endl;
                pSvc->Release();
                pLoc->Release();
                CoUninitialize();
                return;
            }

            // Execute WMI query to fetch raw SMBIOS data
            IEnumWbemClassObject* pEnumerator = NULL;
            hres = pSvc->ExecQuery(
                bstr_t("WQL"),
                bstr_t("SELECT * FROM MSSMBios_RawSMBiosTables"),
                WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY,
                NULL, &pEnumerator);

            if (FAILED(hres)) {
                std::cerr << "Query for WMI data failed." << std::endl;
                pSvc->Release();
                pLoc->Release();
                CoUninitialize();
                return;
            }

            // Process the WMI query results
            IWbemClassObject* pclsObj = NULL;
            ULONG uReturn = 0;

            while (pEnumerator) {
                hres = pEnumerator->Next(WBEM_INFINITE, 1, &pclsObj, &uReturn);
                if (0 == uReturn) break;

                VARIANT vtProp;
                // Retrieve SMBiosData
                hres = pclsObj->Get(L"SMBiosData", 0, &vtProp, 0, 0);
                if (SUCCEEDED(hres) && (vtProp.vt == (VT_ARRAY | VT_UI1))) {
                    SAFEARRAY* pSafeArray = vtProp.parray;
                    long lLbound, lUbound;
                    SafeArrayGetLBound(pSafeArray, 1, &lLbound);
                    SafeArrayGetUBound(pSafeArray, 1, &lUbound);

                    // Fill raw vector with SMBiosData
                    raw.resize(lUbound - lLbound + 1);
                    memcpy(&raw[0], pSafeArray->pvData, raw.size());
                }
                VariantClear(&vtProp);

                // Retrieve SmbiosMajorVersion
                hres = pclsObj->Get(L"SmbiosMajorVersion", 0, &vtProp, 0, 0);
                if (SUCCEEDED(hres)) {
                    majorVersion = (BYTE)vtProp.intVal;
                }
                VariantClear(&vtProp);

                // Retrieve SmbiosMinorVersion
                hres = pclsObj->Get(L"SmbiosMinorVersion", 0, &vtProp, 0, 0);
                if (SUCCEEDED(hres)) {
                    minorVersion = (BYTE)vtProp.intVal;
                }
                VariantClear(&vtProp);

                pclsObj->Release();
            }

            // Cleanup
            pSvc->Release();
            pLoc->Release();
            pEnumerator->Release();
            CoUninitialize();
        }

        void SMBIOS::FillStructs()
        {
            std::vector<std::shared_ptr<Structure>> structureList{};
            std::vector< std::shared_ptr<MemoryDevice>> memoryDeviceList{};

            if (raw.size()) {
                int offset = 0;
                byte type = raw[offset];
                while (offset + 4 < raw.size() && type != 127) {

                    type = raw[offset];
                    int length = raw[offset + 1];
                    unsigned short handle = (unsigned short)((raw[offset + 2] << 8) | raw[offset + 3]);

                    if (offset + length > raw.size())
                        break;
                    std::vector<byte> data (length);

                    auto iter_first = raw.begin();
                    auto iter_last = raw.begin();
                    std::advance(iter_first, offset);
                    std::advance(iter_last, offset+length);
                    
                    offset += length;

                    std::vector<std::string> stringsList{};
                    if (offset < raw.size() && raw[offset] == 0)
                        offset++;

                    while (offset < raw.size() && raw[offset] != 0) {
                        std::stringstream sb{};
                        while (offset < raw.size() && raw[offset] != 0) {
                            sb<<(char)raw[offset];
                            offset++;
                        }
                        offset++;
                        stringsList.emplace_back(std::move(sb.str()));
                    }
                    offset++;
                    switch (type) 
                    {
                    case 0x00:
                        biosInformation.reset( new BIOSInformation(
                            type, handle, data, stringsList) );
                        structureList.push_back(biosInformation);
                        break;
                    case 0x01:
                        systemInformation.reset( new SystemInformation(
                            type, handle, data, stringsList) );
                        structureList.push_back(systemInformation);
                        break;
                    case 0x02:
                        baseBoardInformation.reset(new BaseBoardInformation(
                        type, handle, data, stringsList));
                        structureList.push_back(baseBoardInformation);
                        break;
                    case 0x04:
                        processorInformation.reset(new ProcessorInformation(
                        type, handle, data, stringsList));
                        structureList.push_back(processorInformation);
                        break;
                    case 0x11:
                        memoryDeviceList.push_back(
                            std::shared_ptr<MemoryDevice>(
                                new MemoryDevice(
                                    type, handle, data, stringsList)));
                        structureList.push_back(memoryDeviceList[memoryDeviceList.size()-1]);
                        break;
                    default:
                        structureList.push_back(std::shared_ptr<Structure>(new Structure(
                            type, handle, data, stringsList)));
                        break;
                    }
                }
            }

            memoryDevices = memoryDeviceList;
            table = structureList;
        }

        // Function to generate a formatted report of the SMBIOS information
        std::string SMBIOS::GetReport() const {
            std::ostringstream report;

            if (version != nullptr) {
                report << "SMBIOS Version: " << version->ToString(2) << std::endl;
            }

            if (BIOS != nullptr) {
                report << "BIOS Vendor: " << biosInformation->GetVendor() << std::endl;
                report << "BIOS Version: " << biosInformation->GetVersion() << std::endl;
            }

            if (System != nullptr) {
                report << "System Manufacturer: " << systemInformation->GetManufacturerName() << std::endl;
                report << "System Name: " << systemInformation->GetProductName() << std::endl;
                report << "System Version: " << systemInformation->GetVersion() << std::endl;
            }

            if (Board != nullptr) {
                report << "Mainboard Manufacturer: " << baseBoardInformation->GetManufacturerName() << std::endl;
                report << "Mainboard Name: " << baseBoardInformation->GetProductName() << std::endl;
                report << "Mainboard Version: " << baseBoardInformation->GetVersion() << std::endl;
            }

            if (Processor != nullptr) {
                report << "Processor Manufacturer: " << processorInformation->GetManufacturerName() << std::endl;
                report << "Processor Version: " << processorInformation->GetVersion() << std::endl;
            }

            for (size_t i = 0; i < memoryDevices.size(); ++i) {
                const auto& memDevice = memoryDevices[i];
                report << "Memory Device [" << i << "] Manufacturer: " << memDevice->GetManufacturerName() << std::endl;
                report << "Memory Device [" << i << "] Part Number: " << memDevice->GetPartNumber() << std::endl;
            }

            // Add raw SMBios table if available
            if (!raw.empty()) {
                report << "SMBIOS Table" << std::endl;
                std::string base64 = "SMBIOS Base64 Representation"; // Placeholder for actual base64 encoding
                for (size_t i = 0; i < std::ceil(base64.length() / 64.0); ++i) {
                    report << " ";
                    for (size_t j = 0; j < 0x40; ++j) {
                        size_t index = (i << 6) | j;
                        if (index < base64.length()) {
                            report << base64[index];
                        }
                    }
                    report << std::endl;
                }
            }

            return report.str();
        }

        const BIOSInformation* SMBIOS::BIOS()
        {
            assert(false);
            return nullptr;
        }

        const SystemInformation* SMBIOS::System()
        {
            assert(false);
            return nullptr;
        }

        const BaseBoardInformation* SMBIOS::Board()
        {
            assert(false);
            return nullptr;
        }

        const ProcessorInformation* SMBIOS::Processor()
        {
            assert(false);
            return nullptr;
        }

        const std::vector<std::shared_ptr<MemoryDevice>> SMBIOS::MemoryDevices()
        {
            assert(false);
            return std::vector<std::shared_ptr<MemoryDevice>>();
        }


        // Structure class methods

        Structure::Structure(uint8_t type, uint16_t handle, const std::vector<uint8_t>& data, const std::vector<std::string>& strings)
            : type(type), handle(handle), data(data), strings(strings) {
        }

        int Structure::GetByte(int offset) {
            if (offset >= 0 && offset < data.size()) {
                return data[offset];
            }
            return 0;
        }

        int Structure::GetWord(int offset) {
            if (offset + 1 < data.size() && offset >= 0) {
                return (data[offset + 1] << 8) | data[offset];
            }
            return 0;
        }

        std::string Structure::GetString(int offset) {
            if (offset < data.size() && data[offset] > 0 && data[offset] <= strings.size()) {
                return strings[data[offset] - 1];
            }
            return "";
        }

        const uint8_t Structure::GetType() const {
            return type;
        }

        const uint16_t Structure::GetHandle() const {
            return handle;
        }

        // BIOSInformation class methods

        BIOSInformation::BIOSInformation(const std::string& vendor, const std::string& version)
            : Structure(0x00, 0, {}, {}), vendor(vendor), version(version) {
        }

        BIOSInformation::BIOSInformation(uint8_t type, uint16_t handle, const std::vector<uint8_t>& data, const std::vector<std::string>& strings)
            : Structure(type, handle, data, strings) {
            vendor = GetString(0x04);
            version = GetString(0x05);
        }

        const std::string BIOSInformation::GetVendor() const {
            return vendor;
        }

        const std::string BIOSInformation::GetVersion() const {
            return version;
        }

        // SystemInformation class methods

        SystemInformation::SystemInformation(const std::string& manufacturerName, const std::string& productName,
            const std::string& version, const std::string& serialNumber, const std::string& family)
            : Structure(0x01, 0, {}, {}), manufacturerName(manufacturerName), productName(productName),
            version(version), serialNumber(serialNumber), family(family) {
        }

        SystemInformation::SystemInformation(uint8_t type, uint16_t handle, const std::vector<uint8_t>& data, const std::vector<std::string>& strings)
            : Structure(type, handle, data, strings) {
            manufacturerName = GetString(0x04);
            productName = GetString(0x05);
            version = GetString(0x06);
            serialNumber = GetString(0x07);
            family = GetString(0x1A);
        }

        const std::string SystemInformation::GetManufacturerName() const {
            return manufacturerName;
        }

        const std::string SystemInformation::GetProductName() const {
            return productName;
        }

        const std::string SystemInformation::GetVersion() const {
            return version;
        }

        const std::string SystemInformation::GetSerialNumber() const {
            return serialNumber;
        }

        const std::string MemoryDevice::GetPartNumber() const
        {
            return partNumber;
        }

        const int MemoryDevice::GetSpeed() const
        {
            return speed;
        }

        const std::string SystemInformation::GetFamily() const {
            return family;
        }

        // BaseBoardInformation class methods

        BaseBoardInformation::BaseBoardInformation(const std::string& manufacturerName, const std::string& productName,
            const std::string& version, const std::string& serialNumber)
            : Structure(0x02, 0, {}, {}), manufacturerName(manufacturerName), productName(productName),
            version(version), serialNumber(serialNumber) {
        }

        BaseBoardInformation::BaseBoardInformation(uint8_t type, uint16_t handle, const std::vector<uint8_t>& data, const std::vector<std::string>& strings)
            : Structure(type, handle, data, strings) {
            manufacturerName = OHM_H::TrimString(GetString(0x04));
            productName = OHM_H::TrimString(GetString(0x05));
            version = OHM_H::TrimString(GetString(0x06));
            serialNumber = OHM_H::TrimString(GetString(0x07));
        }

        const std::string BaseBoardInformation::GetManufacturerName() const {
            return manufacturerName;
        }

        const std::string BaseBoardInformation::GetProductName() const {
            return productName;
        }

        const std::string BaseBoardInformation::GetVersion() const {
            return version;
        }

        const std::string BaseBoardInformation::GetSerialNumber() const {
            return serialNumber;
        }

        // ProcessorInformation class methods

        ProcessorInformation::ProcessorInformation(uint8_t type, uint16_t handle, const std::vector<uint8_t>& data, const std::vector<std::string>& strings)
            : Structure(type, handle, data, strings) {
            manufacturerName = OHM_H::TrimString(GetString(0x07));
            version = OHM_H::TrimString(GetString(0x10));
            coreCount = GetByte(0x23);
            coreEnabled = GetByte(0x24);
            threadCount = GetByte(0x25);
            externalClock = GetWord(0x12);
        }

        const std::string ProcessorInformation::GetManufacturerName() const
        {
            return manufacturerName;
        }
        const std::string ProcessorInformation::GetVersion() const
        {
            return version;
        }
        const int ProcessorInformation::GetCoreCount() const
        {
            return coreCount;
        }
        const int ProcessorInformation::GetCoreEnabled() const
        {
            return coreEnabled;
        }
        const int ProcessorInformation::GetThreadCount() const
        {
            return threadCount;
        }
        const int ProcessorInformation::GetExternalClock() const
        {
            return externalClock;
        }

        // MemoryDevice class methods

        MemoryDevice::MemoryDevice(uint8_t type, uint16_t handle, const std::vector<uint8_t>& data, const std::vector<std::string>& strings)
            : Structure(type, handle, data, strings) {
            deviceLocator = OHM_H::TrimString(GetString(0x10));
            bankLocator = OHM_H::TrimString(GetString(0x11));
            manufacturerName = OHM_H::TrimString(GetString(0x17));
            serialNumber = OHM_H::TrimString(GetString(0x18));
            partNumber = OHM_H::TrimString(GetString(0x1A));
            speed = GetWord(0x15);
        }

        const std::string MemoryDevice::GetDeviceLocator() const {
            return deviceLocator;
        }

        const std::string MemoryDevice::GetBankLocator() const
        {
            return bankLocator;
        }


    }
}