#ifndef SMBIOS_H
#define SMBIOS_H

#include <stdint.h>
#include <string>
#include <vector>
#include <memory>

namespace OpenHardwareMonitor {
	namespace Hardware {

		class SMBIOS {
		public:
			SMBIOS();
			std::string GetReport() const;

			const BIOSInformation* BIOS();
			const SystemInformation* System();
			const BaseBoardInformation* Board();
			const ProcessorInformation* Processor();
			const std::vector<std::shared_ptr<MemoryDevice>> MemoryDevices();

		private:
			// this is for UNIX
			// static std::string ReadSysFS(const std::string& path);

			std::vector<byte> raw	{};
			std::vector<std::shared_ptr<Structure>> table	{};
			std::vector<std::shared_ptr<MemoryDevice>> memoryDevices {};

			std::shared_ptr<Version>				version				{ nullptr };
			std::shared_ptr<BIOSInformation>		biosInformation		{ nullptr };
			std::shared_ptr<SystemInformation>		systemInformation	{ nullptr };
			std::shared_ptr<BaseBoardInformation>	baseBoardInformation { nullptr };
			std::shared_ptr<ProcessorInformation>	processorInformation { nullptr };

			void ProcessRawSMBios();
			void GetRawData();
			void FillStructs();
		};

		class Version {
		public:
			Version(int major, int minor) : major_(major), minor_(minor) {}
			std::string ToString(int precision) const {
				std::ostringstream oss;
				oss << major_ << '.' << minor_;
				return oss.str();
			}

		private:
			int major_;
			int minor_;
		};

		class Structure {
		protected:
			uint8_t type;
			uint16_t handle;
			std::vector<uint8_t> data;
			std::vector<std::string> strings;

			int GetByte(int offset);
			int GetWord(int offset);
			std::string GetString(int offset);

		public:
			Structure(uint8_t type, uint16_t handle, const std::vector<uint8_t>& data, const std::vector<std::string>& strings);

			const uint8_t GetType() const;
			const uint16_t GetHandle() const;
		};

		class BIOSInformation : public Structure {
		private:
			std::string vendor;
			std::string version;

		public:
			BIOSInformation(const std::string& vendor, const std::string& version);
			BIOSInformation(uint8_t type, uint16_t handle, const std::vector<uint8_t>& data, const std::vector<std::string>& strings);

			const std::string GetVendor() const;
			const std::string GetVersion() const;
		};

		class SystemInformation : public Structure {
		private:
			std::string manufacturerName;
			std::string productName;
			std::string version;
			std::string serialNumber;
			std::string family;

		public:
			SystemInformation(const std::string& manufacturerName, const std::string& productName,
				const std::string& version, const std::string& serialNumber, const std::string& family);
			SystemInformation(uint8_t type, uint16_t handle, const std::vector<uint8_t>& data, const std::vector<std::string>& strings);

			const std::string GetManufacturerName() const;
			const std::string GetProductName() const;
			const std::string GetVersion() const;
			const std::string GetSerialNumber() const;
			const std::string GetFamily() const;
		};

		class BaseBoardInformation : public Structure {
		private:
			std::string manufacturerName;
			std::string productName;
			std::string version;
			std::string serialNumber;

		public:
			BaseBoardInformation(const std::string& manufacturerName, const std::string& productName,
				const std::string& version, const std::string& serialNumber);
			BaseBoardInformation(uint8_t type, uint16_t handle, const std::vector<uint8_t>& data, const std::vector<std::string>& strings);

			const std::string GetManufacturerName() const;
			const std::string GetProductName() const;
			const std::string GetVersion() const;
			const std::string GetSerialNumber() const;
		};

		class ProcessorInformation : public Structure {
		private:
			std::string manufacturerName;
			std::string version;
			int coreCount;
			int coreEnabled;
			int threadCount;
			int externalClock;
		public:
			const std::string GetManufacturerName() const;
			const std::string GetVersion() const;
			const int GetCoreCount() const;
			const int GetCoreEnabled() const;
			const int GetThreadCount() const;
			const int GetExternalClock() const;
			ProcessorInformation(uint8_t type, uint16_t handle, const std::vector<uint8_t>& data, const std::vector<std::string>& strings);
		};

		class MemoryDevice : public Structure {
		private:
			std::string deviceLocator;
			std::string bankLocator;
			std::string manufacturerName;
			std::string serialNumber;
			std::string partNumber;
			int speed;

		public:
			MemoryDevice(uint8_t type, uint16_t handle, const std::vector<uint8_t>& data, const std::vector<std::string>& strings);

			const std::string GetDeviceLocator() const;
			const std::string GetBankLocator() const;
			const std::string GetManufacturerName() const;
			const std::string GetSerialNumber() const;
			const std::string GetPartNumber() const;
			const int GetSpeed() const;
		};
	}
}

#endif
