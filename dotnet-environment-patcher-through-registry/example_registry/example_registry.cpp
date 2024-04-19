// example_registry.cpp : main project file.

#include "stdafx.h"

using namespace System;
using namespace Microsoft::Win32;

int main(array<System::String ^> ^args)
{
	//for (int i=0; i< args->Length; ++i)
	//	Console::WriteLine(""+i+" "+ args[i]);
	int count = args->Length;
	if (count <2)
	{
		Console::WriteLine("Run this program with arguments: [enviroment_patcher Value Key] to set them in registry");
	}
	else if (count == 2)
	{


		RegistryKey^ rk;
		rk  = Registry::LocalMachine->OpenSubKey("SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment", true);
		if (!rk)
		{
			Console::WriteLine("Failed to open CurrentUser/Software key");
			return -1;
		}
		/*
		RegistryKey^ nk = rk->CreateSubKey("NewRegKey");
		if (!nk)
		{
		Console::WriteLine("Failed to create 'NewRegKey'");
		return -1;
		}
		*/
		String^ keyName = args[0];
		String^ keyValue = args[1];
		try
		{
			rk->SetValue(keyName, keyValue);
		}
		catch (Exception^)
		{
			Console::WriteLine("Failed to set new values in 'NewRegKey'");
			return -1;
		}

		Console::WriteLine(keyName +" set with property " + keyValue);
	}
	else
	{
		Console::WriteLine("Wrong arguments count, should be zero or two");
		return -1;
	}
	return 0;
}
