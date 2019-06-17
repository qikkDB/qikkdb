#include "Configuration.h"


/// <summary>
/// Loads configuration YAML file, parses it and sets configuration values
/// </summary>
void Configuration::LoadConfigurationFile()
{
	// loading and parsing file
	try {
		yamlParsed_ = YAML::LoadFile(this->configurationFile); 
	}
	catch (YAML::ParserException&) {
		//BOOST_LOG_TRIVIAL(info) << "Configuration file could not be parsed. Using default values." << std::endl;
		std::cerr << "Configuration file could not be parsed. Using default values." << std::endl;
	}
	catch (YAML::BadFile&) {
		//BOOST_LOG_TRIVIAL(info) << "Configuration file could not found. Using default values." << std::endl;
		std::cerr << "Configuration file could not found. Using default values." << std::endl;
	}

	// setting particular YAML entries into configuration values
	this->SetupConfigurationValue("UsingGPU", this->usingGPU_);
	this->SetupConfigurationValue("Dir", this->dir_);
	this->SetupConfigurationValue("DatabaseDir", this->databaseDir_);
	this->SetupConfigurationValue("BlockSize", this->blockSize_);
	this->SetupConfigurationValue("BlockCount", this->blockCount_);
	this->SetupConfigurationValue("GroupByBuckets", this->groupByBuckets_);
	this->SetupConfigurationValue("ListenIP", this->listenIP_);
	this->SetupConfigurationValue("ListenPort", this->listenPort_);
	this->SetupConfigurationValue("Timeout", this->timeout_);
	this->SetupConfigurationValue("GPUCachePercent", this->GPUCachePercent_);
}
