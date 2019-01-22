#pragma once
#include <yaml-cpp\yaml.h>
#include <spdlog_setup/conf.h>

class Configuration
{
public:
	static Configuration& GetInstance()
	{
		static Configuration instance;
		return instance;
	}

private:
	Configuration() {
		log_ = spdlog::get("root");
		this->LoadConfigurationFile();
	}
	
	// logger
	std::shared_ptr<spdlog::logger> log_;
	
	// parsed YAML file
	YAML::Node yamlParsed_;

	// configuration values
	std::string configurationFile = "../configuration/config.yml";
	bool usingGPU_ = true;
	std::string dir_ = "./";
	std::string databaseDir_ = "./";
	int blockSize_ = 1024;
	int blockCount_ = 1024;
	int groupByBuckets_ = 65536;
	std::string listenIP_ = "127.0.0.1";
	short listenPort_ = 12345;
	int timeout_ = 5000;

	void LoadConfigurationFile();

	/// <summary>
	/// Sets the configuration value of type T by entry key in YAML file
	/// </summary>
	/// <param name="entryKey">Entry key in YAML file</param>
	/// <param name="value">Configuration value which is set</param>
	template <class T>
	void SetupConfigurationValue(const char* entryKey, T& configurationValue)
	{
		if (yamlParsed_[entryKey]) {
			try {
				configurationValue = yamlParsed_[entryKey].as<T>();
				log_->info("Configuration entry loaded. {}: {}", entryKey, configurationValue);
			}
			catch (YAML::TypedBadConversion<T>& e) {
				log_->warn("Configuration entry wrong conversion, using default value.");
			}			
		}
		else {
			log_->warn("Configuration entry not found, using default value.");
		}
	}

public:
	Configuration(Configuration const&) = delete;
	void operator=(Configuration const&) = delete;
	
	const bool IsUsingGPU() {
		return usingGPU_;
	}

	const std::string & GetDir() {
		return dir_;
	}

	const std::string & GetDatabaseDir() {
		return databaseDir_;
	}

	const int GetBlockSize() {
		return blockSize_;
	}

	const int GetBlockCount() {
		return blockCount_;
	}

	const int GetGroupByBuckets() {
		return groupByBuckets_;
	}

	const std::string & GetListenIP() {
		return listenIP_;
	}

	const short GetListenPort() {
		return listenPort_;
	}

	const int GetTimeout() {
		return timeout_;
	}
};