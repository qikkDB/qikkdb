#include "Configuration.h"
#include "boost/filesystem.hpp"

/// <summary>
/// Loads configuration YAML file, parses it and sets configuration values
/// </summary>
void Configuration::LoadConfigurationFile()
{
    // loading and parsing file
    try
    {
        if (boost::filesystem::exists(this->configurationFile)) // If user specific config exsits, load that
        {
            yamlParsed_ = YAML::LoadFile(this->configurationFile);
            // BOOST_LOG_TRIVIAL(info) << "Loaded specific configuration." << '\n';
            CudaLogBoost::getInstance(CudaLogBoost::info) << "Loaded specific configuration." << '\n';
        }
        else // Otherwise load default config file
        {
            yamlParsed_ = YAML::LoadFile(this->configurationFileDefault);
            // BOOST_LOG_TRIVIAL(info) << "Loaded default configuration." << '\n';
            CudaLogBoost::getInstance(CudaLogBoost::info) << "Loaded default configuration." << '\n';
        }
    }
    catch (YAML::ParserException&)
    {
        // BOOST_LOG_TRIVIAL(warning) << "Configuration file could not be parsed. Using default values." << '\n';
        CudaLogBoost::getInstance(CudaLogBoost::warning)
            << "Configuration file could not be parsed. Using default values." << '\n';
    }
    catch (YAML::BadFile&)
    {
        // BOOST_LOG_TRIVIAL(warning) << "Configuration file could not found. Using default values." << '\n';
        CudaLogBoost::getInstance(CudaLogBoost::warning)
            << "Configuration file could not found. Using default values." << '\n';
    }

    // setting particular YAML entries into configuration values
    this->SetupConfigurationValue("UsingGPU", this->usingGPU_);
    this->SetupConfigurationValue("UsingCompression", this->usingCompression_);
    this->SetupConfigurationValue("Dir", this->dir_);
    this->SetupConfigurationValue("DatabaseDir", this->databaseDir_);
    this->SetupConfigurationValue("BlockSize", this->blockSize_);
    this->SetupConfigurationValue("BlockCount", this->blockCount_);
    this->SetupConfigurationValue("GroupByBuckets", this->groupByBuckets_);
    this->SetupConfigurationValue("ListenIP", this->listenIP_);
    this->SetupConfigurationValue("ListenPort", this->listenPort_);
    this->SetupConfigurationValue("Timeout", this->timeout_);
    this->SetupConfigurationValue("GPUCachePercent", this->GPUCachePercent_);
    this->SetupConfigurationValue("DBSaveInterval", this->DBSaveInterval_);

    // Check group by buckets and warn user if not optimal
    if ((groupByBuckets_ & (groupByBuckets_ - 1)) != 0)
    {
        CudaLogBoost::getInstance(CudaLogBoost::warning)
            << "GroupByBuckets is " << groupByBuckets_
            << ". For more speed change it to some power of 2 (e.g. 32768) in config." << '\n';
    }
}
