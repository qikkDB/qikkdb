#pragma once

#include <yaml-cpp/yaml.h>
#include <iostream>
#include "CudaLogBoost.h"

class Configuration
{
public:
    static Configuration& GetInstance()
    {
        static Configuration instance;
        return instance;
    }

private:
    Configuration()
    {
        this->LoadConfigurationFile();
    }

    // parsed YAML file
    YAML::Node yamlParsed_;

    // Config and default config file
    std::string configurationFile = "../configuration/main_config";
    std::string configurationFileDefault = configurationFile + ".default";

    // Configuration values (if even default config does not exist)
    bool usingGPU_ = true;
    bool usingMultipleGPUs_ = true;
    bool usingWhereEvaluationSpeedup_ = false;
    std::string databaseDir_ = "../databases/";
    std::string testDatabaseDir_ = "../test_databases/";
    int32_t blockSize_ = 262144;
    int32_t groupByBuckets_ = 262144;
    std::string listenIP_ = "127.0.0.1";
    short listenPort_ = 12345;
    int32_t timeout_ = 3600000;
    int32_t GPUMemoryUsagePercent_ = 100;
    int32_t GPUCachePercent_ = 75;
    int32_t DBSaveInterval_ = 3600000;

    void LoadConfigurationFile();

    /// <summary>
    /// Sets the configuration value of type T by entry key in YAML file
    /// </summary>
    /// <param name="entryKey">Entry key in YAML file</param>
    /// <param name="value">Configuration value which is set</param>
    template <class T>
    void SetupConfigurationValue(const char* entryKey, T& configurationValue)
    {
        if (yamlParsed_[entryKey])
        {
            try
            {
                configurationValue = yamlParsed_[entryKey].as<T>();
                // BOOST_LOG_TRIVIAL(info) << "Configuration entry loaded. " << entryKey << ": " << configurationValue << std::endl;
                CudaLogBoost::getInstance(CudaLogBoost::info)
                    << "Configuration entry loaded. " << entryKey << ": " << configurationValue << '\n';
            }
            catch (YAML::TypedBadConversion<T>&)
            {
                // BOOST_LOG_TRIVIAL(warning) << "Configuration entry wrong conversion, using default value." << std::endl;
                CudaLogBoost::getInstance(CudaLogBoost::warning)
                    << "Configuration entry (" << entryKey
                    << ") has a wrong conversion, using default value." << '\n';
            }
        }
        else
        {
            // BOOST_LOG_TRIVIAL(warning) << "Configuration entry not found, using default value." << std::endl;
            CudaLogBoost::getInstance(CudaLogBoost::warning)
                << "Configuration entry (" << entryKey << ") not found, using default value." << '\n';
        }
    }

public:
    Configuration(Configuration const&) = delete;
    void operator=(Configuration const&) = delete;

    bool IsUsingGPU() const
    {
        return usingGPU_;
    }

    bool IsUsingMultipleGPUs() const
    {
        return usingMultipleGPUs_;
    }

    bool IsUsingWhereEvaluationSpeedup() const
    {
        return usingWhereEvaluationSpeedup_;
    }

    const std::string& GetDatabaseDir() const
    {
        return databaseDir_;
    }

	const std::string& GetTestDatabaseDir() const
    {
        return testDatabaseDir_;
    }

    int32_t GetBlockSize() const
    {
        return blockSize_;
    }

    int32_t GetGroupByBuckets() const
    {
        return groupByBuckets_;
    }

    const std::string& GetListenIP()
    {
        return listenIP_;
    }

    short GetListenPort() const
    {
        return listenPort_;
    }

    int32_t GetTimeout() const
    {
        return timeout_;
    }

    int32_t GetGPUMemoryUsagePercent() const
    {
        return GPUMemoryUsagePercent_;
    }


    int32_t GetGPUCachePercentage() const
    {
        return GPUCachePercent_;
    }

    int32_t GetDBSaveInterval() const
    {
        return DBSaveInterval_;
    }
};
