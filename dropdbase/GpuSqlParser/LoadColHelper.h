#pragma once

#include <atomic>

class LoadColHelper
{
private:
    LoadColHelper() = default;

public:
    std::atomic_int32_t countSkippedBlocks;
    static LoadColHelper& getInstance()
    {
        static LoadColHelper instance;
        return instance;
    }
};