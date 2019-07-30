#pragma once

class LoadColHelper
{
private:
    LoadColHelper() = default;

public:
    int countSkippedBlocks;
    static LoadColHelper& getInstance()
    {
        static LoadColHelper instance;
        return instance;
    }
};