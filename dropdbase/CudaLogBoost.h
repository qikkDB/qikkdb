#pragma once
#include <string>
#include <sstream>

class CudaLogBoost
{
public:
    enum Severity
    {
        trace,
        debug,
        info,
        warning,
        error,
        fatal,
    };

    static CudaLogBoost& getInstance(Severity severity)
    {
        static CudaLogBoost instanceTrace(Severity::trace);
        static CudaLogBoost instanceDebug(Severity::debug);
        static CudaLogBoost instanceInfo(Severity::info);
        static CudaLogBoost instanceWarning(Severity::warning);
        static CudaLogBoost instanceError(Severity::error);
        static CudaLogBoost instanceFatal(Severity::fatal);

        switch (severity)
        {
        case trace:
            return instanceTrace;
            break;
        case debug:
            return instanceDebug;
            break;
        case info:
            return instanceInfo;
            break;
        case warning:
            return instanceWarning;
            break;
        case error:
            return instanceError;
            break;
        case fatal:
            return instanceFatal;
            break;
        default:
            return instanceInfo;
            break;
        }
    }

    CudaLogBoost(CudaLogBoost const&) = delete;
    void operator=(CudaLogBoost const&) = delete;

    CudaLogBoost& operator<<(const std::string& text);
    CudaLogBoost& operator<<(int64_t value);
    CudaLogBoost& operator<<(size_t value);
    CudaLogBoost& operator<<(int32_t value);
    CudaLogBoost& operator<<(short value);
    CudaLogBoost& operator<<(char value);
    CudaLogBoost& operator<<(double value);
    CudaLogBoost& operator<<(float value);

private:
    Severity severity;
    std::stringstream buffer_;
    CudaLogBoost(Severity severity);
};