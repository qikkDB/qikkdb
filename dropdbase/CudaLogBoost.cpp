#include "CudaLogBoost.h"
#include <boost/log/trivial.hpp>

CudaLogBoost& CudaLogBoost::operator<<(const std::string& text)
{
    std::unique_lock<std::mutex> lock(logMutex_);
    if (text == "\n")
    {
        std::string str = buffer_.str();
        buffer_.str(std::string());
        switch (severity)
        {
        case trace:
            BOOST_LOG_TRIVIAL(trace) << str;
            break;
        case debug:
            BOOST_LOG_TRIVIAL(debug) << str;
            break;
        case info:
            BOOST_LOG_TRIVIAL(info) << str;
            break;
        case warning:
            BOOST_LOG_TRIVIAL(warning) << str;
            break;
        case error:
            BOOST_LOG_TRIVIAL(error) << str;
            break;
        case fatal:
            BOOST_LOG_TRIVIAL(fatal) << str;
            break;
        default:
            BOOST_LOG_TRIVIAL(info) << str;
            break;
        }
    }
    else
    {
        buffer_ << text;
    }

    return *this;
}

CudaLogBoost& CudaLogBoost::operator<<(int64_t value)
{
    std::unique_lock<std::mutex> lock(logMutex_);
    buffer_ << value;
    return *this;
}

CudaLogBoost& CudaLogBoost::operator<<(size_t value)
{
    std::unique_lock<std::mutex> lock(logMutex_);
    buffer_ << value;
    return *this;
}


CudaLogBoost& CudaLogBoost::operator<<(int32_t value)
{
    std::unique_lock<std::mutex> lock(logMutex_);
    buffer_ << value;
    return *this;
}

CudaLogBoost& CudaLogBoost::operator<<(short value)
{
    std::unique_lock<std::mutex> lock(logMutex_);
    buffer_ << value;
    return *this;
}

CudaLogBoost& CudaLogBoost::operator<<(char value)
{
    std::unique_lock<std::mutex> lock(logMutex_);
    if (value == '\n')
    {
        std::string str = buffer_.str();
        buffer_.str(std::string());
        switch (severity)
        {
        case trace:
            BOOST_LOG_TRIVIAL(trace) << str;
            break;
        case debug:
            BOOST_LOG_TRIVIAL(debug) << str;
            break;
        case info:
            BOOST_LOG_TRIVIAL(info) << str;
            break;
        case warning:
            BOOST_LOG_TRIVIAL(warning) << str;
            break;
        case error:
            BOOST_LOG_TRIVIAL(error) << str;
            break;
        case fatal:
            BOOST_LOG_TRIVIAL(fatal) << str;
            break;
        default:
            BOOST_LOG_TRIVIAL(info) << str;
            break;
        }
    }
    else
    {
        buffer_ << value;
    }
    return *this;
}

CudaLogBoost& CudaLogBoost::operator<<(double value)
{
    std::unique_lock<std::mutex> lock(logMutex_);
    buffer_ << value;
    return *this;
}

CudaLogBoost& CudaLogBoost::operator<<(float value)
{
    std::unique_lock<std::mutex> lock(logMutex_);
    buffer_ << value;
    return *this;
}

CudaLogBoost& CudaLogBoost::operator<<(void* value)
{
    std::unique_lock<std::mutex> lock(logMutex_);
    buffer_ << value;
    return *this;
}

CudaLogBoost& CudaLogBoost::operator<<(const char* value)
{
    std::unique_lock<std::mutex> lock(logMutex_);
    if (strcmp(value, "\n") == 0)
    {
        std::string str = buffer_.str();
        buffer_.str(std::string());
        switch (severity)
        {
        case trace:
            BOOST_LOG_TRIVIAL(trace) << str;
            break;
        case debug:
            BOOST_LOG_TRIVIAL(debug) << str;
            break;
        case info:
            BOOST_LOG_TRIVIAL(info) << str;
            break;
        case warning:
            BOOST_LOG_TRIVIAL(warning) << str;
            break;
        case error:
            BOOST_LOG_TRIVIAL(error) << str;
            break;
        case fatal:
            BOOST_LOG_TRIVIAL(fatal) << str;
            break;
        default:
            BOOST_LOG_TRIVIAL(info) << str;
            break;
        }
    }
    else
    {
        buffer_ << value;
    }
    return *this;
}

CudaLogBoost::CudaLogBoost(Severity severity) : severity(severity), buffer_()
{
}
