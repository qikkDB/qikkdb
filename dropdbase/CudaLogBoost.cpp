#include "CudaLogBoost.h"
#include <boost/log/trivial.hpp>

CudaLogBoost& CudaLogBoost::operator<<(const std::string& text)
{
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
    buffer_ << value;
    return *this;
}

CudaLogBoost& CudaLogBoost::operator<<(size_t value)
{
    buffer_ << value;
    return *this;
}


CudaLogBoost& CudaLogBoost::operator<<(int32_t value)
{
    buffer_ << value;
    return *this;
}

CudaLogBoost& CudaLogBoost::operator<<(short value)
{
    buffer_ << value;
    return *this;
}

CudaLogBoost& CudaLogBoost::operator<<(char value)
{
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
    buffer_ << value;
    return *this;
}

CudaLogBoost& CudaLogBoost::operator<<(float value)
{
    buffer_ << value;
    return *this;
}

CudaLogBoost::CudaLogBoost(Severity severity) : severity(severity)
{
}
