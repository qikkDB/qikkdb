#pragma once

#include <string>
#include <utility>
#include <boost/algorithm/string.hpp>

namespace StringUnaryOperationsCpu
{
struct ltrim
{
    static constexpr bool isMonotonous = false;
    __host__ std::string operator()(const char* str, int32_t len) const
    {
        std::string strObj(str, len);
        boost::trim_left(strObj);
        return strObj;
    }
};

struct rtrim
{
    static constexpr bool isMonotonous = false;
    __host__ std::string operator()(const char* str, int32_t len) const
    {
        std::string strObj(str, len);
        boost::trim_right(strObj);
        return strObj;
    }
};

struct lower
{
    static constexpr bool isMonotonous = false;
    __host__ std::string operator()(const char* str, int32_t len) const
    {
        std::string strObj(str, len);
        boost::to_lower(strObj);
        return strObj;
    }
};

struct upper
{
    static constexpr bool isMonotonous = false;
    __host__ std::string operator()(const char* str, int32_t len) const
    {
        std::string strObj(str, len);
        boost::to_upper(strObj);
        return strObj;
    }
};

struct reverse
{
    static constexpr bool isMonotonous = false;
    __host__ std::string operator()(const char* str, int32_t len) const
    {
        std::string strObj(str, len);
        std::reverse(strObj.begin(), strObj.end());
        return strObj;
    }
};

struct len
{
    static constexpr bool isMonotonous = false;
    __host__ int32_t operator()(const char* str, int32_t len) const
    {
        return len;
    }
};
} // namespace StringUnaryOperationsCpu

namespace StringBinaryOperationsCpu
{
struct left
{
    static constexpr bool isMonotonous = true;
    template <typename T>
    __host__ std::string operator()(const char* str, int32_t len, T arg) const
    {
        std::string strObj(str, len);
        return strObj.substr(0, arg);
    }
};

struct right
{
    static constexpr bool isMonotonous = false;
    template <typename T>
    __host__ std::string operator()(const char* str, int32_t len, T arg) const
    {
        std::string strObj(str, len);
        return strObj.substr(len - arg);
    }
};

struct concat
{
    static constexpr bool isMonotonous = false;
    __host__ std::string operator()(const char* strLeft, int32_t lenLeft, const char* strRight, int32_t lenRight) const
    {
        std::string strObjLeft(strLeft, lenLeft);
        std::string strObjRight(strRight, lenRight);
        return strObjLeft + strObjRight;
    }
};

} // namespace StringBinaryOperationsCpu