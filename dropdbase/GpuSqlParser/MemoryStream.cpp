//
// Created by Martin Staňo on 2019-01-15.
//

#include "MemoryStream.h"

// template<>
// void MemoryStream::insert(const char *value)
//{
//    int len = static_cast<int>(strlen(value));
//    insert<int>(len);
//    std::copy(value, value + len * sizeof(char), buffer.end());
//}

template <>
void MemoryStream::insert(const std::string& value)
{
    int len = static_cast<int>(value.length());
    insert<int32_t>(len);
    std::copy(value.begin(), value.end(), std::back_inserter(buffer));
}

template <>
std::string MemoryStream::read()
{
    int32_t len = read<int32_t>();
    std::string str(buffer.begin() + readOffset, buffer.begin() + readOffset + len);
    readOffset += len;
    return str;
}