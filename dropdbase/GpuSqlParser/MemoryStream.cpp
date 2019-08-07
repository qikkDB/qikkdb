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
void MemoryStream::Insert(const std::string& value)
{
    int len = static_cast<int>(value.length());
    Insert<int32_t>(len);
    std::copy(value.begin(), value.end(), std::back_inserter(buffer_));
}

template <>
std::string MemoryStream::Read()
{
    int32_t len = Read<int32_t>();
    std::string str(buffer_.begin() + readOffset_, buffer_.begin() + readOffset_ + len);
    readOffset_ += len;
    return str;
}