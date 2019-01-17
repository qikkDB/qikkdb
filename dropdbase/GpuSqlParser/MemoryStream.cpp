//
// Created by Martin Staňo on 2019-01-15.
//

#include "MemoryStream.h"

//template<>
//void MemoryStream::insert(const char *value)
//{
//    int len = static_cast<int>(strlen(value));
//    insert<int>(len);
//    std::copy(value, value + len * sizeof(char), buffer.end());
//}

template<>
void MemoryStream::insert(std::string &value)
{
    int len = static_cast<int>(value.length());
    insert<int>(len);
    std::copy(value.data(), value.data() + len * sizeof(char), buffer.end());
}

template<>
std::string MemoryStream::read()
{
    int len = read<int>();
    std::string str(buffer.begin(), buffer.begin() + len);
    buffer.erase(buffer.begin(), buffer.begin() + len);
    return str;
}