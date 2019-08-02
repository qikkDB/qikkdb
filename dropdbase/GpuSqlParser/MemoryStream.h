//
// Created by Martin Staňo on 2019-01-15.
//

#ifndef DROPDBASE_INSTAREA_MEMORYSTREAM_H
#define DROPDBASE_INSTAREA_MEMORYSTREAM_H

#include <vector>
#include <string>

/// Custom byte array memory stream used to store and read arbitrary data type operands
/// of dispatcher functions
class MemoryStream
{

private:
    std::vector<char> buffer;
    int32_t readOffset;

public:
    MemoryStream()
    {
        readOffset = 0;
        buffer.reserve(8192);
    }

    void reset()
    {
        readOffset = 0;
    }

    template <typename T>
    void insert(T value)
    {
        char* valuePtr = reinterpret_cast<char*>(&value);
        std::copy(valuePtr, valuePtr + sizeof(T), std::back_inserter(buffer));
    }

    template <typename T>
    T read()
    {
        T value = *reinterpret_cast<T*>(buffer.data() + readOffset);
        readOffset += sizeof(T);
        return value;
    }
};

template <>
void MemoryStream::insert(const std::string& value);

template <>
std::string MemoryStream::read();
#endif // DROPDBASE_INSTAREA_MEMORYSTREAM_H
