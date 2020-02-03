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
    std::vector<char> buffer_;
    int32_t readOffset_;

public:
    MemoryStream();
   
    void Reset();
    
    void Clear();
    
    template <typename T>
    void Insert(T value)
    {
        char* valuePtr = reinterpret_cast<char*>(&value);
        std::copy(valuePtr, valuePtr + sizeof(T), std::back_inserter(buffer_));
    }

    template <typename T>
    T Read()
    {
        T value = *reinterpret_cast<T*>(buffer_.data() + readOffset_);
        readOffset_ += sizeof(T);
        return value;
    }
};

template <>
void MemoryStream::Insert(const std::string& value);

template <>
std::string MemoryStream::Read();
#endif // DROPDBASE_INSTAREA_MEMORYSTREAM_H
