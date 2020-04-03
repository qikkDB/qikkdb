#pragma once

#include <stdint.h>
#include <unordered_set>


class Allocator
{
private:
    int32_t device_id_;
    std::unordered_set<int8_t*> allocated_pointers_;

public:
    explicit Allocator(int32_t deviceId);

    int8_t* Allocate(size_t numBytes);

    void Deallocate(int8_t* ptr);

    void Clear();
};
