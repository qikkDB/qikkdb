#include <stdint.h>

class DeviceChanger
{
private:
    int32_t old_device_id_;

public:
    DeviceChanger(int32_t newDevideId);

    ~DeviceChanger();
};
