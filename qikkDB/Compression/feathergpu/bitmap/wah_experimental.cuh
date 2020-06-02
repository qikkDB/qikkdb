#pragma once
#include "../util/ptx.cuh"

template <char CWARP_SIZE>
__device__  void wah_compress_phase_1 (unsigned long data_id, unsigned long comp_data_id, unsigned int *data, unsigned int *compressed_data, unsigned long length)
{
    unsigned int value_org, value = 0, value_neighbor;
    unsigned long pos=comp_data_id, pos_data=data_id;

    const unsigned long lane = get_lane_id();
    char neighborId = lane - 1;

    // In first shuffle it does not matter who contacts lane ==0,
    // but this will make sense in next iteration
    if (lane == 0 )  {
        neighborId = 31;
    }

    // TODO: fix for cases where data is not a multiplicyty of warp_size
    // then in last warp not all threads in warp may be executed
    // and as a result a partialy infalted data may appear
    for (unsigned int i = 0; i < CWORD_SIZE(unsigned int) && pos_data < length; ++i)
    {
        value_org = 0;
        // Last warp waits
        if(lane != 31)
            value_org = data[pos_data];

        //pos_data move is warp_size - 1
        pos_data += CWARP_SIZE - 1;

        // Each thread gets appropriate number of bits and shift by 'lane' bits.
        // lane == 31 reads 0 bits, this is intentional.
        value = GETNBITS(value_org, 31 - lane) << lane;

        // Get neighbour's value
        value_neighbor = shfl_get_value(value_org, neighborId);

        // get remaining bits (lane==0 takes 0 bits this is intentional)
        value |= GETNPBITS(value_neighbor, lane, 31 - neighborId);

        // check type of word and set correctly the word type bit
        // if word is 0 or 1 set correct
       if (value == 127) value = 193;
       if (value == 0) value = 192;

        // TODO: This is just for test to check if we correctly inflate the data
        // flush compression buffer
        compressed_data[pos] = value;

        // We inflate 31 words to 32 words so we move by CWARP_SIZE
        pos += CWARP_SIZE;
    }
}

template <char CWARP_SIZE >
__global__ void wah_compress_kernel ( unsigned int *data, unsigned int *compressed_data, unsigned long length)
{
    //TODO: Check if data id is correct
    const unsigned int bit_length = 32;
    const unsigned int warp_lane = threadIdx.x % CWARP_SIZE;
    const unsigned long data_block = blockIdx.x * blockDim.x + threadIdx.x - warp_lane;
    const unsigned long data_id = (data_block * CWORD_SIZE(unsigned int) - 1) + warp_lane;
    const unsigned long cdata_id = data_block * bit_length + warp_lane;

    wah_compress_phase_1 <CWARP_SIZE> (data_id, cdata_id, data, compressed_data, length);
}

template < char CWARP_SIZE >
__host__ void run_wah_compress( unsigned int *data, unsigned int *compressed_data, const unsigned long length)
{
    const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
    const unsigned long block_number = (length + block_size * CWORD_SIZE(unsigned int) - 1) / (block_size * CWORD_SIZE(unsigned int));
    wah_compress_kernel <CWARP_SIZE> <<<block_number, block_size>>> (data, compressed_data, length);
}
