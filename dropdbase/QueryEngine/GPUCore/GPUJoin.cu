#include "GPUJoin.cuh"

GPUJoin::GPUJoin(int32_t hashTableSize) :
	hashTableSize_(hashTableSize),
	joinTableSize_(hashTableSize),
	hash_prefix_sum_temp_buffer_(nullptr),
	hash_prefix_sum_temp_buffer_size_(0),
	join_prefix_sum_temp_buffer_(nullptr),
	join_prefix_sum_temp_buffer_size_(0)
{
	GPUMemory::alloc(&HashTableHisto_, hashTableSize_);
	GPUMemory::alloc(&HashTablePrefixSum_, hashTableSize_);
	GPUMemory::alloc(&HashTableHashBuckets_, hashTableSize_);

	GPUMemory::allocAndSet(&JoinTableHisto_, 0, joinTableSize_);
	GPUMemory::alloc(&JoinTablePrefixSum_, joinTableSize_);

	cub::DeviceScan::InclusiveSum(hash_prefix_sum_temp_buffer_, hash_prefix_sum_temp_buffer_size_, HashTableHisto_, HashTablePrefixSum_, hashTableSize_);
	GPUMemory::alloc<int8_t>(reinterpret_cast<int8_t**>(&hash_prefix_sum_temp_buffer_), hash_prefix_sum_temp_buffer_size_);

	cub::DeviceScan::InclusiveSum(join_prefix_sum_temp_buffer_, join_prefix_sum_temp_buffer_size_, JoinTableHisto_, JoinTablePrefixSum_, joinTableSize_);
	GPUMemory::alloc<int8_t>(reinterpret_cast<int8_t**>(&join_prefix_sum_temp_buffer_), join_prefix_sum_temp_buffer_size_);
}

GPUJoin::~GPUJoin()
{
	GPUMemory::free(HashTableHisto_);
	GPUMemory::free(HashTablePrefixSum_);
	GPUMemory::free(HashTableHashBuckets_);

	GPUMemory::free(JoinTableHisto_);
	GPUMemory::free(JoinTablePrefixSum_);

	GPUMemory::free(hash_prefix_sum_temp_buffer_);
	GPUMemory::free(join_prefix_sum_temp_buffer_);
}