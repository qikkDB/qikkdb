#include "QueryEngine/Context.h"
#include "QueryEngine/GPUCore/GPUMemory.cuh"
#include "QueryEngine/GPUCore/GPUFilterConst.cuh"
#include "QueryEngine/GPUCore/GPUReconstruct.cuh"
#include "QueryEngine/GPUCore/GPULogic.cuh"

int main(int argc, char** argv)
{
	Context::getInstance();

	const int32_t data_count = 5;
	int32_t A[data_count] = { 1, 2, 3, 4, 5 };
	int32_t B[data_count] = { 1, 0, 1, 0, 1 };

	int32_t* A_ptr;
	int32_t* B_ptr;

	GPUMemory::alloc(&A_ptr, data_count);
	GPUMemory::alloc(&B_ptr, data_count);

	GPUMemory::copyHostToDevice(A_ptr, A, data_count);
	GPUMemory::copyHostToDevice(B_ptr, B, data_count);

	int8_t *mask;
	GPUMemory::alloc(&mask, data_count);

	//GPUFilterConst::gt(mask, A_ptr, 2, data_count);

	GPULogic::and(mask, A_ptr, B_ptr, data_count);

	int32_t out_size;
	int32_t result[data_count];
	GPUReconstruct::reconstructCol(result, &out_size, A_ptr, mask, data_count);

	for (int32_t i = 0; i < data_count; i++)
		printf("%d ", A[i]);
	printf("\n");

	for (int32_t i = 0; i < data_count; i++)
		printf("%d ", B[i]);
	printf("\n");

	for (int32_t i = 0; i < out_size; i++)
		printf("%d ", result[i]);
	printf("\n");

	return 0;
}