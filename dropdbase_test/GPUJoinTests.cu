#include "../dropdbase/QueryEngine/GPUCore/GPUJoin.cuh"
#include "../dropdbase/QueryEngine/GPUCore/GPUMemory.cuh"
#include "gtest/gtest.h"

TEST(GPUJoinTests, JoinTest)
{
    const int32_t blockSize = 8;
    const int32_t hashTableSize = blockSize;

    const int32_t RTableDataElementCount = 16;
    const int32_t STableDataElementCount = 8;
	const int32_t QTableDataElementCount = blockSize;

    int32_t RTable[RTableDataElementCount] = {5, 1, 0, 2, 0, 2, 1, 7, 0, 2, 5, 0, 1, 1, 7, 0};
    int32_t STable[STableDataElementCount] = {5, 1, 7, 0, 2, 3, 4, 6};

	int32_t QATable[QTableDataElementCount];
    int32_t QBTable[QTableDataElementCount];

    int32_t* d_RTable;
    int32_t* d_STable;
    int32_t* d_QATable;
    int32_t* d_QBTable;

    GPUMemory::alloc(&d_RTable, RTableDataElementCount);
    GPUMemory::alloc(&d_STable, STableDataElementCount);
    GPUMemory::alloc(&d_QATable, QTableDataElementCount);
    GPUMemory::alloc(&d_QBTable, QTableDataElementCount);

	GPUMemory::copyHostToDevice(d_RTable, RTable, RTableDataElementCount);
	GPUMemory::copyHostToDevice(d_STable, STable, STableDataElementCount);

    // Create a join instance
    GPUJoin gpuJoin(hashTableSize);

	// Hash the values and then join them
	for (int32_t i = 0; i < RTableDataElementCount; i += blockSize)
	{
        gpuJoin.HashBlock(&d_RTable[i], blockSize);
		for (int32_t j = 0; j < STableDataElementCount; j += blockSize)
		{
			gpuJoin.JoinBlock(d_QATable, d_QBTable, nullptr, &d_STable[j], blockSize);

			// DEBUG - Copy the blocks back and print their content
			GPUMemory::copyDeviceToHost(QATable, d_QATable, QTableDataElementCount);
			GPUMemory::copyDeviceToHost(QBTable, d_QBTable, QTableDataElementCount);

			std::printf("#### RESULT INFO ###\n");
			for (int32_t debug_idx = 0; debug_idx < QTableDataElementCount; debug_idx++)
			{
				std::printf("%d %d\n", QATable[debug_idx], QBTable[debug_idx]);
			}
		}

		// DEBUG - Info about inner buffers
		gpuJoin.printDebugInfo();
	}

    GPUMemory::free(d_RTable);
    GPUMemory::free(d_STable);
    GPUMemory::free(d_QATable);
    GPUMemory::free(d_QBTable);

    ASSERT_EQ(false, true);
}
