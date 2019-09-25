#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>

#include "../Context.h"
#include "../GPUError.h"
#include "MaybeDeref.cuh"
#include "DateOperations.h"
#include "cuda_ptr.h"
#include "GPUReconstruct.cuh"
#include "GPUMemory.cuh"

/// Kernel for extracting date or time variable (e.g. days, hours)
/// from datetime column or constant
/// <param name="output">block of the result data</param>
/// <param name="dateTimeCol">input timestamp (column or constant)</param>
/// <param name="dataElementCount">the count of elements in the input block
/// (or of output block if input is constant)</param>
template <typename OP, typename T>
__global__ void kernel_extract(int32_t* output, T dateTimeCol, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        output[i] = OP{}.operator()(maybe_deref(dateTimeCol, i));
    }
}


__global__ void kernel_fill_date_string(GPUMemory::GPUString outCol,
                                        int32_t* years,
                                        int32_t* months,
                                        int32_t* days,
                                        int32_t* hours,
                                        int32_t* minutes,
                                        int32_t* seconds,
                                        int32_t dataElementCount);

/// GPUDate class is for extracting (conversion) variables (e.g. days, hours)
/// from datetime column or constant
class GPUDate
{
public:
    /// Extract values (one value - year/month/dat/hour/minute/second - per row) from a datetime column
    /// <param name="output">Output GPU buffer for result (int32_t)</param>
    /// <param name="dateTimeCol">Input GPU buffer - datetime (int64_t)</param>
    /// <param name="dataElementCount">Row count (e.g. size of block to process)</param>
    template <typename OP, typename T>
    static void Extract(int32_t* output, T dateTimeCol, int32_t dataElementCount)
    {
        static_assert(std::is_same<typename std::remove_pointer<T>::type, int64_t>::value,
                      "DateTime can only be extracted from int64 columns");
        kernel_extract<OP>
            <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                output, dateTimeCol, dataElementCount);
        cudaDeviceSynchronize();
        CheckCudaError(cudaGetLastError());
    }

    template <typename OP, typename T>
    static void DateToString(GPUMemory::GPUString* output, T dateTimeCol, int32_t dataElementCount)
    {
        static_assert(std::is_same<typename std::remove_pointer<T>::type, int64_t>::value,
                      "DateTime can only be extracted from int64 columns");

        if (dataElementCount > 0)
        {

            // Length of date format "YYYY-MM-DD HH:MM:SS" = 19

            const int32_t dateFormatLength = 19;

            GPUMemory::alloc(&(output->stringIndices), dataElementCount);

            cuda_ptr<int32_t> lengths(dataElementCount);
            kernel_fill_array<<<Context::getInstance().calcGridDim(dataElementCount),
                                Context::getInstance().getBlockDim()>>>(lengths.get(), dateFormatLength,
                                                                        dataElementCount);

            GPUReconstruct::PrefixSum(output->stringIndices, lengths.get(), dataElementCount);

            int64_t totalCharCount;
            GPUMemory::copyDeviceToHost(&totalCharCount, output->stringIndices + dataElementCount - 1, 1);
            GPUMemory::alloc(&(output->allChars), totalCharCount);

            cuda_ptr<int32_t> years(dataElementCount);
            cuda_ptr<int32_t> months(dataElementCount);
            cuda_ptr<int32_t> days(dataElementCount);
            cuda_ptr<int32_t> hours(dataElementCount);
            cuda_ptr<int32_t> minutes(dataElementCount);
            cuda_ptr<int32_t> seconds(dataElementCount);


            kernel_extract<DateOperations::year>
                <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                    years.get(), dateTimeCol, dataElementCount);

            kernel_extract<DateOperations::month>
                <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                    months.get(), dateTimeCol, dataElementCount);

            kernel_extract<DateOperations::day>
                <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                    days.get(), dateTimeCol, dataElementCount);

            kernel_extract<DateOperations::hour>
                <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                    hours.get(), dateTimeCol, dataElementCount);

            kernel_extract<DateOperations::minute>
                <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                    minutes.get(), dateTimeCol, dataElementCount);

            kernel_extract<DateOperations::second>
                <<<Context::getInstance().calcGridDim(dataElementCount), Context::getInstance().getBlockDim()>>>(
                    seconds.get(), dateTimeCol, dataElementCount);

            kernel_fill_date_string<<<Context::getInstance().calcGridDim(dataElementCount),
                                      Context::getInstance().getBlockDim()>>>(*output, years.get(),
                                                                              months.get(), days.get(),
                                                                              hours.get(), minutes.get(),
                                                                              seconds.get(), dataElementCount);
            cudaDeviceSynchronize();
            CheckCudaError(cudaGetLastError());
        }
        else
        {
            output->allChars = nullptr;
            output->allChars = nullptr;
        }
    }
};
