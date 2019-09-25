#include "GPUDate.cuh"
#include "GPUCast.cuh"

__global__ void kernel_fill_date_string(GPUMemory::GPUString outCol,
                                        int32_t* years,
                                        int32_t* months,
                                        int32_t* days,
                                        int32_t* hours,
                                        int32_t* minutes,
                                        int32_t* seconds,
                                        int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        int64_t stringIndex = GetStringIndex(outCol.stringIndices, i);

        NumericToString(outCol.allChars, stringIndex, maybe_deref(years, i), 4);
		outCol.allChars[stringIndex++] = '-';

        NumericToString(outCol.allChars, stringIndex, maybe_deref(months, i), 2);
        outCol.allChars[stringIndex++] = '-';

        NumericToString(outCol.allChars, stringIndex, maybe_deref(days, i), 2);
        outCol.allChars[stringIndex++] = ' ';

		NumericToString(outCol.allChars, stringIndex, maybe_deref(hours, i), 2);
        outCol.allChars[stringIndex++] = ':';

		NumericToString(outCol.allChars, stringIndex, maybe_deref(minutes, i), 2);
        outCol.allChars[stringIndex++] = ':';

		NumericToString(outCol.allChars, stringIndex, maybe_deref(seconds, i), 2);

    }
}