#include "GPUReconstruct.cuh"
#include "cuda_ptr.h"

// Polygon WKT format:
// POLYGON((179.9999 89.9999, 0.0000 0.0000, 179.9999 89.9999), (-179.9999 -89.9999, 52.1300 -27.0380, -179.9999 -89.9999))

/// Kernel for reconstructing null masks according to calculated prefixSum and inMask
__global__ void
kernel_reconstruct_null_mask(nullmask_t* outNullMask, nullmask_t* inNullMask, int32_t* prefixSum, int8_t* filterMask, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        // Select the elemnts that are "visible" in the mask
        // If the mask is 1 for the output, use the prefix sum for array compaction
        // The prefix sum includes values from the input array on the same element so the index has to be modified
        if (filterMask[i] && (prefixSum[i] - 1) >= 0)
        {
            int outBitMaskIdx = NullValues::GetBitMaskIdx(prefixSum[i] - 1);
            int outBitMaskShiftIdx = NullValues::GetShiftMaskIdx(prefixSum[i] - 1);
            nullmask_t bitFromiPosition = NullValues::GetConcreteBitFromBitmask(inNullMask, i);
            atomicOr(reinterpret_cast<nullmask_cuda_t*>(outNullMask) + outBitMaskIdx,
                     (bitFromiPosition << outBitMaskShiftIdx));
        }
    }
}


__global__ void kernel_compress_null_mask(nullmask_t* outMask, nullarray_t* inArray, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        int outBitMaskIdx = NullValues::GetBitMaskIdx(i);
        int outBitMaskShiftIdx = NullValues::GetShiftMaskIdx(i);
        atomicOr(reinterpret_cast<nullmask_cuda_t*>(outMask) + outBitMaskIdx,
                 (inArray[i] & static_cast<nullmask_t>(1U)) << outBitMaskShiftIdx);
    }
}


__global__ void kernel_reconstruct_string_chars(GPUMemory::GPUString outStringCol,
                                                GPUMemory::GPUString inStringCol,
                                                int32_t* inStringLengths,
                                                int32_t* prefixSum,
                                                int8_t* inMask,
                                                int32_t stringCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < stringCount; i += stride)
    {
        if (inMask[i] && (prefixSum[i] - 1) >= 0)
        {
            int64_t inIndex = (i == 0) ? 0 : inStringCol.stringIndices[i - 1];
            int64_t outIndex = (prefixSum[i] - 1 == 0) ? 0 : outStringCol.stringIndices[prefixSum[i] - 2];
            for (int32_t j = 0; j < inStringLengths[i]; j++)
            {
                outStringCol.allChars[outIndex + j] = inStringCol.allChars[inIndex + j];
            }
        }
    }
}

/// Helping function to calculate number of digints of integer part of float
__device__ int32_t GetNumberOfIntegerPartDigits(float number)
{
    return (floorf(fabsf(number)) > 3.0f ? static_cast<int32_t>(log10f(floorf(fabsf(number)))) : 0) +
           1 + (number < 0 ? 1 : 0);
}

__global__ void kernel_predict_wkt_lengths(int32_t* outStringLengths, GPUMemory::GPUPolygon inPolygonCol, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        // Count POLYGON word and parentheses ("POLYGON((), ())")
        int32_t charCounter = 11 + (4 * (inPolygonCol.PolyCountAt(i) - 1));


        const int32_t subpolyStartIdx = inPolygonCol.PolyIdxAt(i);
        const int32_t subpolyEndIdx = subpolyStartIdx + inPolygonCol.PolyCountAt(i);

        // If subpolyStartIdx == subpolyEndIdx it means the row in the output polygon is empty
        // Set the char counter for the standard null value for polygons: POLYGON((0.0000 0.0000, 0.0000 0.0000))
        if (subpolyStartIdx == subpolyEndIdx)
        {
            charCounter = 7 + 4 + 4 * WKT_DECIMAL_PLACES + 12;
        }
        else
        {
            // Else calculate the required WKT size
            for (int32_t j = subpolyStartIdx; j < subpolyEndIdx; j++)
            {
                const int32_t pointCount = inPolygonCol.PointCountAt(j); // - 2;
                const int32_t pointStartIdx = inPolygonCol.PointIdxAt(j); // + 1;
                const int32_t pointEndIdx = pointStartIdx + pointCount;

                // Count the decimal part and colons between points (".0000 .0000, .0000 .0000")
                // Add +1 to add the first point to WKT
                charCounter += (pointCount + 1) * (2 * WKT_DECIMAL_PLACES + 5) - 2;
                for (int32_t k = pointStartIdx; k < pointEndIdx; k++)
                {
                    // Count the integer part ("150".0000, "-0".1000)
                    charCounter += GetNumberOfIntegerPartDigits(inPolygonCol.polyPoints[k].latitude) +
                                   GetNumberOfIntegerPartDigits(inPolygonCol.polyPoints[k].longitude);
                }

                // Repeat the last element for the WKT format
                charCounter +=
                    GetNumberOfIntegerPartDigits(inPolygonCol.polyPoints[pointStartIdx].latitude) +
                    GetNumberOfIntegerPartDigits(inPolygonCol.polyPoints[pointStartIdx].longitude);
            }
        }

        outStringLengths[i] = charCounter;
    }
}

__global__ void
kernel_generate_poly_submask(int8_t* outMask, int8_t* inMask, GPUMemory::GPUPolygon polygon, int32_t size)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < size; i += stride)
    {
        for (int32_t j = 0; j < polygon.PolyCountAt(i); j++)
        {
            outMask[polygon.PolyIdxAt(i) + j] = inMask[i];
        }
    }
}

__global__ void
kernel_generate_point_submask(int8_t* outMask, int8_t* inMask, GPUMemory::GPUPolygon polygon, int32_t size)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < size; i += stride)
    {
        for (int32_t j = 0; j < polygon.PointCountAt(i); j++)
        {
            outMask[polygon.PointIdxAt(i) + j] = inMask[i];
        }
    }
}

/// Kernel for reconstructing polygon subPolygons
__global__ void kernel_reconstruct_polyCount_col(int32_t* outPolyCount,
                                                 GPUMemory::GPUPolygon polygon,
                                                 int32_t* prefixSum,
                                                 int8_t* inMask,
                                                 int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        if (inMask[i] && (prefixSum[i] - 1) >= 0)
        {
            outPolyCount[prefixSum[i] - 1] = polygon.PolyCountAt(i);
        }
    }
}

/// Kernel for reconstructing polygon points
__global__ void kernel_reconstruct_pointCount_col(int32_t* outPointCount,
                                                  GPUMemory::GPUPolygon polygon,
                                                  int32_t* prefixSum,
                                                  int8_t* inMask,
                                                  int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride)
    {
        if (inMask[i] && (prefixSum[i] - 1) >= 0)
        {
            outPointCount[prefixSum[i] - 1] = polygon.PointCountAt(i);
        }
    }
}

/// Helping function to "print" float to GPU char array
__device__ void FloatToString(char* allChars, int64_t& startIndex, float number)
{
    // Append sign
    if (number < 0)
    {
        allChars[startIndex] = '-';
        // (note that there is no addres move because we will count with negative sign later)
    }

    // Append integer part
    int32_t integerPart = static_cast<int32_t>(floorf(fabsf(number)));
    int32_t digits = GetNumberOfIntegerPartDigits(number);
    startIndex += digits;
    do
    {
        allChars[--startIndex] = ('0' + (integerPart % 10));
        integerPart /= 10;
    } while (integerPart > 0); // Dynamic integer part places
    startIndex += digits - (number < 0 ? 1 : 0);

    // Append decimal part
    int32_t decimalPart =
        static_cast<int32_t>(roundf(fmodf(fabsf(number), 1.0f) * powf(10.0f, WKT_DECIMAL_PLACES)));
    allChars[startIndex++] = '.';
    startIndex += WKT_DECIMAL_PLACES;
    for (int32_t i = 0; i < WKT_DECIMAL_PLACES; i++) // Fixed decimal places
    {
        allChars[--startIndex] = ('0' + (decimalPart % 10));
        decimalPart /= 10;
    }
    startIndex += WKT_DECIMAL_PLACES;
}


__global__ void
kernel_convert_poly_to_wkt(GPUMemory::GPUString outWkt, GPUMemory::GPUPolygon inPolygonCol, int32_t dataElementCount)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < dataElementCount; i += stride) // via complex polygons
    {
        // "POLYGON("
        const int64_t stringStartIndex = (i == 0 ? 0 : outWkt.stringIndices[i - 1]);
        for (int32_t j = 0; j < 7; j++)
        {
            outWkt.allChars[stringStartIndex + j] = WKT_POLYGON[j];
        }
        int64_t charId = stringStartIndex + 7;
        outWkt.allChars[charId++] = '(';

        const int32_t subpolyStartIdx = inPolygonCol.PolyIdxAt(i);
        const int32_t subpolyEndIdx = subpolyStartIdx + inPolygonCol.PolyCountAt(i);

        // If subpolyStartIdx == subpolyEndIdx it means the row in the output polygon is empty
        // We need to add "(0.0000 0.0000, 0.0000 0.0000)" to the result
        if (subpolyStartIdx == subpolyEndIdx)
        {
            const char* emptyWkt = "(0.0000 0.0000, 0.0000 0.0000)";

            for (int32_t i = 0; i < (2 + 4 * WKT_DECIMAL_PLACES + 12); i++)
            {
                outWkt.allChars[charId++] = emptyWkt[i];
            }
        }
        else
        {

            for (int32_t j = subpolyStartIdx; j < subpolyEndIdx; j++) // via sub-polygons
            {
                outWkt.allChars[charId++] = '(';
                const int32_t pointCount = inPolygonCol.PointCountAt(j); // - 2;
                const int32_t pointStartIdx = inPolygonCol.PointIdxAt(j); // + 1;
                const int32_t pointEndIdx = pointStartIdx + pointCount;

                for (int32_t k = pointStartIdx; k < pointEndIdx; k++) // via points
                {
                    FloatToString(outWkt.allChars, charId, inPolygonCol.polyPoints[k].latitude);
                    outWkt.allChars[charId++] = ' ';
                    FloatToString(outWkt.allChars, charId, inPolygonCol.polyPoints[k].longitude);

                    outWkt.allChars[charId++] = ',';
                    outWkt.allChars[charId++] = ' ';

                    // Repeat the last element for the WKT reconstruction
                    if (k == pointEndIdx - 1)
                    {
                        FloatToString(outWkt.allChars, charId,
                                      inPolygonCol.polyPoints[pointStartIdx].latitude);
                        outWkt.allChars[charId++] = ' ';
                        FloatToString(outWkt.allChars, charId,
                                      inPolygonCol.polyPoints[pointStartIdx].longitude);
                    }
                }

                outWkt.allChars[charId++] = ')';
                if (j < subpolyEndIdx - 1)
                {
                    outWkt.allChars[charId++] = ',';
                    outWkt.allChars[charId++] = ' ';
                }
            }
        }
        outWkt.allChars[charId++] = ')';

        // Lengths mis-match check
        /*
        if (charId != outWkt.stringIndices[i])
        {
            printf("Not match fin id! %d\n", outWkt.stringIndices[i] - charId);
        }
        else{
            printf("Match OK\n");
        }
        */
    }
}

void GPUReconstruct::ReconstructStringColKeep(GPUMemory::GPUString* outStringCol,
                                              int32_t* outDataElementCount,
                                              GPUMemory::GPUString inStringCol,
                                              int8_t* inMask,
                                              int32_t inDataElementCount,
                                              nullmask_t** outNullMask,
                                              nullmask_t* nullMask)
{
    Context& context = Context::getInstance();

    if (inMask) // If mask is used (if inMask is not nullptr)
    {
        // Malloc a new buffer for the prefix sum vector
        cuda_ptr<int32_t> inPrefixSumPointer(inDataElementCount);
        PrefixSum(inPrefixSumPointer.get(), inMask, inDataElementCount);
        GPUMemory::copyDeviceToHost(outDataElementCount, inPrefixSumPointer.get() + inDataElementCount - 1, 1);

        if (*outDataElementCount > 0) // Not empty result set
        {
            int64_t inTotalCharCount;
            GPUMemory::copyDeviceToHost(&inTotalCharCount,
                                        inStringCol.stringIndices + inDataElementCount - 1, 1);

            // Compute lenghts from indices (reversed inclusive prefix sum)
            cuda_ptr<int32_t> inLengths(inDataElementCount);
            kernel_lengths_from_indices<<<context.calcGridDim(inDataElementCount), context.getBlockDim()>>>(
                inLengths.get(), inStringCol.stringIndices, inDataElementCount);

            // Reconstruct lenghts according to mask
            cuda_ptr<int32_t> outLengths(*outDataElementCount);
            kernel_reconstruct_col<<<context.calcGridDim(inDataElementCount), context.getBlockDim()>>>(
                outLengths.get(), inLengths.get(), inPrefixSumPointer.get(), inMask, inDataElementCount);

            // Compute new indices as prefix sum of reconstructed lengths
            GPUMemory::alloc(&(outStringCol->stringIndices), *outDataElementCount);
            PrefixSum(outStringCol->stringIndices, outLengths.get(), *outDataElementCount);

            int64_t outTotalCharCount;
            GPUMemory::copyDeviceToHost(&outTotalCharCount,
                                        outStringCol->stringIndices + *outDataElementCount - 1, 1);
            GPUMemory::alloc(&(outStringCol->allChars), outTotalCharCount);

            // Reconstruct chars
            kernel_reconstruct_string_chars<<<context.calcGridDim(inDataElementCount), context.getBlockDim()>>>(
                *outStringCol, inStringCol, inLengths.get(), inPrefixSumPointer.get(), inMask, inDataElementCount);
            if (nullMask && outNullMask)
            {
                const size_t outBitMaskSize = GPUMemory::CalculateNullMaskSize(*outDataElementCount, true);
                GPUMemory::allocAndSet(outNullMask, 0, outBitMaskSize);
                kernel_reconstruct_null_mask<<<context.calcGridDim(inDataElementCount), context.getBlockDim()>>>(
                    reinterpret_cast<nullmask_t*>(*outNullMask), nullMask, inPrefixSumPointer.get(),
                    inMask, inDataElementCount);
            }
        }
        else // Empty result set
        {
            outStringCol->allChars = nullptr;
            outStringCol->stringIndices = nullptr;
            if (outNullMask)
            {
                *outNullMask = nullptr;
            }
        }
    }
    else // If mask is not used (is nullptr), just copy pointers from inCol to outCol
    {
        *outStringCol = inStringCol;
        *outDataElementCount = inDataElementCount;
        if (outNullMask)
        {
            *outNullMask = nullMask;
        }
    }

    // Get last error
    CheckCudaError(cudaGetLastError());
}


void GPUReconstruct::ReconstructStringCol(std::string* outStringData,
                                          int32_t* outDataElementCount,
                                          GPUMemory::GPUString inStringCol,
                                          int8_t* inMask,
                                          int32_t inDataElementCount,
                                          nullmask_t* outNullMask,
                                          nullmask_t* nullMask)
{
    GPUMemory::GPUString outStringCol;
    if (inMask) // If mask is used (if inMask is not nullptr)
    {
        nullmask_t* outNullMaskGPUPointer = nullptr;
        ReconstructStringColKeep(&outStringCol, outDataElementCount, inStringCol, inMask,
                                 inDataElementCount, &outNullMaskGPUPointer, nullMask);
        if (outNullMaskGPUPointer)
        {
            const size_t outBitMaskSize = GPUMemory::CalculateNullMaskSize(*outDataElementCount);
            GPUMemory::copyDeviceToHost(outNullMask, outNullMaskGPUPointer, outBitMaskSize);
            if (inMask)
            {
                GPUMemory::free(outNullMaskGPUPointer);
            }
        }
    }
    else // If mask is not used
    {
        *outDataElementCount = inDataElementCount;
        outStringCol = inStringCol;
        if (nullMask)
        {
            const size_t outBitMaskSize = GPUMemory::CalculateNullMaskSize(*outDataElementCount);
            GPUMemory::copyDeviceToHost(outNullMask, nullMask, outBitMaskSize);
        }
    }

    if (*outDataElementCount > 0)
    {
        // Copy string indices to host
        std::unique_ptr<int64_t[]> hostStringIndices = std::make_unique<int64_t[]>(*outDataElementCount);
        GPUMemory::copyDeviceToHost(hostStringIndices.get(), outStringCol.stringIndices, *outDataElementCount);
        int64_t fullCharCount = hostStringIndices[*outDataElementCount - 1];

        // Copy all chars to host
        std::unique_ptr<char[]> hostAllChars = std::make_unique<char[]>(fullCharCount);
        GPUMemory::copyDeviceToHost(hostAllChars.get(), outStringCol.allChars, fullCharCount);

        // Fill output string array
        for (int32_t i = 0; i < *outDataElementCount; i++)
        {
            size_t length = static_cast<size_t>(
                i == 0 ? hostStringIndices[0] : hostStringIndices[i] - hostStringIndices[i - 1]);
            if (length == 0)
            {
                outStringData[i] = std::string();
            }
            else
            {
                outStringData[i] =
                    std::string(hostAllChars.get() + (i == 0 ? 0 : hostStringIndices[i - 1]), length);
            }
        }
        if (inMask)
        {
            // Free GPUString because it is not going out
            GPUMemory::free(outStringCol);
        }
    }
}


void GPUReconstruct::ReconstructStringColRaw(std::vector<int32_t>& keysStringLengths,
                                             std::vector<char>& keysAllChars,
                                             int32_t* outDataElementCount,
                                             GPUMemory::GPUString inStringCol,
                                             int8_t* inMask,
                                             int32_t inDataElementCount)
{
    Context& context = Context::getInstance();

    if (inMask) // If mask is used (if inMask is not nullptr)
    {
        // Malloc a new buffer for the prefix sum vector
        cuda_ptr<int32_t> inPrefixSumPointer(inDataElementCount);
        PrefixSum(inPrefixSumPointer.get(), inMask, inDataElementCount);
        GPUMemory::copyDeviceToHost(outDataElementCount, inPrefixSumPointer.get() + inDataElementCount - 1, 1);

        if (*outDataElementCount > 0) // Not empty result set
        {
            // Compute lenghts from indices (reversed inclusive prefix sum)
            cuda_ptr<int32_t> inLengths(inDataElementCount);
            kernel_lengths_from_indices<<<context.calcGridDim(inDataElementCount), context.getBlockDim()>>>(
                inLengths.get(), inStringCol.stringIndices, inDataElementCount);

            // Reconstruct lenghts according to mask
            cuda_ptr<int32_t> outLengths(*outDataElementCount);
            kernel_reconstruct_col<<<context.calcGridDim(inDataElementCount), context.getBlockDim()>>>(
                outLengths.get(), inLengths.get(), inPrefixSumPointer.get(), inMask, inDataElementCount);
            // Copy lengths to host
            keysStringLengths.resize(*outDataElementCount);
            GPUMemory::copyDeviceToHost(keysStringLengths.data(), outLengths.get(), *outDataElementCount);

            // Compute new indices as prefix sum of reconstructed lengths
            GPUMemory::GPUString outStringCol;
            GPUMemory::alloc(&(outStringCol.stringIndices), *outDataElementCount);
            PrefixSum(outStringCol.stringIndices, outLengths.get(), *outDataElementCount);

            int64_t outTotalCharCount;
            GPUMemory::copyDeviceToHost(&outTotalCharCount,
                                        outStringCol.stringIndices + *outDataElementCount - 1, 1);
            GPUMemory::alloc(&(outStringCol.allChars), outTotalCharCount);

            // Reconstruct chars
            kernel_reconstruct_string_chars<<<context.calcGridDim(inDataElementCount), context.getBlockDim()>>>(
                outStringCol, inStringCol, inLengths.get(), inPrefixSumPointer.get(), inMask, inDataElementCount);
            // Copy chars to host
            keysAllChars.resize(outTotalCharCount);
            GPUMemory::copyDeviceToHost(keysAllChars.data(), outStringCol.allChars, outTotalCharCount);
            GPUMemory::free(outStringCol);
        }
    }
    else // If mask is not used (is nullptr), just copy pointers from inCol to outCol
    {
        *outDataElementCount = inDataElementCount;

        // Compute lenghts from indices (reversed inclusive prefix sum)
        int64_t outTotalCharCount;
        GPUMemory::copyDeviceToHost(&outTotalCharCount, inStringCol.stringIndices + inDataElementCount - 1, 1);
        cuda_ptr<int32_t> lengths(inDataElementCount);
        kernel_lengths_from_indices<<<context.calcGridDim(inDataElementCount), context.getBlockDim()>>>(
            lengths.get(), inStringCol.stringIndices, inDataElementCount);
        keysStringLengths.resize(inDataElementCount);
        GPUMemory::copyDeviceToHost(keysStringLengths.data(), lengths.get(), inDataElementCount);
        keysAllChars.resize(outTotalCharCount);
        GPUMemory::copyDeviceToHost(keysAllChars.data(), inStringCol.allChars, outTotalCharCount);
    }

    // Get last error
    CheckCudaError(cudaGetLastError());
}


void GPUReconstruct::ConvertPolyColToWKTCol(GPUMemory::GPUString& outStringCol,
                                            GPUMemory::GPUPolygon inPolygonCol,
                                            int32_t dataElementCount)
{
    Context& context = Context::getInstance();
    if (dataElementCount > 0)
    {
        // "Predict" (pre-calculate) string lengths
        cuda_ptr<int32_t> stringLengths(dataElementCount);
        kernel_predict_wkt_lengths<<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(
            stringLengths.get(), inPolygonCol, dataElementCount);
        CheckCudaError(cudaGetLastError());

        // Alloc and compute string indices as a prefix sum of the string lengths
        GPUMemory::alloc(&(outStringCol.stringIndices), dataElementCount);
        PrefixSum(outStringCol.stringIndices, stringLengths.get(), dataElementCount);

        // Get total char count and alloc array for all chars
        int64_t totalCharCount;
        GPUMemory::copyDeviceToHost(&totalCharCount, outStringCol.stringIndices + dataElementCount - 1, 1);
        GPUMemory::alloc(&(outStringCol.allChars), totalCharCount);

        // Finally convert polygons to WKTs
        kernel_convert_poly_to_wkt<<<context.calcGridDim(dataElementCount), context.getBlockDim()>>>(
            outStringCol, inPolygonCol, dataElementCount);
        CheckCudaError(cudaGetLastError());
    }
    else
    {
        outStringCol.allChars = nullptr;
        outStringCol.stringIndices = nullptr;
    }
}

void GPUReconstruct::ReconstructPolyColKeep(GPUMemory::GPUPolygon* outCol,
                                            int32_t* outDataElementCount,
                                            GPUMemory::GPUPolygon inCol,
                                            int8_t* inMask,
                                            int32_t inDataElementCount,
                                            nullmask_t** outNullMask,
                                            nullmask_t* nullMask)
{
    Context& context = Context::getInstance();

    if (inMask) // If mask is used (if inMask is not nullptr)
    {
        // A buffer for the prefix sum vector
        cuda_ptr<int32_t> inPrefixSumPointer(inDataElementCount);
        PrefixSum(inPrefixSumPointer.get(), inMask, inDataElementCount);
        GPUMemory::copyDeviceToHost(outDataElementCount, inPrefixSumPointer.get() + inDataElementCount - 1, 1);

        if (*outDataElementCount > 0) // Not empty result set
        {
            // Reconstruct each array independently
            int32_t inSubpolySize;
            GPUMemory::copyDeviceToHost(&inSubpolySize, inCol.polyIdx + inDataElementCount - 1, 1);

            // Complex polygons (reconstruct polyCount and sum it to polyIdx)
            // Alloc a temp count buffer and the result index buffer
            cuda_ptr<int32_t> polyCount(*outDataElementCount);
            GPUMemory::alloc(&(outCol->polyIdx), *outDataElementCount);

            kernel_reconstruct_polyCount_col<<<context.calcGridDim(inDataElementCount), context.getBlockDim()>>>(
                polyCount.get(), inCol, inPrefixSumPointer.get(), inMask, inDataElementCount);
            CheckCudaError(cudaGetLastError());

            PrefixSum(outCol->polyIdx, polyCount.get(), *outDataElementCount);

            // Subpolygons (reconstruct pointCount and sum it to pointIdx)
            int32_t outSubpolySize;
            GPUMemory::copyDeviceToHost(&outSubpolySize, outCol->polyIdx + *outDataElementCount - 1, 1);

            // If result set is empty
            if (outSubpolySize == 0)
            {
                outCol->pointIdx = nullptr;
                outCol->polyPoints = nullptr;

                return;
            }

            int32_t inPointSize;
            GPUMemory::copyDeviceToHost(&inPointSize, inCol.pointIdx + inSubpolySize - 1, 1);

            cuda_ptr<int8_t> subpolyMask(inSubpolySize);
            kernel_generate_poly_submask<<<context.calcGridDim(inDataElementCount), context.getBlockDim()>>>(
                subpolyMask.get(), inMask, inCol, inDataElementCount);
            CheckCudaError(cudaGetLastError());

            cuda_ptr<int32_t> subpolyPrefixSumPointer(inSubpolySize);
            PrefixSum(subpolyPrefixSumPointer.get(), subpolyMask.get(), inSubpolySize);

            cuda_ptr<int32_t> pointCount(outSubpolySize);
            GPUMemory::alloc(&(outCol->pointIdx), outSubpolySize);

            kernel_reconstruct_pointCount_col<<<context.calcGridDim(inSubpolySize), context.getBlockDim()>>>(
                pointCount.get(), inCol, subpolyPrefixSumPointer.get(), subpolyMask.get(), inSubpolySize);
            CheckCudaError(cudaGetLastError());
            PrefixSum(outCol->pointIdx, pointCount.get(), outSubpolySize);

            // Points (reconstruct polyPoints)
            int32_t outPointSize;
            GPUMemory::copyDeviceToHost(&outPointSize, outCol->pointIdx + outSubpolySize - 1, 1);

            cuda_ptr<int8_t> pointMask(inPointSize);
            kernel_generate_point_submask<<<context.calcGridDim(inSubpolySize), context.getBlockDim()>>>(
                pointMask.get(), subpolyMask.get(), inCol, inSubpolySize);
            CheckCudaError(cudaGetLastError());

            cuda_ptr<int32_t> pointPrefixSumPointer(inPointSize);
            PrefixSum(pointPrefixSumPointer.get(), pointMask.get(), inPointSize);

            GPUMemory::alloc(&(outCol->polyPoints), outPointSize);
            kernel_reconstruct_col<<<context.calcGridDim(inSubpolySize), context.getBlockDim()>>>(
                outCol->polyPoints, inCol.polyPoints, pointPrefixSumPointer.get(), pointMask.get(), inPointSize);
            CheckCudaError(cudaGetLastError());
            if (nullMask && outNullMask)
            {
                const size_t outBitMaskSize = GPUMemory::CalculateNullMaskSize(*outDataElementCount, true);
                GPUMemory::allocAndSet(outNullMask, 0, outBitMaskSize);
                kernel_reconstruct_null_mask<<<context.calcGridDim(inDataElementCount), context.getBlockDim()>>>(
                    reinterpret_cast<nullmask_t*>(*outNullMask), nullMask, inPrefixSumPointer.get(),
                    inMask, inDataElementCount);
            }
        }
        else // Empty result set
        {
            outCol->polyPoints = nullptr;
            outCol->pointIdx = nullptr;
            outCol->polyIdx = nullptr;
            if (outNullMask)
            {
                *outNullMask = nullptr;
            }
        }
    }
    else // If mask is not used (is nullptr), just copy pointers from inCol to outCol
    {
        *outCol = inCol;
        *outDataElementCount = inDataElementCount;
        if (outNullMask)
        {
            *outNullMask = nullMask;
        }
    }

    // Get last error
    CheckCudaError(cudaGetLastError());
}


void GPUReconstruct::ReconstructPolyColToWKT(std::string* outStringData,
                                             int32_t* outDataElementCount,
                                             GPUMemory::GPUPolygon inPolygonCol,
                                             int8_t* inMask,
                                             int32_t inDataElementCount,
                                             nullmask_t* outNullMask,
                                             nullmask_t* nullMask)
{
    GPUMemory::GPUPolygon reconstructedPolygonCol;
    nullmask_t* outNullMaskGPUPointer = nullptr;
    ReconstructPolyColKeep(&reconstructedPolygonCol, outDataElementCount, inPolygonCol, inMask,
                           inDataElementCount, &outNullMaskGPUPointer, nullMask);
    if (outNullMaskGPUPointer)
    {
        const size_t outBitMaskSize = GPUMemory::CalculateNullMaskSize(*outDataElementCount);
        GPUMemory::copyDeviceToHost(outNullMask, outNullMaskGPUPointer, outBitMaskSize);
        if (inMask)
        {
            GPUMemory::free(outNullMaskGPUPointer);
        }
    }
    GPUMemory::GPUString gpuWkt;
    ConvertPolyColToWKTCol(gpuWkt, reconstructedPolygonCol, *outDataElementCount);
    if (inMask)
    {
        GPUMemory::free(reconstructedPolygonCol);
    }
    // Use reconstruct without mask - just to convert GPUString to CPU string array
    ReconstructStringCol(outStringData, outDataElementCount, gpuWkt, nullptr, *outDataElementCount);
    if (gpuWkt.allChars)
    {
        GPUMemory::free(gpuWkt);
    }
}

void GPUReconstruct::ReconstructPointColToWKT(std::string* outStringData,
                                              int32_t* outDataElementCount,
                                              NativeGeoPoint* inPointCol,
                                              int8_t* inMask,
                                              int32_t inDataElementCount,
                                              nullmask_t* outNullMask,
                                              nullmask_t* nullMask)
{
    NativeGeoPoint* reconstructedPointCol;
    nullmask_t* outNullMaskGPUPointer = nullptr;
    reconstructColKeep<NativeGeoPoint>(&reconstructedPointCol, outDataElementCount, inPointCol,
                                       inMask, inDataElementCount, &outNullMaskGPUPointer, nullMask);
    GPUMemory::GPUString gpuWkt;
    if (outNullMaskGPUPointer)
    {
        const size_t outBitMaskSize = GPUMemory::CalculateNullMaskSize(*outDataElementCount);
        GPUMemory::copyDeviceToHost(outNullMask, outNullMaskGPUPointer, outBitMaskSize);
        if (inMask)
        {
            GPUMemory::free(outNullMaskGPUPointer);
        }
    }
    ConvertPointColToWKTCol(gpuWkt, reconstructedPointCol, *outDataElementCount);
    if (inMask && reconstructedPointCol)
    {
        GPUMemory::free(reconstructedPointCol);
    }
    // Use reconstruct without mask - just to convert GPUString to CPU string array
    ReconstructStringCol(outStringData, outDataElementCount, gpuWkt, nullptr, *outDataElementCount);
    if (!gpuWkt.allChars)
    {
        GPUMemory::free(gpuWkt);
    }
}


cuda_ptr<nullmask_t> GPUReconstruct::CompressNullMask(nullarray_t* inputNullArray, int32_t dataElementCount)
{
    const size_t outBitMaskSize = GPUMemory::CalculateNullMaskSize(dataElementCount, true);
    cuda_ptr<nullmask_t> nullMaskCompressed(outBitMaskSize, 0);
    kernel_compress_null_mask<<<Context::getInstance().calcGridDim(dataElementCount),
                                Context::getInstance().getBlockDim()>>>(
        reinterpret_cast<nullmask_t*>(nullMaskCompressed.get()), inputNullArray, dataElementCount);
    return nullMaskCompressed;
}


template <>
void GPUReconstruct::reconstructCol<QikkDB::Types::Point>(QikkDB::Types::Point* outData,
                                                             int32_t* outDataElementCount,
                                                             QikkDB::Types::Point* ACol,
                                                             int8_t* inMask,
                                                             int32_t dataElementCount,
                                                             nullmask_t* outNullMask,
                                                             nullmask_t* nullMask)
{
    // Not supported, just throw an error
    CheckQueryEngineError(QueryEngineErrorType::GPU_EXTENSION_ERROR,
                          "ReconstructCol of Point not supported, use GenerateIndexes instead");
}

template <>
void GPUReconstruct::reconstructCol<QikkDB::Types::ComplexPolygon>(QikkDB::Types::ComplexPolygon* outData,
                                                                      int32_t* outDataElementCount,
                                                                      QikkDB::Types::ComplexPolygon* ACol,
                                                                      int8_t* inMask,
                                                                      int32_t dataElementCount,
                                                                      nullmask_t* outNullMask,
                                                                      nullmask_t* nullMask)
{
    // Not supported, just throw an error
    CheckQueryEngineError(QueryEngineErrorType::GPU_EXTENSION_ERROR,
                          "ReconstructCol of ComplexPolygon not supported, use GenerateIndexes "
                          "instead");
}

template <>
void GPUReconstruct::reconstructColKeep<QikkDB::Types::Point>(QikkDB::Types::Point** outCol,
                                                                 int32_t* outDataElementCount,
                                                                 QikkDB::Types::Point* ACol,
                                                                 int8_t* inMask,
                                                                 int32_t dataElementCount,
                                                                 nullmask_t** outNullMask,
                                                                 nullmask_t* nullMask)
{
    // Not supported, just throw an error
    CheckQueryEngineError(QueryEngineErrorType::GPU_EXTENSION_ERROR,
                          "ReconstructColKeep of Point not supported, use GenerateIndexes instead");
}

template <>
void GPUReconstruct::reconstructColKeep<QikkDB::Types::ComplexPolygon>(QikkDB::Types::ComplexPolygon** outCol,
                                                                          int32_t* outDataElementCount,
                                                                          QikkDB::Types::ComplexPolygon* ACol,
                                                                          int8_t* inMask,
                                                                          int32_t dataElementCount,
                                                                          nullmask_t** outNullMask,
                                                                          nullmask_t* nullMask)
{
    // Not supported, just throw an error
    CheckQueryEngineError(QueryEngineErrorType::GPU_EXTENSION_ERROR,
                          "ReconstructColKeep of ComplexPolygon not supported, use GenerateIndexes "
                          "instead");
}
