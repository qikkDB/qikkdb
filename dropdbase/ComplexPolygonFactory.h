#pragma once

#include "QueryEngine/GPUCore/GPUMemory.cuh"

namespace ColmnarDB
{
namespace Types
{
class ComplexPolygon;
}
} // namespace ColmnarDB

class ComplexPolygonFactory
{
private:
    ComplexPolygonFactory(){};

public:
    static const int MAX_POLYGONS_NUMBER = 8;

    /// <summary>
    /// Converts polygons to GPU representation.
    /// </summary>
    /// <param name="polygons">Polygons to convert.</param>
    /// <returns>Tuple of array for the GPU.</returns>
    static GPUMemory::GPUPolygon PrepareGPUPolygon(const std::vector<ColmnarDB::Types::ComplexPolygon>& polygons);


    static GPUMemory::GPUPolygon PrepareGPUPolygon(const std::vector<ColmnarDB::Types::ComplexPolygon>& polygons,
                                                   const std::string& databaseName,
                                                   const std::string& columnName,
                                                   size_t blockIndex,
                                                   int64_t loadSize,
                                                   int64_t loadOffset);

    /// <summary>
    /// Constructor for creating complex polygon and initializing.
    /// </summary>
    /// <param name="wktPolygon">String of well known text formatted polygon.</param>
    /// <param name="spaceBetweenItems">Represents if there is a space between geo points after a comma and between
    /// polygons also after a comma. Default value is set to 'false'.</param>
    /// <exception cref="FormatException">Format exception with a message that explains the reason why the
    /// exception have been thrown.</exception>
    static ColmnarDB::Types::ComplexPolygon FromWkt(std::string wkt);

    /// <summary>
    /// Method that converts class to a string representation.
    /// </summary>
    /// <returns>ComplexPolygon in format of well known text.</returns>
    static std::string
    WktFromPolygon(const ColmnarDB::Types::ComplexPolygon& polygon, bool fixedPrecision = false);
};