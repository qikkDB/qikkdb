#pragma once

// Struct for polygon argument concatenation
/// <summary>
/// A structure to represent a complex polygon
/// </summary>
/// <param name="geoPoints">points of all polygons</param>
/// <param name="complexPolygonIdx">Start indices of range of polygons in polygon arrays, for each complex polygon </param>
/// <param name="complexPolygonCnt">Length of the polygon range for each complex polygon</param> 
/// <param name="polygonIdx">Start indices of range of points in points array, for each polygon</param> 
/// <param name="polygonCnt">Length of the point range for each polygon</param> 
/// <param name="polygonCount">Length of complexPolygonIdx and complexPolygonCnt</param>
struct ComplexPolygon
{
    NativeGeoPoint* geoPoints;
    int32_t* complexPolygonIdx;
    int32_t* complexPolygonCnt;
    int32_t* polygonIdx;
    int32_t* polygonCnt;
    int32_t polygonCount;
};
