#pragma once

#include "../CpuSqlDispatcher.h"

template <typename T, typename U>
int32_t CpuSqlDispatcher::PointColCol()
{
    auto colNameLeft = arguments_.Read<std::string>();
    auto colNameRight = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int8_t* maskMin = AllocateRegister<int8_t>(reg + "_min", 1, true);
    int8_t* maskMax = AllocateRegister<int8_t>(reg + "_max", 1, true);

    maskMin[0] = 1;
    maskMax[0] = 1;

    return 0;
}

/// Implementation of POINT(a, b) operation dispatching - concatenation of two numeric attributes to single point column
/// Implementation for column constant case
/// Pops data from argument memory stream and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename T, typename U>
int32_t CpuSqlDispatcher::PointColConst()
{
    auto colNameLeft = arguments_.Read<std::string>();
    U cnst = arguments_.Read<U>();
    auto reg = arguments_.Read<std::string>();

    int8_t* maskMin = AllocateRegister<int8_t>(reg + "_min", 1, true);
    int8_t* maskMax = AllocateRegister<int8_t>(reg + "_max", 1, true);

    maskMin[0] = 1;
    maskMax[0] = 1;

    return 0;
}

/// Implementation of POINT(a, b) operation dispatching - concatenation of two numeric attributes to single point column
/// Implementation for onstant column case
/// Pops data from argument memory stream and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename T, typename U>
int32_t CpuSqlDispatcher::PointConstCol()
{
    T cnst = arguments_.Read<T>();
    auto colNameRight = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int8_t* maskMin = AllocateRegister<int8_t>(reg + "_min", 1, true);
    int8_t* maskMax = AllocateRegister<int8_t>(reg + "_max", 1, true);

    maskMin[0] = 1;
    maskMax[0] = 1;

    return 0;
}

/// Implementation of CONTAINS(a, b) operation dispatching - point in polygon
/// Implementation for column constant case
/// Pops data from argument memory stream, converts geo literals to their gpu representation and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename T, typename U>
int32_t CpuSqlDispatcher::ContainsColConst()
{
    auto colName = arguments_.Read<std::string>();
    auto constWkt = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int8_t* maskMin = AllocateRegister<int8_t>(reg + "_min", 1, true);
    int8_t* maskMax = AllocateRegister<int8_t>(reg + "_max", 1, true);

    maskMin[0] = 1;
    maskMax[0] = 1;

    return 0;
}

/// Implementation of CONTAINS(a, b) operation dispatching - point in polygon
/// Implementation for constant column case
/// Pops data from argument memory stream, converts geo literals to their gpu representation and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename T, typename U>
int32_t CpuSqlDispatcher::ContainsConstCol()
{
    auto colName = arguments_.Read<std::string>();
    auto constWkt = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int8_t* maskMin = AllocateRegister<int8_t>(reg + "_min", 1, true);
    int8_t* maskMax = AllocateRegister<int8_t>(reg + "_max", 1, true);

    maskMin[0] = 1;
    maskMax[0] = 1;

    return 0;
}

/// Implementation of CONTAINS(a, b) operation dispatching - point in polygon
/// Implementation for column column case
/// Pops data from argument memory stream, converts geo literals to their gpu representation and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename T, typename U>
int32_t CpuSqlDispatcher::ContainsColCol()
{
    auto colNamePolygon = arguments_.Read<std::string>();
    auto colNamePoint = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int8_t* maskMin = AllocateRegister<int8_t>(reg + "_min", 1, true);
    int8_t* maskMax = AllocateRegister<int8_t>(reg + "_max", 1, true);

    maskMin[0] = 1;
    maskMax[0] = 1;

    return 0;
}

/// Implementation of CONTAINS(a, b) operation dispatching - point in polygon
/// Implementation for constant constant case
/// Pops data from argument memory stream, converts geo literals to their gpu representation and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename T, typename U>
int32_t CpuSqlDispatcher::ContainsConstConst()
{
    auto constPolygonWkt = arguments_.Read<std::string>();
    auto constPointWkt = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int8_t* maskMin = AllocateRegister<int8_t>(reg + "_min", 1, true);
    int8_t* maskMax = AllocateRegister<int8_t>(reg + "_max", 1, true);

    maskMin[0] = 1;
    maskMax[0] = 1;

    return 0;
}

/// Implementation of genric polygon operation (operation which also outputs polygon - CONTAINS does
/// not meet this requrement) based on functor OP eg. INTRSECT(a,b), UNION(a,b) Implementation for
/// column constant case Pops data from argument memory stream, converts geo literals to their gpu
/// representation and loads data to GPU on demand <returns name="statusCode">Finish status code of
/// the operation</returns>
template <typename OP, typename T, typename U>
int32_t CpuSqlDispatcher::PolygonOperationColConst()
{
    auto colName = arguments_.Read<std::string>();
    auto constWkt = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int8_t* maskMin = AllocateRegister<int8_t>(reg + "_min", 1, true);
    int8_t* maskMax = AllocateRegister<int8_t>(reg + "_max", 1, true);

    maskMin[0] = 1;
    maskMax[0] = 1;

    return 0;
}

/// Implementation of genric polygon operation (operation which also outputs polygon - CONTAINS does
/// not meet this requrement) based on functor OP eg. INTRSECT(a,b), UNION(a,b) Implementation for
/// constant column case Pops data from argument memory stream, converts geo literals to their gpu
/// representation and loads data to GPU on demand <returns name="statusCode">Finish status code of
/// the operation</returns>
template <typename OP, typename T, typename U>
int32_t CpuSqlDispatcher::PolygonOperationConstCol()
{
    auto constWkt = arguments_.Read<std::string>();
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int8_t* maskMin = AllocateRegister<int8_t>(reg + "_min", 1, true);
    int8_t* maskMax = AllocateRegister<int8_t>(reg + "_max", 1, true);

    maskMin[0] = 1;
    maskMax[0] = 1;

    return 0;
}
/// Implementation of genric polygon operation (operation which also outputs polygon - CONTAINS does
/// not meet this requrement) based on functor OP eg. INTRSECT(a,b), UNION(a,b) Implementation for
/// column column case Pops data from argument memory stream, converts geo literals to their gpu
/// representation and loads data to GPU on demand <returns name="statusCode">Finish status code of
/// the operation</returns>
template <typename OP, typename T, typename U>
int32_t CpuSqlDispatcher::PolygonOperationColCol()
{
    auto colNameLeft = arguments_.Read<std::string>();
    auto colNameRight = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int8_t* maskMin = AllocateRegister<int8_t>(reg + "_min", 1, true);
    int8_t* maskMax = AllocateRegister<int8_t>(reg + "_max", 1, true);

    maskMin[0] = 1;
    maskMax[0] = 1;

    return 0;
}

/// Implementation of genric polygon operation (operation which also outputs polygon - CONTAINS does
/// not meet this requrement) based on functor OP eg. INTRSECT(a,b), UNION(a,b) Implementation for
/// constant constant case Pops data from argument memory stream, converts geo literals to their gpu
/// representation and loads data to GPU on demand <returns name="statusCode">Finish status code of
/// the operation</returns>
template <typename OP, typename T, typename U>
int32_t CpuSqlDispatcher::PolygonOperationConstConst()
{
    auto constWktLeft = arguments_.Read<std::string>();
    auto constWktRight = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int8_t* maskMin = AllocateRegister<int8_t>(reg + "_min", 1, true);
    int8_t* maskMax = AllocateRegister<int8_t>(reg + "_max", 1, true);

    maskMin[0] = 1;
    maskMax[0] = 1;

    return 0;
}