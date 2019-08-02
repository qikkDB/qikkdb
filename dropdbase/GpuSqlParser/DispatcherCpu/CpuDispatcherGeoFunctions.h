#pragma once

#include "../CpuSqlDispatcher.h"

template <typename T, typename U>
int32_t CpuSqlDispatcher::pointColCol()
{
    auto colNameLeft = arguments.read<std::string>();
    auto colNameRight = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    int8_t* maskMin = allocateRegister<int8_t>(reg + "_min", 1, true);
    int8_t* maskMax = allocateRegister<int8_t>(reg + "_max", 1, true);

    maskMin[0] = 1;
    maskMax[0] = 1;

    return 0;
}

/// Implementation of POINT(a, b) operation dispatching - concatenation of two numeric attributes to single point column
/// Implementation for column constant case
/// Pops data from argument memory stream and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename T, typename U>
int32_t CpuSqlDispatcher::pointColConst()
{
    auto colNameLeft = arguments.read<std::string>();
    U cnst = arguments.read<U>();
    auto reg = arguments.read<std::string>();

    int8_t* maskMin = allocateRegister<int8_t>(reg + "_min", 1, true);
    int8_t* maskMax = allocateRegister<int8_t>(reg + "_max", 1, true);

    maskMin[0] = 1;
    maskMax[0] = 1;

    return 0;
}

/// Implementation of POINT(a, b) operation dispatching - concatenation of two numeric attributes to single point column
/// Implementation for onstant column case
/// Pops data from argument memory stream and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename T, typename U>
int32_t CpuSqlDispatcher::pointConstCol()
{
    T cnst = arguments.read<T>();
    auto colNameRight = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    int8_t* maskMin = allocateRegister<int8_t>(reg + "_min", 1, true);
    int8_t* maskMax = allocateRegister<int8_t>(reg + "_max", 1, true);

    maskMin[0] = 1;
    maskMax[0] = 1;

    return 0;
}

/// Implementation of CONTAINS(a, b) operation dispatching - point in polygon
/// Implementation for column constant case
/// Pops data from argument memory stream, converts geo literals to their gpu representation and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename T, typename U>
int32_t CpuSqlDispatcher::containsColConst()
{
    auto colName = arguments.read<std::string>();
    auto constWkt = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    int8_t* maskMin = allocateRegister<int8_t>(reg + "_min", 1, true);
    int8_t* maskMax = allocateRegister<int8_t>(reg + "_max", 1, true);

    maskMin[0] = 1;
    maskMax[0] = 1;

    return 0;
}

/// Implementation of CONTAINS(a, b) operation dispatching - point in polygon
/// Implementation for constant column case
/// Pops data from argument memory stream, converts geo literals to their gpu representation and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename T, typename U>
int32_t CpuSqlDispatcher::containsConstCol()
{
    auto colName = arguments.read<std::string>();
    auto constWkt = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    int8_t* maskMin = allocateRegister<int8_t>(reg + "_min", 1, true);
    int8_t* maskMax = allocateRegister<int8_t>(reg + "_max", 1, true);

    maskMin[0] = 1;
    maskMax[0] = 1;

    return 0;
}

/// Implementation of CONTAINS(a, b) operation dispatching - point in polygon
/// Implementation for column column case
/// Pops data from argument memory stream, converts geo literals to their gpu representation and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename T, typename U>
int32_t CpuSqlDispatcher::containsColCol()
{
    auto colNamePolygon = arguments.read<std::string>();
    auto colNamePoint = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    int8_t* maskMin = allocateRegister<int8_t>(reg + "_min", 1, true);
    int8_t* maskMax = allocateRegister<int8_t>(reg + "_max", 1, true);

    maskMin[0] = 1;
    maskMax[0] = 1;

    return 0;
}

/// Implementation of CONTAINS(a, b) operation dispatching - point in polygon
/// Implementation for constant constant case
/// Pops data from argument memory stream, converts geo literals to their gpu representation and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename T, typename U>
int32_t CpuSqlDispatcher::containsConstConst()
{
    auto constPolygonWkt = arguments.read<std::string>();
    auto constPointWkt = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    int8_t* maskMin = allocateRegister<int8_t>(reg + "_min", 1, true);
    int8_t* maskMax = allocateRegister<int8_t>(reg + "_max", 1, true);

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
int32_t CpuSqlDispatcher::polygonOperationColConst()
{
    auto colName = arguments.read<std::string>();
    auto constWkt = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    int8_t* maskMin = allocateRegister<int8_t>(reg + "_min", 1, true);
    int8_t* maskMax = allocateRegister<int8_t>(reg + "_max", 1, true);

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
int32_t CpuSqlDispatcher::polygonOperationConstCol()
{
    auto constWkt = arguments.read<std::string>();
    auto colName = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    int8_t* maskMin = allocateRegister<int8_t>(reg + "_min", 1, true);
    int8_t* maskMax = allocateRegister<int8_t>(reg + "_max", 1, true);

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
int32_t CpuSqlDispatcher::polygonOperationColCol()
{
    auto colNameLeft = arguments.read<std::string>();
    auto colNameRight = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    int8_t* maskMin = allocateRegister<int8_t>(reg + "_min", 1, true);
    int8_t* maskMax = allocateRegister<int8_t>(reg + "_max", 1, true);

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
int32_t CpuSqlDispatcher::polygonOperationConstConst()
{
    auto constWktLeft = arguments.read<std::string>();
    auto constWktRight = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    int8_t* maskMin = allocateRegister<int8_t>(reg + "_min", 1, true);
    int8_t* maskMax = allocateRegister<int8_t>(reg + "_max", 1, true);

    maskMin[0] = 1;
    maskMax[0] = 1;

    return 0;
}