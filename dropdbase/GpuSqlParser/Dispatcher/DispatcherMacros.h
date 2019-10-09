#pragma once
#define BEGIN_DISPATCH_TABLE(OP) \
    std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE* DataType::DATA_TYPE_SIZE> OP = {

#define END_DISPATCH_TABLE \
    }                      \
    ;

#define DISPATCHER_FUN(DOP, args...) &DOP<args>
#define DISPATCHER_FUNCTION(DOP, args...)                                       \
    DISPATCHER_FUN(DOP##ConstConst, args), DISPATCHER_FUN(DOP##ConstCol, args), \
        DISPATCHER_FUN(DOP##ColConst, args), DISPATCHER_FUN(DOP##ColCol, args),

#define DISPATCHER_ERR(SUFFIX, args...) \
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandler##SUFFIX<args>

#define DISPATCHER_ERROR(args...)                                     \
    DISPATCHER_ERR(ConstConst, args), DISPATCHER_ERR(ConstCol, args), \
        DISPATCHER_ERR(ColConst, args), DISPATCHER_ERR(ColCol, args),

#define DISPATCHER_INVALID_TYPE(OP, TYPE)                        \
    DISPATCHER_ERROR(OP, TYPE, int32_t)                          \
    DISPATCHER_ERROR(OP, TYPE, int64_t)                          \
    DISPATCHER_ERROR(OP, TYPE, float)                            \
    DISPATCHER_ERROR(OP, TYPE, double)                           \
    DISPATCHER_ERROR(OP, TYPE, ColmnarDB::Types::Point)          \
    DISPATCHER_ERROR(OP, TYPE, ColmnarDB::Types::ComplexPolygon) \
    DISPATCHER_ERROR(OP, TYPE, std::string)                      \
    DISPATCHER_ERROR(OP, TYPE, int8_t)

#define DISPATCHER_INT32_ENABLE_0(DOP, OP, TYPE) DISPATCHER_ERROR(OP, TYPE, int32_t)
#define DISPATCHER_INT32_ENABLE_1(DOP, OP, TYPE) DISPATCHER_FUNCTION(DOP, OP, TYPE, int32_t)

#define DISPATCHER_INT64_ENABLE_0(DOP, OP, TYPE) DISPATCHER_ERROR(OP, TYPE, int32_t)
#define DISPATCHER_INT64_ENABLE_1(DOP, OP, TYPE) DISPATCHER_FUNCTION(DOP, OP, TYPE, int32_t)

#define DISPATCHER_FLOAT_ENABLE_0(DOP, OP, TYPE) DISPATCHER_ERROR(OP, TYPE, int32_t)
#define DISPATCHER_FLOAT_ENABLE_1(DOP, OP, TYPE) DISPATCHER_FUNCTION(DOP, OP, TYPE, int32_t)

#define DISPATCHER_DOUBLE_ENABLE_0(DOP, OP, TYPE) DISPATCHER_ERROR(OP, TYPE, int32_t)
#define DISPATCHER_DOUBLE_ENABLE_1(DOP, OP, TYPE) DISPATCHER_FUNCTION(DOP, OP, TYPE, int32_t)

#define DISPATCHER_POINT_ENABLE_0(DOP, OP, TYPE) DISPATCHER_ERROR(OP, TYPE, int32_t)
#define DISPATCHER_POINT_ENABLE_1(DOP, OP, TYPE) DISPATCHER_FUNCTION(DOP, OP, TYPE, int32_t)

#define DISPATCHER_POLYGON_ENABLE_0(DOP, OP, TYPE) DISPATCHER_ERROR(OP, TYPE, int32_t)
#define DISPATCHER_POLYGON_ENABLE_1(DOP, OP, TYPE) DISPATCHER_FUNCTION(DOP, OP, TYPE, int32_t)

#define DISPATCHER_STRING_ENABLE_0(DOP, OP, TYPE) DISPATCHER_ERROR(OP, TYPE, int32_t)
#define DISPATCHER_STRING_ENABLE_1(DOP, OP, TYPE) DISPATCHER_FUNCTION(DOP, OP, TYPE, int32_t)

#define DISPATCHER_INT8_ENABLE_0(DOP, OP, TYPE) DISPATCHER_ERROR(OP, TYPE, int32_t)
#define DISPATCHER_INT8_ENABLE_1(DOP, OP, TYPE) DISPATCHER_FUNCTION(DOP, OP, TYPE, int32_t)


#define DISPATCHER_TYPE(DOP, OP, TYPE, INT32_ENABLE, INT64_ENABLE, FLOAT_ENABLE, DOUBLE_ENABLE, \
                        POINT_ENABLE, POLYGON_ENABLE, STRING_ENABLE, INT8_ENABLE)               \
    DISPATCHER_INT32_ENABLE_##INT32_ENABLE(DOP, OP, TYPE)                                       \
        DISPATCHER_INT64_ENABLE_##INT64_ENABLE(DOP, OP, TYPE)                                   \
            DISPATCHER_FLOAT_ENABLE_##FLOAT_ENABLE(DOP, OP, TYPE)                               \
                DISPATCHER_DOUBLE_ENABLE_##DOUBLE_ENABLE(DOP, OP, TYPE)                         \
                    DISPATCHER_POINT_ENABLE_##POINT_ENABLE(DOP, OP, TYPE)                       \
                        DISPATCHER_POLYGON_ENABLE_##POLYGON_ENABLE(DOP, OP, TYPE)               \
                            DISPATCHER_STRING_ENABLE_##STRING_ENABLE(DOP, OP, TYPE)             \
                                DISPATCHER_INT8_ENABLE_##INT8_ENABLE(DOP, OP, TYPE)