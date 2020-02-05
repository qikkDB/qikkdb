
#define BEGIN_DISPATCH_TABLE(OP) \
    std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE* DataType::DATA_TYPE_SIZE> OP = {

#define BEGIN_UNARY_DISPATCH_TABLE(OP) \
    std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> OP = {

#define END_DISPATCH_TABLE \
    }                      \
    ;
#define DISPATCH_ENTRY_SEPARATOR ,
#define DISPATCHER_FUN(DOP, ...) &DOP<__VA_ARGS__>
#define DISPATCHER_FUNCTION(DOP, ...)                                                         \
    DISPATCHER_FUN(DOP##ConstConst, __VA_ARGS__), DISPATCHER_FUN(DOP##ConstCol, __VA_ARGS__), \
        DISPATCHER_FUN(DOP##ColConst, __VA_ARGS__), DISPATCHER_FUN(DOP##ColCol, __VA_ARGS__),

#ifdef MERGED
#define DISPATCHER_FUNCTION_MERGED(DOP, OP, TYPEL, TYPER)                            \
    DISPATCHER_FUN(DOP, OP, TYPEL, TYPER), DISPATCHER_FUN(DOP, OP, TYPEL, TYPER##*), \
        DISPATCHER_FUN(DOP, OP, TYPEL##*, TYPER), DISPATCHER_FUN(DOP, OP, TYPEL##*, TYPER##*),
#else
#define DISPATCHER_FUNCTION_MERGED(DOP, OP, TYPEL, TYPER)                            \
    DISPATCHER_FUN(DOP##ConstConst, OP, TYPEL, TYPER), DISPATCHER_FUN(DOP##ConstCol, OP, TYPEL, TYPER), \
        DISPATCHER_FUN(DOP##ColConst, OP, TYPEL, TYPER), DISPATCHER_FUN(DOP##ColCol, OP, TYPEL, TYPER),
#endif

#define DISPATCHER_FUNCTION_NOCONST(DOP, ...)                                            \
    DISPATCHER_ERR(ConstConst, __VA_ARGS__), DISPATCHER_FUN(DOP##ConstCol, __VA_ARGS__), \
        DISPATCHER_FUN(DOP##ColConst, __VA_ARGS__), DISPATCHER_FUN(DOP##ColCol, __VA_ARGS__),

#define DISPATCHER_GROUPBY_FUNCTION(DOP, OP, RET_TYPE, ...)                                 \
    DISPATCHER_ERR(ConstConst, OP, __VA_ARGS__), DISPATCHER_ERR(ConstCol, OP, __VA_ARGS__), \
        DISPATCHER_ERR(ColConst, OP, __VA_ARGS__), DISPATCHER_FUN(DOP, OP, RET_TYPE, __VA_ARGS__),

#define DISPATCHER_UNARY_FUNCTION(DOP, ...) \
    DISPATCHER_FUN(DOP##Const, __VA_ARGS__), DISPATCHER_FUN(DOP##Col, __VA_ARGS__),

#define DISPATCHER_UNARY_FUNCTION_NO_COL(DOP, ...) \
    DISPATCHER_FUN(DOP, __VA_ARGS__), DISPATCHER_FUN(DOP, __VA_ARGS__),

#define DISPATCHER_UNARY_FUNCTION_NO_TEMPLATE(DOP) &DOP##Const, &DOP##Col,

#define DISPATCHER_ERR(SUFFIX, ...) \
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandler##SUFFIX<__VA_ARGS__>

#define DISPATCHER_ERROR(...)                                                       \
    DISPATCHER_ERR(ConstConst, __VA_ARGS__), DISPATCHER_ERR(ConstCol, __VA_ARGS__), \
        DISPATCHER_ERR(ColConst, __VA_ARGS__), DISPATCHER_ERR(ColCol, __VA_ARGS__),

#define DISPATCHER_UNARY_ERROR(...) \
    DISPATCHER_ERR(Const, __VA_ARGS__), DISPATCHER_ERR(Col, __VA_ARGS__),

#define DISPATCHER_INVALID_TYPE(OP, TYPE)                        \
    DISPATCHER_ERROR(OP, TYPE, int32_t)                          \
    DISPATCHER_ERROR(OP, TYPE, int64_t)                          \
    DISPATCHER_ERROR(OP, TYPE, float)                            \
    DISPATCHER_ERROR(OP, TYPE, double)                           \
    DISPATCHER_ERROR(OP, TYPE, ColmnarDB::Types::Point)          \
    DISPATCHER_ERROR(OP, TYPE, ColmnarDB::Types::ComplexPolygon) \
    DISPATCHER_ERROR(OP, TYPE, std::string)                      \
    DISPATCHER_ERROR(OP, TYPE, int8_t)

#define DISPATCHER_INVALID_TYPE_NOOP(TYPE)                   \
    DISPATCHER_ERROR(TYPE, int32_t)                          \
    DISPATCHER_ERROR(TYPE, int64_t)                          \
    DISPATCHER_ERROR(TYPE, float)                            \
    DISPATCHER_ERROR(TYPE, double)                           \
    DISPATCHER_ERROR(TYPE, ColmnarDB::Types::Point)          \
    DISPATCHER_ERROR(TYPE, ColmnarDB::Types::ComplexPolygon) \
    DISPATCHER_ERROR(TYPE, std::string)                      \
    DISPATCHER_ERROR(TYPE, int8_t)

#define DISPATCHER_INT32_ENABLE_0(DOP, OP, TYPE) DISPATCHER_ERROR(OP, TYPE, int32_t)
#define DISPATCHER_INT32_ENABLE_1(DOP, OP, TYPE) DISPATCHER_FUNCTION_MERGED(DOP, OP, TYPE, int32_t)

#define DISPATCHER_INT64_ENABLE_0(DOP, OP, TYPE) DISPATCHER_ERROR(OP, TYPE, int64_t)
#define DISPATCHER_INT64_ENABLE_1(DOP, OP, TYPE) DISPATCHER_FUNCTION_MERGED(DOP, OP, TYPE, int64_t)

#define DISPATCHER_FLOAT_ENABLE_0(DOP, OP, TYPE) DISPATCHER_ERROR(OP, TYPE, float)
#define DISPATCHER_FLOAT_ENABLE_1(DOP, OP, TYPE) DISPATCHER_FUNCTION_MERGED(DOP, OP, TYPE, float)

#define DISPATCHER_DOUBLE_ENABLE_0(DOP, OP, TYPE) DISPATCHER_ERROR(OP, TYPE, double)
#define DISPATCHER_DOUBLE_ENABLE_1(DOP, OP, TYPE) DISPATCHER_FUNCTION_MERGED(DOP, OP, TYPE, double)

#define DISPATCHER_POINT_ENABLE_0(DOP, OP, TYPE) DISPATCHER_ERROR(OP, TYPE, ColmnarDB::Types::Point)
#define DISPATCHER_POINT_ENABLE_1(DOP, OP, TYPE) \
    DISPATCHER_FUNCTION_MERGED(DOP, OP, TYPE, ColmnarDB::Types::Point)

#define DISPATCHER_POLYGON_ENABLE_0(DOP, OP, TYPE) \
    DISPATCHER_ERROR(OP, TYPE, ColmnarDB::Types::ComplexPolygon)
#define DISPATCHER_POLYGON_ENABLE_1(DOP, OP, TYPE) \
    DISPATCHER_FUNCTION_MERGED(DOP, OP, TYPE, ColmnarDB::Types::ComplexPolygon)

#define DISPATCHER_STRING_ENABLE_0(DOP, OP, TYPE) DISPATCHER_ERROR(OP, TYPE, std::string)
#define DISPATCHER_STRING_ENABLE_1(DOP, OP, TYPE) \
    DISPATCHER_FUNCTION_MERGED(DOP, OP, TYPE, std::string)

#define DISPATCHER_INT8_ENABLE_0(DOP, OP, TYPE) DISPATCHER_ERROR(OP, TYPE, int8_t)
#define DISPATCHER_INT8_ENABLE_1(DOP, OP, TYPE) DISPATCHER_FUNCTION_MERGED(DOP, OP, TYPE, int8_t)


#define DISPATCHER_INT32_NOLEFT_ENABLE_0(DOP, OP) DISPATCHER_ERROR(OP, int32_t)
#define DISPATCHER_INT32_NOLEFT_ENABLE_1(DOP, OP) DISPATCHER_FUNCTION(DOP, OP, int32_t)

#define DISPATCHER_INT64_NOLEFT_ENABLE_0(DOP, OP) DISPATCHER_ERROR(OP, int64_t)
#define DISPATCHER_INT64_NOLEFT_ENABLE_1(DOP, OP) DISPATCHER_FUNCTION(DOP, OP, int64_t)

#define DISPATCHER_FLOAT_NOLEFT_ENABLE_0(DOP, OP) DISPATCHER_ERROR(OP, float)
#define DISPATCHER_FLOAT_NOLEFT_ENABLE_1(DOP, OP) DISPATCHER_FUNCTION(DOP, OP, float)

#define DISPATCHER_DOUBLE_NOLEFT_ENABLE_0(DOP, OP) DISPATCHER_ERROR(OP, double)
#define DISPATCHER_DOUBLE_NOLEFT_ENABLE_1(DOP, OP) DISPATCHER_FUNCTION(DOP, OP, double)

#define DISPATCHER_POINT_NOLEFT_ENABLE_0(DOP, OP) DISPATCHER_ERROR(OP, ColmnarDB::Types::Point)
#define DISPATCHER_POINT_NOLEFT_ENABLE_1(DOP, OP) \
    DISPATCHER_FUNCTION(DOP, OP, ColmnarDB::Types::Point)

#define DISPATCHER_POLYGON_NOLEFT_ENABLE_0(DOP, OP) \
    DISPATCHER_ERROR(OP, ColmnarDB::Types::ComplexPolygon)
#define DISPATCHER_POLYGON_NOLEFT_ENABLE_1(DOP, OP) \
    DISPATCHER_FUNCTION(DOP, OP, ColmnarDB::Types::ComplexPolygon)

#define DISPATCHER_STRING_NOLEFT_ENABLE_0(DOP, OP) DISPATCHER_ERROR(OP, std::string)
#define DISPATCHER_STRING_NOLEFT_ENABLE_1(DOP, OP) DISPATCHER_FUNCTION(DOP, OP, std::string)

#define DISPATCHER_INT8_NOLEFT_ENABLE_0(DOP, OP) DISPATCHER_ERROR(OP, int8_t)
#define DISPATCHER_INT8_NOLEFT_ENABLE_1(DOP, OP) DISPATCHER_FUNCTION(DOP, OP, int8_t)


#define DISPATCHER_INT32_NOOP_ENABLE_0(DOP, TYPE) DISPATCHER_ERROR(TYPE, int32_t)
#define DISPATCHER_INT32_NOOP_ENABLE_1(DOP, TYPE) DISPATCHER_FUNCTION(DOP, TYPE, int32_t)

#define DISPATCHER_INT64_NOOP_ENABLE_0(DOP, TYPE) DISPATCHER_ERROR(TYPE, int64_t)
#define DISPATCHER_INT64_NOOP_ENABLE_1(DOP, TYPE) DISPATCHER_FUNCTION(DOP, TYPE, int64_t)

#define DISPATCHER_FLOAT_NOOP_ENABLE_0(DOP, TYPE) DISPATCHER_ERROR(TYPE, float)
#define DISPATCHER_FLOAT_NOOP_ENABLE_1(DOP, TYPE) DISPATCHER_FUNCTION(DOP, TYPE, float)

#define DISPATCHER_DOUBLE_NOOP_ENABLE_0(DOP, TYPE) DISPATCHER_ERROR(TYPE, double)
#define DISPATCHER_DOUBLE_NOOP_ENABLE_1(DOP, TYPE) DISPATCHER_FUNCTION(DOP, TYPE, double)

#define DISPATCHER_POINT_NOOP_ENABLE_0(DOP, TYPE) DISPATCHER_ERROR(TYPE, ColmnarDB::Types::Point)
#define DISPATCHER_POINT_NOOP_ENABLE_1(DOP, TYPE) \
    DISPATCHER_FUNCTION(DOP, TYPE, ColmnarDB::Types::Point)

#define DISPATCHER_POLYGON_NOOP_ENABLE_0(DOP, TYPE) \
    DISPATCHER_ERROR(TYPE, ColmnarDB::Types::ComplexPolygon)
#define DISPATCHER_POLYGON_NOOP_ENABLE_1(DOP, TYPE) \
    DISPATCHER_FUNCTION(DOP, TYPE, ColmnarDB::Types::ComplexPolygon)

#define DISPATCHER_STRING_NOOP_ENABLE_0(DOP, TYPE) DISPATCHER_ERROR(TYPE, std::string)
#define DISPATCHER_STRING_NOOP_ENABLE_1(DOP, TYPE) DISPATCHER_FUNCTION(DOP, TYPE, std::string)

#define DISPATCHER_INT8_NOOP_ENABLE_0(DOP, TYPE) DISPATCHER_ERROR(TYPE, int8_t)
#define DISPATCHER_INT8_NOOP_ENABLE_1(DOP, TYPE) DISPATCHER_FUNCTION(DOP, TYPE, int8_t)


#define DISPATCHER_INT32_NOCONST_NOOP_ENABLE_0(DOP, TYPE) DISPATCHER_ERROR(TYPE, int32_t)
#define DISPATCHER_INT32_NOCONST_NOOP_ENABLE_1(DOP, TYPE) \
    DISPATCHER_FUNCTION_NOCONST(DOP, TYPE, int32_t)

#define DISPATCHER_INT64_NOCONST_NOOP_ENABLE_0(DOP, TYPE) DISPATCHER_ERROR(TYPE, int64_t)
#define DISPATCHER_INT64_NOCONST_NOOP_ENABLE_1(DOP, TYPE) \
    DISPATCHER_FUNCTION_NOCONST(DOP, TYPE, int64_t)

#define DISPATCHER_FLOAT_NOCONST_NOOP_ENABLE_0(DOP, TYPE) DISPATCHER_ERROR(TYPE, float)
#define DISPATCHER_FLOAT_NOCONST_NOOP_ENABLE_1(DOP, TYPE) \
    DISPATCHER_FUNCTION_NOCONST(DOP, TYPE, float)

#define DISPATCHER_DOUBLE_NOCONST_NOOP_ENABLE_0(DOP, TYPE) DISPATCHER_ERROR(TYPE, double)
#define DISPATCHER_DOUBLE_NOCONST_NOOP_ENABLE_1(DOP, TYPE) \
    DISPATCHER_FUNCTION_NOCONST(DOP, TYPE, double)

#define DISPATCHER_POINT_NOCONST_NOOP_ENABLE_0(DOP, TYPE) \
    DISPATCHER_ERROR(TYPE, ColmnarDB::Types::Point)
#define DISPATCHER_POINT_NOCONST_NOOP_ENABLE_1(DOP, TYPE) \
    DISPATCHER_FUNCTION_NOCONST(DOP, TYPE, ColmnarDB::Types::Point)

#define DISPATCHER_POLYGON_NOCONST_NOOP_ENABLE_0(DOP, TYPE) \
    DISPATCHER_ERROR(TYPE, ColmnarDB::Types::ComplexPolygon)
#define DISPATCHER_POLYGON_NOCONST_NOOP_ENABLE_1(DOP, TYPE) \
    DISPATCHER_FUNCTION_NOCONST(DOP, TYPE, ColmnarDB::Types::ComplexPolygon)

#define DISPATCHER_STRING_NOCONST_NOOP_ENABLE_0(DOP, TYPE) DISPATCHER_ERROR(TYPE, std::string)
#define DISPATCHER_STRING_NOCONST_NOOP_ENABLE_1(DOP, TYPE) \
    DISPATCHER_FUNCTION_NOCONST(DOP, TYPE, std::string)

#define DISPATCHER_INT8_NOCONST_NOOP_ENABLE_0(DOP, TYPE) DISPATCHER_ERROR(TYPE, int8_t)
#define DISPATCHER_INT8_NOCONST_NOOP_ENABLE_1(DOP, TYPE) \
    DISPATCHER_FUNCTION_NOCONST(DOP, TYPE, int8_t)


#define DISPATCHER_INT32_GROUPBY_ENABLE_0(DOP, OP, TYPE) DISPATCHER_ERROR(OP, TYPE, int32_t)
#define DISPATCHER_INT32_GROUPBY_ENABLE_1(DOP, OP, TYPE) \
    DISPATCHER_GROUPBY_FUNCTION(DOP, OP, int32_t, TYPE, int32_t)

#define DISPATCHER_INT64_GROUPBY_ENABLE_0(DOP, OP, TYPE) DISPATCHER_ERROR(OP, TYPE, int64_t)
#define DISPATCHER_INT64_GROUPBY_ENABLE_1(DOP, OP, TYPE) \
    DISPATCHER_GROUPBY_FUNCTION(DOP, OP, int64_t, TYPE, int64_t)

#define DISPATCHER_FLOAT_GROUPBY_ENABLE_0(DOP, OP, TYPE) DISPATCHER_ERROR(OP, TYPE, float)
#define DISPATCHER_FLOAT_GROUPBY_ENABLE_1(DOP, OP, TYPE) \
    DISPATCHER_GROUPBY_FUNCTION(DOP, OP, float, TYPE, float)

#define DISPATCHER_DOUBLE_GROUPBY_ENABLE_0(DOP, OP, TYPE) DISPATCHER_ERROR(OP, TYPE, double)
#define DISPATCHER_DOUBLE_GROUPBY_ENABLE_1(DOP, OP, TYPE) \
    DISPATCHER_GROUPBY_FUNCTION(DOP, OP, double, TYPE, double)

#define DISPATCHER_POINT_GROUPBY_ENABLE_0(DOP, OP, TYPE) \
    DISPATCHER_ERROR(OP, TYPE, ColmnarDB::Types::Point)
#define DISPATCHER_POINT_GROUPBY_ENABLE_1(DOP, OP, TYPE) \
    DISPATCHER_GROUPBY_FUNCTION(DOP, OP, ColmnarDB::Types::Point, TYPE, ColmnarDB::Types::Point)

#define DISPATCHER_POLYGON_GROUPBY_ENABLE_0(DOP, OP, TYPE) \
    DISPATCHER_ERROR(OP, TYPE, ColmnarDB::Types::ComplexPolygon)
#define DISPATCHER_POLYGON_GROUPBY_ENABLE_1(DOP, OP, TYPE) \
    DISPATCHER_GROUPBY_FUNCTION(DOP, OP, ColmnarDB::Types::ComplexPolygon, TYPE, ColmnarDB::Types::ComplexPolygon)

#define DISPATCHER_STRING_GROUPBY_ENABLE_0(DOP, OP, TYPE) DISPATCHER_ERROR(OP, TYPE, std::string)
#define DISPATCHER_STRING_GROUPBY_ENABLE_1(DOP, OP, TYPE) \
    DISPATCHER_GROUPBY_FUNCTION(DOP, OP, std::string, TYPE, std::string)

#define DISPATCHER_INT8_GROUPBY_ENABLE_0(DOP, OP, TYPE) DISPATCHER_ERROR(OP, TYPE, int8_t)
#define DISPATCHER_INT8_GROUPBY_ENABLE_1(DOP, OP, TYPE) \
    DISPATCHER_GROUPBY_FUNCTION(DOP, OP, int8_t, TYPE, int8_t)


#define DISPATCHER_INT32_GROUPBYRET_ENABLE_0(DOP, OP, RET, TYPE) DISPATCHER_ERROR(OP, TYPE, int32_t)
#define DISPATCHER_INT32_GROUPBYRET_ENABLE_1(DOP, OP, RET, TYPE) \
    DISPATCHER_GROUPBY_FUNCTION(DOP, OP, RET, TYPE, int32_t)

#define DISPATCHER_INT64_GROUPBYRET_ENABLE_0(DOP, OP, RET, TYPE) DISPATCHER_ERROR(OP, TYPE, int64_t)
#define DISPATCHER_INT64_GROUPBYRET_ENABLE_1(DOP, OP, RET, TYPE) \
    DISPATCHER_GROUPBY_FUNCTION(DOP, OP, RET, TYPE, int64_t)

#define DISPATCHER_FLOAT_GROUPBYRET_ENABLE_0(DOP, OP, RET, TYPE) DISPATCHER_ERROR(OP, TYPE, float)
#define DISPATCHER_FLOAT_GROUPBYRET_ENABLE_1(DOP, OP, RET, TYPE) \
    DISPATCHER_GROUPBY_FUNCTION(DOP, OP, RET, TYPE, float)

#define DISPATCHER_DOUBLE_GROUPBYRET_ENABLE_0(DOP, OP, RET, TYPE) DISPATCHER_ERROR(OP, TYPE, double)
#define DISPATCHER_DOUBLE_GROUPBYRET_ENABLE_1(DOP, OP, RET, TYPE) \
    DISPATCHER_GROUPBY_FUNCTION(DOP, OP, RET, TYPE, double)

#define DISPATCHER_POINT_GROUPBYRET_ENABLE_0(DOP, OP, RET, TYPE) \
    DISPATCHER_ERROR(OP, TYPE, ColmnarDB::Types::Point)
#define DISPATCHER_POINT_GROUPBYRET_ENABLE_1(DOP, OP, RET, TYPE) \
    DISPATCHER_GROUPBY_FUNCTION(DOP, OP, RET, TYPE, ColmnarDB::Types::Point)

#define DISPATCHER_POLYGON_GROUPBYRET_ENABLE_0(DOP, OP, RET, TYPE) \
    DISPATCHER_ERROR(OP, TYPE, ColmnarDB::Types::ComplexPolygon)
#define DISPATCHER_POLYGON_GROUPBYRET_ENABLE_1(DOP, OP, RET, TYPE) \
    DISPATCHER_GROUPBY_FUNCTION(DOP, OP, RET, TYPE, ColmnarDB::Types::ComplexPolygon)

#define DISPATCHER_STRING_GROUPBYRET_ENABLE_0(DOP, OP, RET, TYPE) \
    DISPATCHER_ERROR(OP, TYPE, std::string)
#define DISPATCHER_STRING_GROUPBYRET_ENABLE_1(DOP, OP, RET, TYPE) \
    DISPATCHER_GROUPBY_FUNCTION(DOP, OP, RET, TYPE, std::string)

#define DISPATCHER_INT8_GROUPBYRET_ENABLE_0(DOP, OP, RET, TYPE) DISPATCHER_ERROR(OP, TYPE, int8_t)
#define DISPATCHER_INT8_GROUPBYRET_ENABLE_1(DOP, OP, RET, TYPE) \
    DISPATCHER_GROUPBY_FUNCTION(DOP, OP, RET, TYPE, int8_t)


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

#define DISPATCHER_TYPE_SPECIFIC(DOP, OP, INT32_ENABLE, INT64_ENABLE, FLOAT_ENABLE, DOUBLE_ENABLE, \
                                 POINT_ENABLE, POLYGON_ENABLE, STRING_ENABLE, INT8_ENABLE)         \
    DISPATCHER_INT32_NOLEFT_ENABLE_##INT32_ENABLE(DOP, OP)                                         \
        DISPATCHER_INT64_NOLEFT_ENABLE_##INT64_ENABLE(DOP, OP)                                     \
            DISPATCHER_FLOAT_NOLEFT_ENABLE_##FLOAT_ENABLE(DOP, OP)                                 \
                DISPATCHER_DOUBLE_NOLEFT_ENABLE_##DOUBLE_ENABLE(DOP, OP)                           \
                    DISPATCHER_POINT_NOLEFT_ENABLE_##POINT_ENABLE(DOP, OP)                         \
                        DISPATCHER_POLYGON_NOLEFT_ENABLE_##POLYGON_ENABLE(DOP, OP)                 \
                            DISPATCHER_STRING_NOLEFT_ENABLE_##STRING_ENABLE(DOP, OP)               \
                                DISPATCHER_INT8_NOLEFT_ENABLE_##INT8_ENABLE(DOP, OP)

#define DISPATCHER_TYPE_NOOP(DOP, TYPE, INT32_ENABLE, INT64_ENABLE, FLOAT_ENABLE, DOUBLE_ENABLE, \
                             POINT_ENABLE, POLYGON_ENABLE, STRING_ENABLE, INT8_ENABLE)           \
    DISPATCHER_INT32_NOOP_ENABLE_##INT32_ENABLE(DOP, TYPE)                                       \
        DISPATCHER_INT64_NOOP_ENABLE_##INT64_ENABLE(DOP, TYPE)                                   \
            DISPATCHER_FLOAT_NOOP_ENABLE_##FLOAT_ENABLE(DOP, TYPE)                               \
                DISPATCHER_DOUBLE_NOOP_ENABLE_##DOUBLE_ENABLE(DOP, TYPE)                         \
                    DISPATCHER_POINT_NOOP_ENABLE_##POINT_ENABLE(DOP, TYPE)                       \
                        DISPATCHER_POLYGON_NOOP_ENABLE_##POLYGON_ENABLE(DOP, TYPE)               \
                            DISPATCHER_STRING_NOOP_ENABLE_##STRING_ENABLE(DOP, TYPE)             \
                                DISPATCHER_INT8_NOOP_ENABLE_##INT8_ENABLE(DOP, TYPE)
#define DISPATCHER_TYPE_NOOP_NOCONST(DOP, TYPE, INT32_ENABLE, INT64_ENABLE, FLOAT_ENABLE, DOUBLE_ENABLE, \
                                     POINT_ENABLE, POLYGON_ENABLE, STRING_ENABLE, INT8_ENABLE)           \
    DISPATCHER_INT32_NOCONST_NOOP_ENABLE_##INT32_ENABLE(DOP, TYPE)                                       \
        DISPATCHER_INT64_NOCONST_NOOP_ENABLE_##INT64_ENABLE(DOP, TYPE)                                   \
            DISPATCHER_FLOAT_NOCONST_NOOP_ENABLE_##FLOAT_ENABLE(DOP, TYPE)                               \
                DISPATCHER_DOUBLE_NOCONST_NOOP_ENABLE_##DOUBLE_ENABLE(DOP, TYPE)                         \
                    DISPATCHER_POINT_NOCONST_NOOP_ENABLE_##POINT_ENABLE(DOP, TYPE)                       \
                        DISPATCHER_POLYGON_NOCONST_NOOP_ENABLE_##POLYGON_ENABLE(DOP, TYPE)               \
                            DISPATCHER_STRING_NOCONST_NOOP_ENABLE_##STRING_ENABLE(DOP, TYPE)             \
                                DISPATCHER_INT8_NOCONST_NOOP_ENABLE_##INT8_ENABLE(DOP, TYPE)

#define DISPATCHER_GROUPBY_TYPE(DOP, OP, TYPE, INT32_ENABLE, INT64_ENABLE, FLOAT_ENABLE, DOUBLE_ENABLE, \
                                POINT_ENABLE, POLYGON_ENABLE, STRING_ENABLE, INT8_ENABLE)               \
    DISPATCHER_INT32_GROUPBY_ENABLE_##INT32_ENABLE(DOP, OP, TYPE)                                       \
        DISPATCHER_INT64_GROUPBY_ENABLE_##INT64_ENABLE(DOP, OP, TYPE)                                   \
            DISPATCHER_FLOAT_GROUPBY_ENABLE_##FLOAT_ENABLE(DOP, OP, TYPE)                               \
                DISPATCHER_DOUBLE_GROUPBY_ENABLE_##DOUBLE_ENABLE(DOP, OP, TYPE)                         \
                    DISPATCHER_POINT_GROUPBY_ENABLE_##POINT_ENABLE(DOP, OP, TYPE)                       \
                        DISPATCHER_POLYGON_GROUPBY_ENABLE_##POLYGON_ENABLE(DOP, OP, TYPE)               \
                            DISPATCHER_STRING_GROUPBY_ENABLE_##STRING_ENABLE(DOP, OP, TYPE)             \
                                DISPATCHER_INT8_GROUPBY_ENABLE_##INT8_ENABLE(DOP, OP, TYPE)

#define DISPATCHER_GROUPBY_TYPE_WITH_RET(DOP, OP, RET, TYPE, INT32_ENABLE, INT64_ENABLE, FLOAT_ENABLE,            \
                                         DOUBLE_ENABLE, POINT_ENABLE, POLYGON_ENABLE, STRING_ENABLE, INT8_ENABLE) \
    DISPATCHER_INT32_GROUPBYRET_ENABLE_##INT32_ENABLE(DOP, OP, RET, TYPE)                                         \
        DISPATCHER_INT64_GROUPBYRET_ENABLE_##INT64_ENABLE(DOP, OP, RET, TYPE)                                     \
            DISPATCHER_FLOAT_GROUPBYRET_ENABLE_##FLOAT_ENABLE(DOP, OP, RET, TYPE)                                 \
                DISPATCHER_DOUBLE_GROUPBYRET_ENABLE_##DOUBLE_ENABLE(DOP, OP, RET, TYPE)                           \
                    DISPATCHER_POINT_GROUPBYRET_ENABLE_##POINT_ENABLE(DOP, OP, RET, TYPE)                         \
                        DISPATCHER_POLYGON_GROUPBYRET_ENABLE_##POLYGON_ENABLE(DOP, OP, RET, TYPE)                 \
                            DISPATCHER_STRING_GROUPBYRET_ENABLE_##STRING_ENABLE(DOP, OP, RET, TYPE)               \
                                DISPATCHER_INT8_GROUPBYRET_ENABLE_##INT8_ENABLE(DOP, OP, RET, TYPE)
