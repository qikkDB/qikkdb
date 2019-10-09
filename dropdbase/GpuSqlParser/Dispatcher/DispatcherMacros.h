#define BEGIN_DISPATCH_TABLE(OP) \
    std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE* DataType::DATA_TYPE_SIZE> OP = {

#define END_DISPATCH_TABLE \
    }                      \
    ;

#define ARITHMETIC(x) GpuSqlDispatcher::Arithmetic##x
#define DISPATCHER_FUN(DOP, args...) &DOP<args>
#define DISPATCHER_FUNCTION(DOP, args...)                                       \
    DISPATCHER_FUN(DOP(ConstConst), args), DISPATCHER_FUN(DOP(ConstCol), args), \
        DISPATCHER_FUN(DOP(ColConst), args), DISPATCHER_FUN(DOP(ColCol), args),

#define DISPATCHER_ERR(SUFFIX, args...) \
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandler##SUFFIX<args>

#define DISPATCHER_ERROR(args...)                                     \
    DISPATCHER_ERR(ConstConst, args), DISPATCHER_ERR(ConstCol, args), \
        DISPATCHER_ERR(ColConst, args), DISPATCHER_ERR(ColCol, args),

#define DISPATCHER_INVALID_TYPE(OP, TYPE)                                      \
    DISPATCHER_ERROR(OP, TYPE, int32_t), DISPATCHER_ERROR(OP, TYPE, int64_t),  \
        DISPATCHER_ERROR(OP, TYPE, float), DISPATCHER_ERROR(OP, TYPE, double), \
        DISPATCHER_ERROR(OP, TYPE, ColmnarDB::Types::Point),                   \
        DISPATCHER_ERROR(OP, TYPE, ColmnarDB::Types::ComplexPolygon),          \
        DISPATCHER_ERROR(OP, TYPE, std::string), DISPATCHER_ERROR(OP, TYPE, int8_t),\
