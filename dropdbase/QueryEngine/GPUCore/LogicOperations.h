/// Logic relation operation functors
namespace LogicOperations
{
/// A logical binary AND operation
struct logicalAnd
{
    template <typename T, typename U, typename V>
    __device__ __host__ T operator()(U a, V b)
    {
        return a && b;
    }
};

/// A logical binary OR operation
struct logicalOr
{
    template <typename T, typename U, typename V>
    __device__ __host__ T operator()(U a, V b)
    {
        return a || b;
    }
};
} // namespace LogicOperations
