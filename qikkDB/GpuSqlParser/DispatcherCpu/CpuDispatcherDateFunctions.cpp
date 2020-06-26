#include "CpuDispatcherDateFunctions.h"
#include <iomanip>
#include <array>
#include "../../QueryEngine/GPUCore/DateOperations.h"

std::array<CpuSqlDispatcher::CpuDispatchFunction, DataType::DATA_TYPE_SIZE> CpuSqlDispatcher::dateToStringFunctions_ = {
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::toString, int32_t>,
    &CpuSqlDispatcher::DateToStringConst,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::toString, float>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::toString, double>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::toString, QikkDB::Types::Point>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::toString, QikkDB::Types::ComplexPolygon>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::toString, std::string>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::toString, int8_t>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::toString, int32_t>,
    &CpuSqlDispatcher::DateToStringCol,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::toString, float>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::toString, double>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::toString, QikkDB::Types::Point>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::toString, QikkDB::Types::ComplexPolygon>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::toString, std::string>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::toString, int8_t>};
std::array<CpuSqlDispatcher::CpuDispatchFunction, DataType::DATA_TYPE_SIZE> CpuSqlDispatcher::yearFunctions_ = {
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::year, int32_t>,
    &CpuSqlDispatcher::DateExtractConst<DateOperations::year>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::year, float>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::year, double>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::year, QikkDB::Types::Point>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::year, QikkDB::Types::ComplexPolygon>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::year, std::string>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::year, int8_t>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::year, int32_t>,
    &CpuSqlDispatcher::DateExtractCol<DateOperations::year>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::year, float>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::year, double>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::year, QikkDB::Types::Point>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::year, QikkDB::Types::ComplexPolygon>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::year, std::string>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::year, int8_t>};
std::array<CpuSqlDispatcher::CpuDispatchFunction, DataType::DATA_TYPE_SIZE> CpuSqlDispatcher::monthFunctions_ = {
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::month, int32_t>,
    &CpuSqlDispatcher::DateExtractConst<DateOperations::month>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::month, float>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::month, double>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::month, QikkDB::Types::Point>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::month, QikkDB::Types::ComplexPolygon>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::month, std::string>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::month, int8_t>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::month, int32_t>,
    &CpuSqlDispatcher::DateExtractCol<DateOperations::month>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::month, float>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::month, double>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::month, QikkDB::Types::Point>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::month, QikkDB::Types::ComplexPolygon>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::month, std::string>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::month, int8_t>};
std::array<CpuSqlDispatcher::CpuDispatchFunction, DataType::DATA_TYPE_SIZE> CpuSqlDispatcher::dayFunctions_ = {
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::day, int32_t>,
    &CpuSqlDispatcher::DateExtractConst<DateOperations::day>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::day, float>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::day, double>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::day, QikkDB::Types::Point>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::day, QikkDB::Types::ComplexPolygon>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::day, std::string>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::day, int8_t>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::day, int32_t>,
    &CpuSqlDispatcher::DateExtractCol<DateOperations::day>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::day, float>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::day, double>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::day, QikkDB::Types::Point>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::day, QikkDB::Types::ComplexPolygon>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::day, std::string>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::day, int8_t>};
std::array<CpuSqlDispatcher::CpuDispatchFunction, DataType::DATA_TYPE_SIZE> CpuSqlDispatcher::hourFunctions_ = {
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::hour, int32_t>,
    &CpuSqlDispatcher::DateExtractConst<DateOperations::hour>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::hour, float>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::hour, double>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::hour, QikkDB::Types::Point>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::hour, QikkDB::Types::ComplexPolygon>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::hour, std::string>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::hour, int8_t>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::hour, int32_t>,
    &CpuSqlDispatcher::DateExtractCol<DateOperations::hour>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::hour, float>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::hour, double>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::hour, QikkDB::Types::Point>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::hour, QikkDB::Types::ComplexPolygon>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::hour, std::string>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::hour, int8_t>};
std::array<CpuSqlDispatcher::CpuDispatchFunction, DataType::DATA_TYPE_SIZE> CpuSqlDispatcher::minuteFunctions_ = {
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::minute, int32_t>,
    &CpuSqlDispatcher::DateExtractConst<DateOperations::minute>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::minute, float>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::minute, double>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::minute, QikkDB::Types::Point>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::minute, QikkDB::Types::ComplexPolygon>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::minute, std::string>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::minute, int8_t>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::minute, int32_t>,
    &CpuSqlDispatcher::DateExtractCol<DateOperations::minute>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::minute, float>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::minute, double>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::minute, QikkDB::Types::Point>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::minute, QikkDB::Types::ComplexPolygon>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::minute, std::string>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::minute, int8_t>};
std::array<CpuSqlDispatcher::CpuDispatchFunction, DataType::DATA_TYPE_SIZE> CpuSqlDispatcher::secondFunctions_ = {
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::second, int32_t>,
    &CpuSqlDispatcher::DateExtractConst<DateOperations::second>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::second, float>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::second, double>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::second, QikkDB::Types::Point>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::second, QikkDB::Types::ComplexPolygon>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::second, std::string>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::second, int8_t>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::second, int32_t>,
    &CpuSqlDispatcher::DateExtractCol<DateOperations::second>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::second, float>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::second, double>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::second, QikkDB::Types::Point>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::second, QikkDB::Types::ComplexPolygon>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::second, std::string>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::second, int8_t>};
std::array<CpuSqlDispatcher::CpuDispatchFunction, DataType::DATA_TYPE_SIZE> CpuSqlDispatcher::weekdayFunctions_ = {
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::weekday, int32_t>,
    &CpuSqlDispatcher::DateExtractConst<DateOperations::weekday>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::weekday, float>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::weekday, double>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::weekday, QikkDB::Types::Point>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::weekday, QikkDB::Types::ComplexPolygon>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::weekday, std::string>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::weekday, int8_t>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::weekday, int32_t>,
    &CpuSqlDispatcher::DateExtractCol<DateOperations::weekday>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::weekday, float>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::weekday, double>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::weekday, QikkDB::Types::Point>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::weekday, QikkDB::Types::ComplexPolygon>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::weekday, std::string>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::weekday, int8_t>};
std::array<CpuSqlDispatcher::CpuDispatchFunction, DataType::DATA_TYPE_SIZE> CpuSqlDispatcher::dayOfWeekFunctions_ = {
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::dayOfWeek, int32_t>,
    &CpuSqlDispatcher::DateExtractConst<DateOperations::dayOfWeek>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::dayOfWeek, float>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::dayOfWeek, double>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::dayOfWeek, QikkDB::Types::Point>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::dayOfWeek, QikkDB::Types::ComplexPolygon>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::dayOfWeek, std::string>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::dayOfWeek, int8_t>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::dayOfWeek, int32_t>,
    &CpuSqlDispatcher::DateExtractCol<DateOperations::dayOfWeek>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::dayOfWeek, float>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::dayOfWeek, double>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::dayOfWeek, QikkDB::Types::Point>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::dayOfWeek, QikkDB::Types::ComplexPolygon>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::dayOfWeek, std::string>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::dayOfWeek, int8_t>};

int32_t CpuSqlDispatcher::DateToStringCol()
{
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    if (LoadCol<int64_t>(colName))
    {
        return 1;
    }

    // TODO ResultType
    std::string colPointerNameMin;
    std::string colPointerNameMax;
    std::tie(colPointerNameMin, colPointerNameMax) = GetPointerNames(colName);

    auto colValMin = allocatedPointers_.at(colPointerNameMin);
    auto colValMax = allocatedPointers_.at(colPointerNameMax);

    time_t timestampMin = reinterpret_cast<int64_t*>(std::get<0>(colValMin))[0];
    time_t timestampMax = reinterpret_cast<int64_t*>(std::get<0>(colValMax))[0];

    auto tmMin = std::gmtime(&timestampMin);
    auto tmMax = std::gmtime(&timestampMax);
    std::stringstream ssMin;
    std::stringstream ssMax;
    ssMin << std::put_time(tmMin, "%Y-%m-%d %H:%M:%S");
    ssMax << std::put_time(tmMax, "%Y-%m-%d %H:%M:%S");

    std::string resultStringMin = ssMin.str();
    std::string resultStringMax = ssMax.str();

    char* resultMin =
        AllocateRegister<char>(reg + "_min", resultStringMin.size() + 1, std::get<2>(colValMin) || true);
    char* resultMax =
        AllocateRegister<char>(reg + "_max", resultStringMax.size() + 1, std::get<2>(colValMax) || true);

    std::copy(resultStringMin.begin(), resultStringMin.end(), resultMin);
    resultMin[resultStringMin.size()] = '\0';
    std::copy(resultStringMax.begin(), resultStringMax.end(), resultMax);
    resultMax[resultStringMax.size()] = '\0';

    CudaLogBoost::getInstance(CudaLogBoost::debug)
        << "Where evaluation dateToStringCol_min: " << colName << ", " << reg + "_min"
        << ": " << resultMin[0] << '\n';
    CudaLogBoost::getInstance(CudaLogBoost::debug)
        << "Where evaluation dateToStringCol_max: " << colName << ", " << reg + "_max"
        << ": " << resultMax[0] << '\n';

    return 0;
}

int32_t CpuSqlDispatcher::DateToStringConst()
{
    auto cnst = arguments_.Read<int64_t>();
    auto reg = arguments_.Read<std::string>();

    auto tm = std::gmtime(&cnst);
    std::stringstream ss;
    ss << std::put_time(tm, "%Y-%m-%d %H:%M:%S");

    std::string resultString = ss.str();

    char* resultMin = AllocateRegister<char>(reg + "_min", resultString.size() + 1, true);
    char* resultMax = AllocateRegister<char>(reg + "_max", resultString.size() + 1, true);

    std::copy(resultString.begin(), resultString.end(), resultMin);
    resultMin[resultString.size()] = '\0';
    std::copy(resultString.begin(), resultString.end(), resultMax);
    resultMax[resultString.size()] = '\0';

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "Where evaluation dateToStringConst_min: " << reg + "_min"
                                                   << ": " << resultMin[0] << '\n';
    CudaLogBoost::getInstance(CudaLogBoost::debug) << "Where evaluation dateToStringConst_max: " << reg + "_max"
                                                   << ": " << resultMax[0] << '\n';

    return 0;
}