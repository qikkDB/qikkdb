#pragma once
#include "GpuSqlListener.h"
#include "CpuSqlDispatcher.h"

#include "../DataType.h"
#include "../Database.h"
#include <any>
#include <stack>

class GpuWhereEvaluationListener : public GpuSqlListener
{
private:
	const std::shared_ptr<Database> &database;
	int32_t blockIndex;
	CpuSqlDispatcher &dispatcher;
	std::stack<std::pair<std::string, DataType>> parserStack;

	template<typename OP>
	int64_t filterOperation(int64_t left, int64_t right, DataType leftDataType, DataType rightDataType) 
	{
		if (leftDataType == CONST_LONG && rightDataType == CONST_LONG)
		{
			return OP{}.template operator() <int64_t, int64_t > (left, right);
		}

		else if (leftDataType == CONST_LONG && rightDataType == CONST_DOUBLE)
		{
			return OP{}.template operator() <int64_t, double > (left, *reinterpret_cast<double*>(&right));
		}

		else if (leftDataType == CONST_DOUBLE && rightDataType == CONST_LONG)
		{
			return OP{}.template operator() <double, int64_t > (*reinterpret_cast<double*>(&left), right);
		}

		else if (leftDataType == CONST_DOUBLE && rightDataType == CONST_DOUBLE)
		{
			return OP{}.template operator() <double, double > (*reinterpret_cast<double*>(&left), *reinterpret_cast<double*>(&right));
		}
	}

	template<typename OP, typename T>
	T arithmeticOperation(int64_t left, int64_t right, DataType leftDataType, DataType rightDataType)
	{
		if (leftDataType == CONST_LONG && rightDataType == CONST_LONG)
		{
			return OP{}.template operator() <T, int64_t, int64_t> (left, right, nullptr, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
		}

		else if (leftDataType == CONST_LONG && rightDataType == CONST_DOUBLE)
		{
			return OP{}.template operator() <T, int64_t, double> (left, *reinterpret_cast<double*>(&right), nullptr, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
		}

		else if (leftDataType == CONST_DOUBLE && rightDataType == CONST_LONG)
		{
			return OP{}.template operator() <T, double, int64_t> (*reinterpret_cast<double*>(&left), right, nullptr, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
		}

		else if (leftDataType == CONST_DOUBLE && rightDataType == CONST_DOUBLE)
		{
			return OP{}.template operator() <T, double, double> (*reinterpret_cast<double*>(&left), *reinterpret_cast<double*>(&right), nullptr, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
		}
	}

public:
	void exitBinaryOperation(GpuSqlParser::BinaryOperationContext *ctx) override;

	void exitTernaryOperation(GpuSqlParser::TernaryOperationContext *ctx) override;

	void exitUnaryOperation(GpuSqlParser::UnaryOperationContext *ctx) override;

	void exitIntLiteral(GpuSqlParser::IntLiteralContext *ctx) override;

	void exitDecimalLiteral(GpuSqlParser::DecimalLiteralContext *ctx) override;

	void exitStringLiteral(GpuSqlParser::StringLiteralContext *ctx) override;

	void exitBooleanLiteral(GpuSqlParser::BooleanLiteralContext *ctx) override;

	void exitGeoReference(GpuSqlParser::GeoReferenceContext *ctx) override;

	void exitVarReference(GpuSqlParser::VarReferenceContext *ctx) override;

	void exitDateTimeLiteral(GpuSqlParser::DateTimeLiteralContext *ctx) override;

	void exitPiLiteral(GpuSqlParser::PiLiteralContext *ctx) override;

	void exitNowLiteral(GpuSqlParser::NowLiteralContext *ctx) override;
};