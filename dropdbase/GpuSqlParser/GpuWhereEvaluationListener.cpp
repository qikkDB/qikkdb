#include "GpuWhereEvaluationListener.h"
#include "../ColumnBase.h"
#include "../QueryEngine/GPUCore/GPUFilter.cuh"
#include "../QueryEngine/GPUCore/GPULogic.cuh"
#include "../QueryEngine/GPUCore/GPUArithmetic.cuh"
#include "../QueryEngine/GPUCore/GPUArithmeticUnary.cuh"

void GpuWhereEvaluationListener::exitBinaryOperation(GpuSqlParser::BinaryOperationContext * ctx)
{
	std::tuple<int64_t, bool, DataType> right = stackTopAndPop();
	std::tuple<int64_t, bool, DataType> left = stackTopAndPop();

	std::string op = ctx->op->getText();
	stringToUpper(op);

	DataType returnDataType;
	bool ignoreFlag;

	if (op == ">")
	{
		ignoreFlag = std::get<1>(left) || std::get<1>(right);
		if (ignoreFlag)
		{
			parserStack.push(std::make_tuple(0, ignoreFlag, DataType::CONST_LONG));
		}
		else
		{
			int64_t filterResult = filterOperation<FilterConditions::greater>(std::get<0>(left), std::get<0>(right), std::get<2>(left), std::get<2>(right));
			parserStack.push(std::make_tuple(filterResult, ignoreFlag, DataType::CONST_LONG));
		}
	}
	else if (op == "<")
	{
		ignoreFlag = std::get<1>(left) || std::get<1>(right);
		if (ignoreFlag)
		{
			parserStack.push(std::make_tuple(0, ignoreFlag, DataType::CONST_LONG));
		}
		else
		{
			int64_t filterResult = filterOperation<FilterConditions::less>(std::get<0>(left), std::get<0>(right), std::get<2>(left), std::get<2>(right));
			parserStack.push(std::make_tuple(filterResult, ignoreFlag, DataType::CONST_LONG));
		}
	}
	else if (op == ">=")
	{
		ignoreFlag = std::get<1>(left) || std::get<1>(right);
		if (ignoreFlag)
		{
			parserStack.push(std::make_tuple(0, ignoreFlag, DataType::CONST_LONG));
		}
		else
		{
			int64_t filterResult = filterOperation<FilterConditions::greaterEqual>(std::get<0>(left), std::get<0>(right), std::get<2>(left), std::get<2>(right));
			parserStack.push(std::make_tuple(filterResult, ignoreFlag, DataType::CONST_LONG));
		}
	}
	else if (op == "<=")
	{
		ignoreFlag = std::get<1>(left) || std::get<1>(right);
		if (ignoreFlag)
		{
			parserStack.push(std::make_tuple(0, ignoreFlag, DataType::CONST_LONG));
		}
		else
		{
			int64_t filterResult = filterOperation<FilterConditions::lessEqual>(std::get<0>(left), std::get<0>(right), std::get<2>(left), std::get<2>(right));
			parserStack.push(std::make_tuple(filterResult, ignoreFlag, DataType::CONST_LONG));
		}
	}
	else if (op == "=")
	{
		ignoreFlag = std::get<1>(left) || std::get<1>(right);
		if (ignoreFlag)
		{
			parserStack.push(std::make_tuple(0, ignoreFlag, DataType::CONST_LONG));
		}
		else
		{
			int64_t filterResult = filterOperation<FilterConditions::equal>(std::get<0>(left), std::get<0>(right), std::get<2>(left), std::get<2>(right));
			parserStack.push(std::make_tuple(filterResult, ignoreFlag, DataType::CONST_LONG));
		}
	}
	else if (op == "!=" || op == "<>")
	{
		ignoreFlag = std::get<1>(left) || std::get<1>(right);
		if (ignoreFlag)
		{
			parserStack.push(std::make_tuple(0, ignoreFlag, DataType::CONST_LONG));
		}
		else
		{
			int64_t filterResult = filterOperation<FilterConditions::notEqual>(std::get<0>(left), std::get<0>(right), std::get<2>(left), std::get<2>(right));
			parserStack.push(std::make_tuple(filterResult, ignoreFlag, DataType::CONST_LONG));
		}
	}
	else if (op == "AND")
	{
		ignoreFlag = std::get<1>(left) || std::get<1>(right);
		if (ignoreFlag)
		{
			parserStack.push(std::make_tuple(0, ignoreFlag, DataType::CONST_LONG));
		}
		else
		{
			int64_t filterResult = filterOperation<LogicOperations::logicalAnd>(std::get<0>(left), std::get<0>(right), std::get<2>(left), std::get<2>(right));
			parserStack.push(std::make_tuple(filterResult, ignoreFlag, DataType::CONST_LONG));
		}
	}
	else if (op == "OR")
	{
		ignoreFlag = std::get<1>(left) || std::get<1>(right);
		if (ignoreFlag)
		{
			parserStack.push(std::make_tuple(0, ignoreFlag, DataType::CONST_LONG));
		}
		else
		{
			int64_t filterResult = filterOperation<LogicOperations::logicalOr>(std::get<0>(left), std::get<0>(right), std::get<2>(left), std::get<2>(right));
			parserStack.push(std::make_tuple(filterResult, ignoreFlag, DataType::CONST_LONG));
		}
	}
	else if (op == "*")
	{

	}
	else if (op == "/")
	{
	}
	else if (op == "+")
	{
	}
	else if (op == "-")
	{
	}
	else if (op == "%")
	{
	}
	else if (op == "|")
	{
	}
	else if (op == "&")
	{
	}
	else if (op == "^")
	{
	}
	else if (op == "<<")
	{
	}
	else if (op == ">>")
	{
	}
	else if (op == "POINT")
	{
		parserStack.push(std::make_tuple(0, true, DataType::CONST_LONG));
	}
	else if (op == "GEO_CONTAINS")
	{
		parserStack.push(std::make_tuple(0, true, DataType::CONST_LONG));
	}
	else if (op == "GEO_INTERSECT")
	{
		parserStack.push(std::make_tuple(0, true, DataType::CONST_LONG));
	}
	else if (op == "GEO_UNION")
	{
		parserStack.push(std::make_tuple(0, true, DataType::CONST_LONG));
	}
	else if (op == "LOG")
	{
	}
	else if (op == "POW")
	{
	}
	else if (op == "ROOT")
	{
	}
	else if (op == "ATAN2")
	{
	}
}

void GpuWhereEvaluationListener::exitIntLiteral(GpuSqlParser::IntLiteralContext * ctx)
{
	std::string token = ctx->getText();
	parserStack.push(std::make_tuple(std::stoll(token), false, DataType::CONST_LONG));
}

void GpuWhereEvaluationListener::exitDecimalLiteral(GpuSqlParser::DecimalLiteralContext * ctx)
{
	std::string token = ctx->getText();
	parserStack.push(std::make_tuple(std::stod(token), false, DataType::CONST_DOUBLE));
}

void GpuWhereEvaluationListener::exitStringLiteral(GpuSqlParser::StringLiteralContext * ctx)
{
	parserStack.push(std::make_tuple(0, true, DataType::CONST_LONG));
}

void GpuWhereEvaluationListener::exitGeoReference(GpuSqlParser::GeoReferenceContext * ctx)
{
	parserStack.push(std::make_tuple(0, true, DataType::CONST_LONG));
}

void GpuWhereEvaluationListener::exitVarReference(GpuSqlParser::VarReferenceContext * ctx)
{
	std::pair<std::string, DataType> tableColumnData = generateAndValidateColumnName(ctx->columnId());
	const DataType columnType = std::get<1>(tableColumnData);
	const std::string tableColumn = std::get<0>(tableColumnData);

	const size_t splitIndex = tableColumn.find(".");
	const std::string table = tableColumn.substr(0, splitIndex);
	const std::string column = tableColumn.substr(splitIndex + 1);

	switch (columnType)
	{
	case DataType::COLUMN_INT:
	{
		int32_t blockMin = dynamic_cast<ColumnBase<int32_t>*>(database->GetTables().at(table).GetColumns().at(column).get())->GetBlocksList()[blockIndex]->GetMin();
		parserStack.push(std::make_tuple(static_cast<int64_t>(blockMin), false, DataType::CONST_LONG));
		break;
	}
	case DataType::COLUMN_LONG:
	{
		int64_t blockMin = dynamic_cast<ColumnBase<int64_t>*>(database->GetTables().at(table).GetColumns().at(column).get())->GetBlocksList()[blockIndex]->GetMin();
		parserStack.push(std::make_tuple(blockMin, false, DataType::CONST_LONG));
		break;
	}
	case DataType::COLUMN_FLOAT:
	{
		double blockMin = static_cast<double>(dynamic_cast<ColumnBase<float>*>(database->GetTables().at(table).GetColumns().at(column).get())->GetBlocksList()[blockIndex]->GetMin());
		parserStack.push(std::make_tuple(*reinterpret_cast<int64_t*>(&blockMin), false, DataType::CONST_DOUBLE));
		break;
	}
	case DataType::COLUMN_DOUBLE:
	{
		double blockMin = dynamic_cast<ColumnBase<double>*>(database->GetTables().at(table).GetColumns().at(column).get())->GetBlocksList()[blockIndex]->GetMin();
		parserStack.push(std::make_tuple(*reinterpret_cast<int64_t*>(&blockMin), false, DataType::CONST_DOUBLE));
		break;
	}
	case DataType::COLUMN_INT8_T:
	{
		int8_t blockMin = dynamic_cast<ColumnBase<int8_t>*>(database->GetTables().at(table).GetColumns().at(column).get())->GetBlocksList()[blockIndex]->GetMin();
		parserStack.push(std::make_tuple(static_cast<int64_t>(blockMin), false, DataType::CONST_LONG));
		break;
	}
	default:
		parserStack.push(std::make_tuple(0, true, DataType::CONST_LONG));
		break;
	}
}

std::tuple<int64_t, bool, DataType> GpuWhereEvaluationListener::stackTopAndPop()
{
	std::tuple<int64_t, bool, DataType> value = parserStack.top();
	parserStack.pop();
	return value;
}