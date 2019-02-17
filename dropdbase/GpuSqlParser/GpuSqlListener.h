//
// Created by Martin Staňo on 2019-01-15.
//

#ifndef DROPDBASE_INSTAREA_GPUSQLLISTENER_H
#define DROPDBASE_INSTAREA_GPUSQLLISTENER_H

#include "GpuSqlParser.h"
#include "GpuSqlParserBaseListener.h"
#include "ParserExceptions.h"
#include "../Database.h"
#include "../DataType.h"
#include "../PointFactory.h"
#include "../ComplexPolygonFactory.h"
#include <unordered_set>
#include <functional>
#include <string>
#include <memory>
#include <stack>
#include <regex>
#include <boost/functional/hash.hpp>

class GpuSqlDispatcher;

class GpuSqlListener : public GpuSqlParserBaseListener
{
private:
    const std::shared_ptr<Database> &database;
    GpuSqlDispatcher &dispatcher;
    std::stack<std::pair<std::string, DataType>> parserStack;
    std::unordered_set<std::string> loadedTables;
    std::unordered_set<std::string> loadedColumns;
    std::unordered_set<std::pair<std::string, DataType>, boost::hash<std::pair<std::string, DataType>>> groupByColumns;
	std::unordered_set<std::pair<std::string, DataType>, boost::hash<std::pair<std::string, DataType>>> originalGroupByColumns;

    bool usingGroupBy;
    bool insideAgg;
	bool insideGroupBy;

	bool insideSelectColumn;
	bool isAggSelectColumn;

    std::pair<std::string, DataType> stackTopAndPop();

    std::pair<std::string, DataType> generateAndValidateColumnName(GpuSqlParser::ColumnIdContext *ctx);

    void pushTempResult(std::string reg, DataType type);

    void pushArgument(const char *token, DataType dataType);

    bool isLong(const std::string &value);

    bool isDouble(const std::string &value);

    bool isPoint(const std::string &value);

    bool isPolygon(const std::string &value);

    void stringToUpper(std::string &str);

	std::string getRegString(antlr4::ParserRuleContext* ctx);
	DataType getReturnDataType(DataType left, DataType right);
	DataType getReturnDataType(DataType operand);

public:
	GpuSqlListener(const std::shared_ptr<Database> &database, GpuSqlDispatcher &dispatcher);

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

	void enterAggregation(GpuSqlParser::AggregationContext *ctx) override;

    void exitAggregation(GpuSqlParser::AggregationContext *ctx) override;

    void exitSelectColumns(GpuSqlParser::SelectColumnsContext *ctx) override;

	void enterSelectColumn(GpuSqlParser::SelectColumnContext *ctx) override;

    void exitSelectColumn(GpuSqlParser::SelectColumnContext *ctx) override;

    void exitFromTables(GpuSqlParser::FromTablesContext *ctx) override;

    void exitWhereClause(GpuSqlParser::WhereClauseContext *ctx) override;

	void enterGroupByColumns(GpuSqlParser::GroupByColumnsContext *ctx) override;

    void exitGroupByColumns(GpuSqlParser::GroupByColumnsContext *ctx) override;

	void exitGroupByColumn(GpuSqlParser::GroupByColumnContext *ctx) override;

	void exitShowDatabases(GpuSqlParser::ShowDatabasesContext *ctx) override;

	void exitShowTables(GpuSqlParser::ShowTablesContext *ctx) override;

	void exitShowColumns(GpuSqlParser::ShowColumnsContext *ctx) override;

	void exitSqlInsertInto(GpuSqlParser::SqlInsertIntoContext *ctx) override;
};


#endif //DROPDBASE_INSTAREA_GPUSQLLISTENER_H
