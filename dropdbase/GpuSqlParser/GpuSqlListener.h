//
// Created by Martin Staňo on 2019-01-15.
//

#ifndef DROPDBASE_INSTAREA_GPUSQLLISTENER_H
#define DROPDBASE_INSTAREA_GPUSQLLISTENER_H

#include "GpuSqlParser.h"
#include "GpuSqlParserBaseListener.h"
#include "../DataType.h"
#include "../QueryEngine/OrderByType.h"
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <memory>
#include <stack>
#include <boost/functional/hash.hpp>

class GpuSqlDispatcher;
class GpuSqlJoinDispatcher;
class Database;

class GpuSqlListener : public GpuSqlParserBaseListener
{
private:
    const std::shared_ptr<Database> &database;
    GpuSqlDispatcher &dispatcher;
	GpuSqlJoinDispatcher &joinDispatcher;
    std::stack<std::pair<std::string, DataType>> parserStack;
	std::unordered_map<std::string, std::string> tableAliases;
	std::unordered_set<std::string> columnAliases;
	std::unordered_map<std::string, GpuSqlParser::ExpressionContext*> columnAliasContexts;
    std::unordered_set<std::string> loadedTables;
	int32_t linkTableIndex;
	int32_t orderByColumnIndex;
	std::unordered_map<std::string, std::pair<DataType, std::string>> returnColumns;
	std::unordered_map<std::string, std::pair<DataType, OrderBy::Order>> orderByColumns;
    std::unordered_set<std::pair<std::string, DataType>, boost::hash<std::pair<std::string, DataType>>> groupByColumns;
	std::unordered_set<std::pair<std::string, DataType>, boost::hash<std::pair<std::string, DataType>>> originalGroupByColumns;

	bool usingLoad;
	bool usingWhere;

    bool usingGroupBy;
    bool insideAgg;
	bool insideGroupBy;
	bool insideOrderBy;

	bool insideSelectColumn;
	bool isAggSelectColumn;
	bool isSelectColumnValid;

	void pushArgument(const char *token, DataType dataType);
	std::pair<std::string, DataType> stackTopAndPop();
	void stringToUpper(std::string &str);

	void pushTempResult(std::string reg, DataType type);

	bool isLong(const std::string &value);

	bool isDouble(const std::string &value);

	bool isPoint(const std::string &value);

	bool isPolygon(const std::string &value);

	void trimDelimitedIdentifier(std::string &str);

	std::string getRegString(antlr4::ParserRuleContext* ctx);
	DataType getReturnDataType(DataType left, DataType right);
	DataType getReturnDataType(DataType operand);
	DataType getDataTypeFromString(const std::string& dataType);

	std::pair<std::string, DataType> generateAndValidateColumnName(GpuSqlParser::ColumnIdContext *ctx);

	void walkAliasExpression(const std::string& alias);


public:
	GpuSqlListener(const std::shared_ptr<Database> &database, GpuSqlDispatcher &dispatcher, GpuSqlJoinDispatcher& joinDispatcher);

	int64_t resultLimit;
    int64_t resultOffset;

    void exitBinaryOperation(GpuSqlParser::BinaryOperationContext *ctx) override;

    void exitTernaryOperation(GpuSqlParser::TernaryOperationContext *ctx) override;

    void exitUnaryOperation(GpuSqlParser::UnaryOperationContext *ctx) override;

	void exitCastOperation(GpuSqlParser::CastOperationContext *ctx) override;

    void exitIntLiteral(GpuSqlParser::IntLiteralContext *ctx) override;

    void exitDecimalLiteral(GpuSqlParser::DecimalLiteralContext *ctx) override;

    void exitStringLiteral(GpuSqlParser::StringLiteralContext *ctx) override;

    void exitBooleanLiteral(GpuSqlParser::BooleanLiteralContext *ctx) override;

    void exitGeoReference(GpuSqlParser::GeoReferenceContext *ctx) override;

    void exitVarReference(GpuSqlParser::VarReferenceContext *ctx) override;

	void exitDateTimeLiteral(GpuSqlParser::DateTimeLiteralContext *ctx) override;

	void exitPiLiteral(GpuSqlParser::PiLiteralContext *ctx) override;

	void exitNowLiteral(GpuSqlParser::NowLiteralContext *ctx) override;

	void enterAggregation(GpuSqlParser::AggregationContext *ctx) override;

    void exitAggregation(GpuSqlParser::AggregationContext *ctx) override;

    void exitSelectColumns(GpuSqlParser::SelectColumnsContext *ctx) override;

	void enterSelectColumn(GpuSqlParser::SelectColumnContext *ctx) override;

    void exitSelectColumn(GpuSqlParser::SelectColumnContext *ctx) override;

    void exitFromTables(GpuSqlParser::FromTablesContext *ctx) override;

	void exitJoinClause(GpuSqlParser::JoinClauseContext *ctx) override;

	void exitJoinClauses(GpuSqlParser::JoinClausesContext *ctx) override;

    void exitWhereClause(GpuSqlParser::WhereClauseContext *ctx) override;

    void enterWhereClause(GpuSqlParser::WhereClauseContext *ctx) override;

	void enterGroupByColumns(GpuSqlParser::GroupByColumnsContext *ctx) override;

    void exitGroupByColumns(GpuSqlParser::GroupByColumnsContext *ctx) override;

	void exitGroupByColumn(GpuSqlParser::GroupByColumnContext *ctx) override;

	void exitShowDatabases(GpuSqlParser::ShowDatabasesContext *ctx) override;

	void exitShowTables(GpuSqlParser::ShowTablesContext *ctx) override;

	void exitShowColumns(GpuSqlParser::ShowColumnsContext *ctx) override;

	void exitSqlCreateDb(GpuSqlParser::SqlCreateDbContext *ctx) override;

	void exitSqlDropDb(GpuSqlParser::SqlDropDbContext *ctx) override;

	void exitSqlCreateTable(GpuSqlParser::SqlCreateTableContext *ctx) override;

	void exitSqlDropTable(GpuSqlParser::SqlDropTableContext *ctx) override;

	void exitSqlAlterTable(GpuSqlParser::SqlAlterTableContext *ctx) override;

	void exitSqlInsertInto(GpuSqlParser::SqlInsertIntoContext *ctx) override;

	void exitSqlCreateIndex(GpuSqlParser::SqlCreateIndexContext *ctx) override;

	void enterOrderByColumns(GpuSqlParser::OrderByColumnsContext *ctx) override;

	void exitOrderByColumns(GpuSqlParser::OrderByColumnsContext *ctx) override;

	void exitOrderByColumn(GpuSqlParser::OrderByColumnContext *ctx) override;

	void exitLimit(GpuSqlParser::LimitContext *ctx) override;

	void exitOffset(GpuSqlParser::OffsetContext *ctx) override;

	bool GetUsingLoad();

	bool GetUsingWhere();

	void ExtractColumnAliasContexts(GpuSqlParser::SelectColumnsContext *ctx);
};


#endif //DROPDBASE_INSTAREA_GPUSQLLISTENER_H
