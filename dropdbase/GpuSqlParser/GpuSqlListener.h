//
// Created by Martin Staňo on 2019-01-15.
//

#ifndef DROPDBASE_INSTAREA_GPUSQLLISTENER_H
#define DROPDBASE_INSTAREA_GPUSQLLISTENER_H

#include "GpuSqlParser.h"
#include "GpuSqlLexer.h"
#include "GpuSqlParserBaseListener.h"
#include "../DataType.h"
#include "../QueryEngine/OrderByType.h"
#include <unordered_set>
#include <unordered_map>
#include <map>
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
    const std::shared_ptr<Database>& database_;
    GpuSqlDispatcher& dispatcher_;
    GpuSqlJoinDispatcher& joinDispatcher_;
    std::stack<std::pair<std::string, DataType>> parserStack_;
    std::unordered_map<std::string, std::string> tableAliases_;
    std::unordered_set<std::string> columnAliases_;
    std::string currentExpressionAlias_;
    std::unordered_map<std::string, GpuSqlParser::ExpressionContext*> columnAliasContexts_;
    std::unordered_map<std::string, std::string> expandedColumnAliases_;
    std::unordered_map<int64_t, GpuSqlParser::ExpressionContext*> columnNumericAliasContexts_;
    std::unordered_set<std::string> loadedTables_;
    std::unordered_map<std::string, std::string> shortColumnNames_;
    int32_t linkTableIndex_;
    int32_t orderByColumnIndex_;
    std::unordered_map<std::string, std::pair<DataType, std::string>> returnColumns_;
    std::unordered_map<std::string, std::pair<DataType, OrderBy::Order>> orderByColumns_;
    std::unordered_set<std::pair<std::string, DataType>, boost::hash<std::pair<std::string, DataType>>> groupByColumns_;
    std::unordered_set<std::pair<std::string, DataType>, boost::hash<std::pair<std::string, DataType>>> originalGroupByColumns_;

    bool usingGroupBy_;
    bool usingAgg_;
    bool insideAgg_;
    bool insideWhere_;
    bool insideGroupBy_;
    bool insideOrderBy_;
    bool insideAlias_;

    bool insideSelectColumn_;
    bool isAggSelectColumn_;
    bool isSelectColumnValid_;

    void PushArgument(const char* token, DataType dataType);
    std::pair<std::string, DataType> StackTopAndPop();
    void StringToUpper(std::string& str);

    void PushTempResult(std::string reg, DataType type);

    bool IsLong(const std::string& value);

    bool IsDouble(const std::string& value);

    bool IsPoint(const std::string& value);

    bool IsPolygon(const std::string& value);

    void TrimDelimitedIdentifier(std::string& str);

    DataType GetReturnDataType(DataType left, DataType right);
    DataType GetReturnDataType(DataType operand);
    DataType GetDataTypeFromString(const std::string& dataType);

    void TrimReg(std::string& reg);

    std::pair<std::string, DataType> GenerateAndValidateColumnName(GpuSqlParser::ColumnIdContext* ctx);

    void WalkAliasExpression(const std::string& alias);

    void WalkAliasExpression(const int64_t alias);

    time_t DateToLong(std::string dateString);


public:
    GpuSqlListener(const std::shared_ptr<Database>& database,
                   GpuSqlDispatcher& dispatcher,
                   GpuSqlJoinDispatcher& joinDispatcher);

    int64_t ResultLimit;
    int64_t ResultOffset;
    int32_t CurrentSelectColumnIndex;
    bool ContainsAggFunction;

    std::map<int32_t, std::string> ColumnOrder;
    const std::unordered_map<std::string, std::string>& GetAliasList() const;
    void exitBinaryOperation(GpuSqlParser::BinaryOperationContext* ctx) override;

    void exitTernaryOperation(GpuSqlParser::TernaryOperationContext* ctx) override;

    void exitUnaryOperation(GpuSqlParser::UnaryOperationContext* ctx) override;

    void exitCastOperation(GpuSqlParser::CastOperationContext* ctx) override;

    void exitIntLiteral(GpuSqlParser::IntLiteralContext* ctx) override;

    void exitDecimalLiteral(GpuSqlParser::DecimalLiteralContext* ctx) override;

    void exitStringLiteral(GpuSqlParser::StringLiteralContext* ctx) override;

    void exitBooleanLiteral(GpuSqlParser::BooleanLiteralContext* ctx) override;

    void exitGeoReference(GpuSqlParser::GeoReferenceContext* ctx) override;

    void exitVarReference(GpuSqlParser::VarReferenceContext* ctx) override;

    void exitDateTimeLiteral(GpuSqlParser::DateTimeLiteralContext* ctx) override;

    void exitPiLiteral(GpuSqlParser::PiLiteralContext* ctx) override;

    void exitNowLiteral(GpuSqlParser::NowLiteralContext* ctx) override;

    void enterAggregation(GpuSqlParser::AggregationContext* ctx) override;

    void exitAggregation(GpuSqlParser::AggregationContext* ctx) override;

    void exitSelectColumns(GpuSqlParser::SelectColumnsContext* ctx) override;

    void enterSelectColumn(GpuSqlParser::SelectColumnContext* ctx) override;

    void exitSelectColumn(GpuSqlParser::SelectColumnContext* ctx) override;

    void exitSelectAllColumns(GpuSqlParser::SelectAllColumnsContext* ctx) override;

    void exitFromTables(GpuSqlParser::FromTablesContext* ctx) override;

    void exitJoinClause(GpuSqlParser::JoinClauseContext* ctx) override;

    void exitJoinClauses(GpuSqlParser::JoinClausesContext* ctx) override;

    void exitWhereClause(GpuSqlParser::WhereClauseContext* ctx) override;

    void enterWhereClause(GpuSqlParser::WhereClauseContext* ctx) override;

    void enterGroupByColumns(GpuSqlParser::GroupByColumnsContext* ctx) override;

    void exitGroupByColumns(GpuSqlParser::GroupByColumnsContext* ctx) override;

    void exitGroupByColumn(GpuSqlParser::GroupByColumnContext* ctx) override;

    void exitShowDatabases(GpuSqlParser::ShowDatabasesContext* ctx) override;

    void exitShowTables(GpuSqlParser::ShowTablesContext* ctx) override;

    void exitShowColumns(GpuSqlParser::ShowColumnsContext* ctx) override;

    void exitSqlCreateDb(GpuSqlParser::SqlCreateDbContext* ctx) override;

    void exitSqlDropDb(GpuSqlParser::SqlDropDbContext* ctx) override;

    void exitSqlCreateTable(GpuSqlParser::SqlCreateTableContext* ctx) override;

    void exitSqlDropTable(GpuSqlParser::SqlDropTableContext* ctx) override;

    void exitSqlAlterTable(GpuSqlParser::SqlAlterTableContext* ctx) override;

    void exitSqlInsertInto(GpuSqlParser::SqlInsertIntoContext* ctx) override;

    void exitSqlCreateIndex(GpuSqlParser::SqlCreateIndexContext* ctx) override;

    void enterOrderByColumns(GpuSqlParser::OrderByColumnsContext* ctx) override;

    void exitOrderByColumns(GpuSqlParser::OrderByColumnsContext* ctx) override;

    void exitOrderByColumn(GpuSqlParser::OrderByColumnContext* ctx) override;

    void exitLimit(GpuSqlParser::LimitContext* ctx) override;

    void exitOffset(GpuSqlParser::OffsetContext* ctx) override;

	void LimitOffset();

    void SetContainsAggFunction(bool containsAgg);

    void ExtractColumnAliasContexts(GpuSqlParser::SelectColumnsContext* ctx);

    void LockAliasRegisters();
};


#endif // DROPDBASE_INSTAREA_GPUSQLLISTENER_H
