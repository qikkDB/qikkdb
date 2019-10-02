
// Generated from C:/GPU-DB/dropdbase/GpuSqlParser\GpuSqlParser.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"
#include "GpuSqlParser.h"


/**
 * This interface defines an abstract listener for a parse tree produced by GpuSqlParser.
 */
class  GpuSqlParserListener : public antlr4::tree::ParseTreeListener {
public:

  virtual void enterSqlFile(GpuSqlParser::SqlFileContext *ctx) = 0;
  virtual void exitSqlFile(GpuSqlParser::SqlFileContext *ctx) = 0;

  virtual void enterStatement(GpuSqlParser::StatementContext *ctx) = 0;
  virtual void exitStatement(GpuSqlParser::StatementContext *ctx) = 0;

  virtual void enterShowStatement(GpuSqlParser::ShowStatementContext *ctx) = 0;
  virtual void exitShowStatement(GpuSqlParser::ShowStatementContext *ctx) = 0;

  virtual void enterShowDatabases(GpuSqlParser::ShowDatabasesContext *ctx) = 0;
  virtual void exitShowDatabases(GpuSqlParser::ShowDatabasesContext *ctx) = 0;

  virtual void enterShowTables(GpuSqlParser::ShowTablesContext *ctx) = 0;
  virtual void exitShowTables(GpuSqlParser::ShowTablesContext *ctx) = 0;

  virtual void enterShowColumns(GpuSqlParser::ShowColumnsContext *ctx) = 0;
  virtual void exitShowColumns(GpuSqlParser::ShowColumnsContext *ctx) = 0;

  virtual void enterSqlSelect(GpuSqlParser::SqlSelectContext *ctx) = 0;
  virtual void exitSqlSelect(GpuSqlParser::SqlSelectContext *ctx) = 0;

  virtual void enterSqlCreateDb(GpuSqlParser::SqlCreateDbContext *ctx) = 0;
  virtual void exitSqlCreateDb(GpuSqlParser::SqlCreateDbContext *ctx) = 0;

  virtual void enterSqlDropDb(GpuSqlParser::SqlDropDbContext *ctx) = 0;
  virtual void exitSqlDropDb(GpuSqlParser::SqlDropDbContext *ctx) = 0;

  virtual void enterSqlCreateTable(GpuSqlParser::SqlCreateTableContext *ctx) = 0;
  virtual void exitSqlCreateTable(GpuSqlParser::SqlCreateTableContext *ctx) = 0;

  virtual void enterSqlDropTable(GpuSqlParser::SqlDropTableContext *ctx) = 0;
  virtual void exitSqlDropTable(GpuSqlParser::SqlDropTableContext *ctx) = 0;

  virtual void enterSqlAlterTable(GpuSqlParser::SqlAlterTableContext *ctx) = 0;
  virtual void exitSqlAlterTable(GpuSqlParser::SqlAlterTableContext *ctx) = 0;

  virtual void enterSqlCreateIndex(GpuSqlParser::SqlCreateIndexContext *ctx) = 0;
  virtual void exitSqlCreateIndex(GpuSqlParser::SqlCreateIndexContext *ctx) = 0;

  virtual void enterSqlInsertInto(GpuSqlParser::SqlInsertIntoContext *ctx) = 0;
  virtual void exitSqlInsertInto(GpuSqlParser::SqlInsertIntoContext *ctx) = 0;

  virtual void enterNewTableEntries(GpuSqlParser::NewTableEntriesContext *ctx) = 0;
  virtual void exitNewTableEntries(GpuSqlParser::NewTableEntriesContext *ctx) = 0;

  virtual void enterNewTableEntry(GpuSqlParser::NewTableEntryContext *ctx) = 0;
  virtual void exitNewTableEntry(GpuSqlParser::NewTableEntryContext *ctx) = 0;

  virtual void enterAlterTableEntries(GpuSqlParser::AlterTableEntriesContext *ctx) = 0;
  virtual void exitAlterTableEntries(GpuSqlParser::AlterTableEntriesContext *ctx) = 0;

  virtual void enterAlterTableEntry(GpuSqlParser::AlterTableEntryContext *ctx) = 0;
  virtual void exitAlterTableEntry(GpuSqlParser::AlterTableEntryContext *ctx) = 0;

  virtual void enterAddColumn(GpuSqlParser::AddColumnContext *ctx) = 0;
  virtual void exitAddColumn(GpuSqlParser::AddColumnContext *ctx) = 0;

  virtual void enterDropColumn(GpuSqlParser::DropColumnContext *ctx) = 0;
  virtual void exitDropColumn(GpuSqlParser::DropColumnContext *ctx) = 0;

  virtual void enterAlterColumn(GpuSqlParser::AlterColumnContext *ctx) = 0;
  virtual void exitAlterColumn(GpuSqlParser::AlterColumnContext *ctx) = 0;

  virtual void enterRenameColumn(GpuSqlParser::RenameColumnContext *ctx) = 0;
  virtual void exitRenameColumn(GpuSqlParser::RenameColumnContext *ctx) = 0;

  virtual void enterRenameTable(GpuSqlParser::RenameTableContext *ctx) = 0;
  virtual void exitRenameTable(GpuSqlParser::RenameTableContext *ctx) = 0;

  virtual void enterRenameColumnFrom(GpuSqlParser::RenameColumnFromContext *ctx) = 0;
  virtual void exitRenameColumnFrom(GpuSqlParser::RenameColumnFromContext *ctx) = 0;

  virtual void enterRenameColumnTo(GpuSqlParser::RenameColumnToContext *ctx) = 0;
  virtual void exitRenameColumnTo(GpuSqlParser::RenameColumnToContext *ctx) = 0;

  virtual void enterNewTableColumn(GpuSqlParser::NewTableColumnContext *ctx) = 0;
  virtual void exitNewTableColumn(GpuSqlParser::NewTableColumnContext *ctx) = 0;

  virtual void enterNewTableIndex(GpuSqlParser::NewTableIndexContext *ctx) = 0;
  virtual void exitNewTableIndex(GpuSqlParser::NewTableIndexContext *ctx) = 0;

  virtual void enterSelectColumns(GpuSqlParser::SelectColumnsContext *ctx) = 0;
  virtual void exitSelectColumns(GpuSqlParser::SelectColumnsContext *ctx) = 0;

  virtual void enterSelectColumn(GpuSqlParser::SelectColumnContext *ctx) = 0;
  virtual void exitSelectColumn(GpuSqlParser::SelectColumnContext *ctx) = 0;

  virtual void enterSelectAllColumns(GpuSqlParser::SelectAllColumnsContext *ctx) = 0;
  virtual void exitSelectAllColumns(GpuSqlParser::SelectAllColumnsContext *ctx) = 0;

  virtual void enterWhereClause(GpuSqlParser::WhereClauseContext *ctx) = 0;
  virtual void exitWhereClause(GpuSqlParser::WhereClauseContext *ctx) = 0;

  virtual void enterOrderByColumns(GpuSqlParser::OrderByColumnsContext *ctx) = 0;
  virtual void exitOrderByColumns(GpuSqlParser::OrderByColumnsContext *ctx) = 0;

  virtual void enterOrderByColumn(GpuSqlParser::OrderByColumnContext *ctx) = 0;
  virtual void exitOrderByColumn(GpuSqlParser::OrderByColumnContext *ctx) = 0;

  virtual void enterInsertIntoValues(GpuSqlParser::InsertIntoValuesContext *ctx) = 0;
  virtual void exitInsertIntoValues(GpuSqlParser::InsertIntoValuesContext *ctx) = 0;

  virtual void enterInsertIntoColumns(GpuSqlParser::InsertIntoColumnsContext *ctx) = 0;
  virtual void exitInsertIntoColumns(GpuSqlParser::InsertIntoColumnsContext *ctx) = 0;

  virtual void enterIndexColumns(GpuSqlParser::IndexColumnsContext *ctx) = 0;
  virtual void exitIndexColumns(GpuSqlParser::IndexColumnsContext *ctx) = 0;

  virtual void enterGroupByColumns(GpuSqlParser::GroupByColumnsContext *ctx) = 0;
  virtual void exitGroupByColumns(GpuSqlParser::GroupByColumnsContext *ctx) = 0;

  virtual void enterGroupByColumn(GpuSqlParser::GroupByColumnContext *ctx) = 0;
  virtual void exitGroupByColumn(GpuSqlParser::GroupByColumnContext *ctx) = 0;

  virtual void enterFromTables(GpuSqlParser::FromTablesContext *ctx) = 0;
  virtual void exitFromTables(GpuSqlParser::FromTablesContext *ctx) = 0;

  virtual void enterJoinClauses(GpuSqlParser::JoinClausesContext *ctx) = 0;
  virtual void exitJoinClauses(GpuSqlParser::JoinClausesContext *ctx) = 0;

  virtual void enterJoinClause(GpuSqlParser::JoinClauseContext *ctx) = 0;
  virtual void exitJoinClause(GpuSqlParser::JoinClauseContext *ctx) = 0;

  virtual void enterJoinTable(GpuSqlParser::JoinTableContext *ctx) = 0;
  virtual void exitJoinTable(GpuSqlParser::JoinTableContext *ctx) = 0;

  virtual void enterJoinColumnLeft(GpuSqlParser::JoinColumnLeftContext *ctx) = 0;
  virtual void exitJoinColumnLeft(GpuSqlParser::JoinColumnLeftContext *ctx) = 0;

  virtual void enterJoinColumnRight(GpuSqlParser::JoinColumnRightContext *ctx) = 0;
  virtual void exitJoinColumnRight(GpuSqlParser::JoinColumnRightContext *ctx) = 0;

  virtual void enterJoinOperator(GpuSqlParser::JoinOperatorContext *ctx) = 0;
  virtual void exitJoinOperator(GpuSqlParser::JoinOperatorContext *ctx) = 0;

  virtual void enterJoinType(GpuSqlParser::JoinTypeContext *ctx) = 0;
  virtual void exitJoinType(GpuSqlParser::JoinTypeContext *ctx) = 0;

  virtual void enterFromTable(GpuSqlParser::FromTableContext *ctx) = 0;
  virtual void exitFromTable(GpuSqlParser::FromTableContext *ctx) = 0;

  virtual void enterColumnId(GpuSqlParser::ColumnIdContext *ctx) = 0;
  virtual void exitColumnId(GpuSqlParser::ColumnIdContext *ctx) = 0;

  virtual void enterTable(GpuSqlParser::TableContext *ctx) = 0;
  virtual void exitTable(GpuSqlParser::TableContext *ctx) = 0;

  virtual void enterColumn(GpuSqlParser::ColumnContext *ctx) = 0;
  virtual void exitColumn(GpuSqlParser::ColumnContext *ctx) = 0;

  virtual void enterDatabase(GpuSqlParser::DatabaseContext *ctx) = 0;
  virtual void exitDatabase(GpuSqlParser::DatabaseContext *ctx) = 0;

  virtual void enterAlias(GpuSqlParser::AliasContext *ctx) = 0;
  virtual void exitAlias(GpuSqlParser::AliasContext *ctx) = 0;

  virtual void enterIndexName(GpuSqlParser::IndexNameContext *ctx) = 0;
  virtual void exitIndexName(GpuSqlParser::IndexNameContext *ctx) = 0;

  virtual void enterLimit(GpuSqlParser::LimitContext *ctx) = 0;
  virtual void exitLimit(GpuSqlParser::LimitContext *ctx) = 0;

  virtual void enterOffset(GpuSqlParser::OffsetContext *ctx) = 0;
  virtual void exitOffset(GpuSqlParser::OffsetContext *ctx) = 0;

  virtual void enterBlockSize(GpuSqlParser::BlockSizeContext *ctx) = 0;
  virtual void exitBlockSize(GpuSqlParser::BlockSizeContext *ctx) = 0;

  virtual void enterColumnValue(GpuSqlParser::ColumnValueContext *ctx) = 0;
  virtual void exitColumnValue(GpuSqlParser::ColumnValueContext *ctx) = 0;

  virtual void enterDecimalLiteral(GpuSqlParser::DecimalLiteralContext *ctx) = 0;
  virtual void exitDecimalLiteral(GpuSqlParser::DecimalLiteralContext *ctx) = 0;

  virtual void enterCastOperation(GpuSqlParser::CastOperationContext *ctx) = 0;
  virtual void exitCastOperation(GpuSqlParser::CastOperationContext *ctx) = 0;

  virtual void enterGeoReference(GpuSqlParser::GeoReferenceContext *ctx) = 0;
  virtual void exitGeoReference(GpuSqlParser::GeoReferenceContext *ctx) = 0;

  virtual void enterDateTimeLiteral(GpuSqlParser::DateTimeLiteralContext *ctx) = 0;
  virtual void exitDateTimeLiteral(GpuSqlParser::DateTimeLiteralContext *ctx) = 0;

  virtual void enterNowLiteral(GpuSqlParser::NowLiteralContext *ctx) = 0;
  virtual void exitNowLiteral(GpuSqlParser::NowLiteralContext *ctx) = 0;

  virtual void enterIntLiteral(GpuSqlParser::IntLiteralContext *ctx) = 0;
  virtual void exitIntLiteral(GpuSqlParser::IntLiteralContext *ctx) = 0;

  virtual void enterTernaryOperation(GpuSqlParser::TernaryOperationContext *ctx) = 0;
  virtual void exitTernaryOperation(GpuSqlParser::TernaryOperationContext *ctx) = 0;

  virtual void enterAggregation(GpuSqlParser::AggregationContext *ctx) = 0;
  virtual void exitAggregation(GpuSqlParser::AggregationContext *ctx) = 0;

  virtual void enterBinaryOperation(GpuSqlParser::BinaryOperationContext *ctx) = 0;
  virtual void exitBinaryOperation(GpuSqlParser::BinaryOperationContext *ctx) = 0;

  virtual void enterParenExpression(GpuSqlParser::ParenExpressionContext *ctx) = 0;
  virtual void exitParenExpression(GpuSqlParser::ParenExpressionContext *ctx) = 0;

  virtual void enterUnaryOperation(GpuSqlParser::UnaryOperationContext *ctx) = 0;
  virtual void exitUnaryOperation(GpuSqlParser::UnaryOperationContext *ctx) = 0;

  virtual void enterVarReference(GpuSqlParser::VarReferenceContext *ctx) = 0;
  virtual void exitVarReference(GpuSqlParser::VarReferenceContext *ctx) = 0;

  virtual void enterPiLiteral(GpuSqlParser::PiLiteralContext *ctx) = 0;
  virtual void exitPiLiteral(GpuSqlParser::PiLiteralContext *ctx) = 0;

  virtual void enterStringLiteral(GpuSqlParser::StringLiteralContext *ctx) = 0;
  virtual void exitStringLiteral(GpuSqlParser::StringLiteralContext *ctx) = 0;

  virtual void enterBooleanLiteral(GpuSqlParser::BooleanLiteralContext *ctx) = 0;
  virtual void exitBooleanLiteral(GpuSqlParser::BooleanLiteralContext *ctx) = 0;

  virtual void enterGeometry(GpuSqlParser::GeometryContext *ctx) = 0;
  virtual void exitGeometry(GpuSqlParser::GeometryContext *ctx) = 0;

  virtual void enterPointGeometry(GpuSqlParser::PointGeometryContext *ctx) = 0;
  virtual void exitPointGeometry(GpuSqlParser::PointGeometryContext *ctx) = 0;

  virtual void enterLineStringGeometry(GpuSqlParser::LineStringGeometryContext *ctx) = 0;
  virtual void exitLineStringGeometry(GpuSqlParser::LineStringGeometryContext *ctx) = 0;

  virtual void enterPolygonGeometry(GpuSqlParser::PolygonGeometryContext *ctx) = 0;
  virtual void exitPolygonGeometry(GpuSqlParser::PolygonGeometryContext *ctx) = 0;

  virtual void enterMultiPointGeometry(GpuSqlParser::MultiPointGeometryContext *ctx) = 0;
  virtual void exitMultiPointGeometry(GpuSqlParser::MultiPointGeometryContext *ctx) = 0;

  virtual void enterMultiLineStringGeometry(GpuSqlParser::MultiLineStringGeometryContext *ctx) = 0;
  virtual void exitMultiLineStringGeometry(GpuSqlParser::MultiLineStringGeometryContext *ctx) = 0;

  virtual void enterMultiPolygonGeometry(GpuSqlParser::MultiPolygonGeometryContext *ctx) = 0;
  virtual void exitMultiPolygonGeometry(GpuSqlParser::MultiPolygonGeometryContext *ctx) = 0;

  virtual void enterPointOrClosedPoint(GpuSqlParser::PointOrClosedPointContext *ctx) = 0;
  virtual void exitPointOrClosedPoint(GpuSqlParser::PointOrClosedPointContext *ctx) = 0;

  virtual void enterPolygon(GpuSqlParser::PolygonContext *ctx) = 0;
  virtual void exitPolygon(GpuSqlParser::PolygonContext *ctx) = 0;

  virtual void enterLineString(GpuSqlParser::LineStringContext *ctx) = 0;
  virtual void exitLineString(GpuSqlParser::LineStringContext *ctx) = 0;

  virtual void enterPoint(GpuSqlParser::PointContext *ctx) = 0;
  virtual void exitPoint(GpuSqlParser::PointContext *ctx) = 0;


};

