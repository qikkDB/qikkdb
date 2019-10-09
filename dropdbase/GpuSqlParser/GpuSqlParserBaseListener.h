
// Generated from C:/GPU-DB/dropdbase/GpuSqlParser\GpuSqlParser.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"
#include "GpuSqlParserListener.h"


/**
 * This class provides an empty implementation of GpuSqlParserListener,
 * which can be extended to create a listener which only needs to handle a subset
 * of the available methods.
 */
class  GpuSqlParserBaseListener : public GpuSqlParserListener {
public:

  virtual void enterSqlFile(GpuSqlParser::SqlFileContext * /*ctx*/) override { }
  virtual void exitSqlFile(GpuSqlParser::SqlFileContext * /*ctx*/) override { }

  virtual void enterStatement(GpuSqlParser::StatementContext * /*ctx*/) override { }
  virtual void exitStatement(GpuSqlParser::StatementContext * /*ctx*/) override { }

  virtual void enterShowStatement(GpuSqlParser::ShowStatementContext * /*ctx*/) override { }
  virtual void exitShowStatement(GpuSqlParser::ShowStatementContext * /*ctx*/) override { }

  virtual void enterShowDatabases(GpuSqlParser::ShowDatabasesContext * /*ctx*/) override { }
  virtual void exitShowDatabases(GpuSqlParser::ShowDatabasesContext * /*ctx*/) override { }

  virtual void enterShowTables(GpuSqlParser::ShowTablesContext * /*ctx*/) override { }
  virtual void exitShowTables(GpuSqlParser::ShowTablesContext * /*ctx*/) override { }

  virtual void enterShowColumns(GpuSqlParser::ShowColumnsContext * /*ctx*/) override { }
  virtual void exitShowColumns(GpuSqlParser::ShowColumnsContext * /*ctx*/) override { }

  virtual void enterSqlSelect(GpuSqlParser::SqlSelectContext * /*ctx*/) override { }
  virtual void exitSqlSelect(GpuSqlParser::SqlSelectContext * /*ctx*/) override { }

  virtual void enterSqlCreateDb(GpuSqlParser::SqlCreateDbContext * /*ctx*/) override { }
  virtual void exitSqlCreateDb(GpuSqlParser::SqlCreateDbContext * /*ctx*/) override { }

  virtual void enterSqlDropDb(GpuSqlParser::SqlDropDbContext * /*ctx*/) override { }
  virtual void exitSqlDropDb(GpuSqlParser::SqlDropDbContext * /*ctx*/) override { }

  virtual void enterSqlCreateTable(GpuSqlParser::SqlCreateTableContext * /*ctx*/) override { }
  virtual void exitSqlCreateTable(GpuSqlParser::SqlCreateTableContext * /*ctx*/) override { }

  virtual void enterSqlDropTable(GpuSqlParser::SqlDropTableContext * /*ctx*/) override { }
  virtual void exitSqlDropTable(GpuSqlParser::SqlDropTableContext * /*ctx*/) override { }

  virtual void enterSqlAlterTable(GpuSqlParser::SqlAlterTableContext * /*ctx*/) override { }
  virtual void exitSqlAlterTable(GpuSqlParser::SqlAlterTableContext * /*ctx*/) override { }

  virtual void enterSqlAlterDatabase(GpuSqlParser::SqlAlterDatabaseContext * /*ctx*/) override { }
  virtual void exitSqlAlterDatabase(GpuSqlParser::SqlAlterDatabaseContext * /*ctx*/) override { }

  virtual void enterSqlCreateIndex(GpuSqlParser::SqlCreateIndexContext * /*ctx*/) override { }
  virtual void exitSqlCreateIndex(GpuSqlParser::SqlCreateIndexContext * /*ctx*/) override { }

  virtual void enterSqlInsertInto(GpuSqlParser::SqlInsertIntoContext * /*ctx*/) override { }
  virtual void exitSqlInsertInto(GpuSqlParser::SqlInsertIntoContext * /*ctx*/) override { }

  virtual void enterNewTableEntries(GpuSqlParser::NewTableEntriesContext * /*ctx*/) override { }
  virtual void exitNewTableEntries(GpuSqlParser::NewTableEntriesContext * /*ctx*/) override { }

  virtual void enterNewTableEntry(GpuSqlParser::NewTableEntryContext * /*ctx*/) override { }
  virtual void exitNewTableEntry(GpuSqlParser::NewTableEntryContext * /*ctx*/) override { }

  virtual void enterAlterDatabaseEntries(GpuSqlParser::AlterDatabaseEntriesContext * /*ctx*/) override { }
  virtual void exitAlterDatabaseEntries(GpuSqlParser::AlterDatabaseEntriesContext * /*ctx*/) override { }

  virtual void enterAlterDatabaseEntry(GpuSqlParser::AlterDatabaseEntryContext * /*ctx*/) override { }
  virtual void exitAlterDatabaseEntry(GpuSqlParser::AlterDatabaseEntryContext * /*ctx*/) override { }

  virtual void enterRenameDatabase(GpuSqlParser::RenameDatabaseContext * /*ctx*/) override { }
  virtual void exitRenameDatabase(GpuSqlParser::RenameDatabaseContext * /*ctx*/) override { }

  virtual void enterAlterTableEntries(GpuSqlParser::AlterTableEntriesContext * /*ctx*/) override { }
  virtual void exitAlterTableEntries(GpuSqlParser::AlterTableEntriesContext * /*ctx*/) override { }

  virtual void enterAlterTableEntry(GpuSqlParser::AlterTableEntryContext * /*ctx*/) override { }
  virtual void exitAlterTableEntry(GpuSqlParser::AlterTableEntryContext * /*ctx*/) override { }

  virtual void enterAddColumn(GpuSqlParser::AddColumnContext * /*ctx*/) override { }
  virtual void exitAddColumn(GpuSqlParser::AddColumnContext * /*ctx*/) override { }

  virtual void enterDropColumn(GpuSqlParser::DropColumnContext * /*ctx*/) override { }
  virtual void exitDropColumn(GpuSqlParser::DropColumnContext * /*ctx*/) override { }

  virtual void enterAlterColumn(GpuSqlParser::AlterColumnContext * /*ctx*/) override { }
  virtual void exitAlterColumn(GpuSqlParser::AlterColumnContext * /*ctx*/) override { }

  virtual void enterRenameColumn(GpuSqlParser::RenameColumnContext * /*ctx*/) override { }
  virtual void exitRenameColumn(GpuSqlParser::RenameColumnContext * /*ctx*/) override { }

  virtual void enterRenameTable(GpuSqlParser::RenameTableContext * /*ctx*/) override { }
  virtual void exitRenameTable(GpuSqlParser::RenameTableContext * /*ctx*/) override { }

  virtual void enterAddConstraint(GpuSqlParser::AddConstraintContext * /*ctx*/) override { }
  virtual void exitAddConstraint(GpuSqlParser::AddConstraintContext * /*ctx*/) override { }

  virtual void enterDropConstraint(GpuSqlParser::DropConstraintContext * /*ctx*/) override { }
  virtual void exitDropConstraint(GpuSqlParser::DropConstraintContext * /*ctx*/) override { }

  virtual void enterRenameColumnFrom(GpuSqlParser::RenameColumnFromContext * /*ctx*/) override { }
  virtual void exitRenameColumnFrom(GpuSqlParser::RenameColumnFromContext * /*ctx*/) override { }

  virtual void enterRenameColumnTo(GpuSqlParser::RenameColumnToContext * /*ctx*/) override { }
  virtual void exitRenameColumnTo(GpuSqlParser::RenameColumnToContext * /*ctx*/) override { }

  virtual void enterNewTableColumn(GpuSqlParser::NewTableColumnContext * /*ctx*/) override { }
  virtual void exitNewTableColumn(GpuSqlParser::NewTableColumnContext * /*ctx*/) override { }

  virtual void enterNewTableConstraint(GpuSqlParser::NewTableConstraintContext * /*ctx*/) override { }
  virtual void exitNewTableConstraint(GpuSqlParser::NewTableConstraintContext * /*ctx*/) override { }

  virtual void enterSelectColumns(GpuSqlParser::SelectColumnsContext * /*ctx*/) override { }
  virtual void exitSelectColumns(GpuSqlParser::SelectColumnsContext * /*ctx*/) override { }

  virtual void enterSelectColumn(GpuSqlParser::SelectColumnContext * /*ctx*/) override { }
  virtual void exitSelectColumn(GpuSqlParser::SelectColumnContext * /*ctx*/) override { }

  virtual void enterSelectAllColumns(GpuSqlParser::SelectAllColumnsContext * /*ctx*/) override { }
  virtual void exitSelectAllColumns(GpuSqlParser::SelectAllColumnsContext * /*ctx*/) override { }

  virtual void enterWhereClause(GpuSqlParser::WhereClauseContext * /*ctx*/) override { }
  virtual void exitWhereClause(GpuSqlParser::WhereClauseContext * /*ctx*/) override { }

  virtual void enterOrderByColumns(GpuSqlParser::OrderByColumnsContext * /*ctx*/) override { }
  virtual void exitOrderByColumns(GpuSqlParser::OrderByColumnsContext * /*ctx*/) override { }

  virtual void enterOrderByColumn(GpuSqlParser::OrderByColumnContext * /*ctx*/) override { }
  virtual void exitOrderByColumn(GpuSqlParser::OrderByColumnContext * /*ctx*/) override { }

  virtual void enterInsertIntoValues(GpuSqlParser::InsertIntoValuesContext * /*ctx*/) override { }
  virtual void exitInsertIntoValues(GpuSqlParser::InsertIntoValuesContext * /*ctx*/) override { }

  virtual void enterInsertIntoColumns(GpuSqlParser::InsertIntoColumnsContext * /*ctx*/) override { }
  virtual void exitInsertIntoColumns(GpuSqlParser::InsertIntoColumnsContext * /*ctx*/) override { }

  virtual void enterIndexColumns(GpuSqlParser::IndexColumnsContext * /*ctx*/) override { }
  virtual void exitIndexColumns(GpuSqlParser::IndexColumnsContext * /*ctx*/) override { }

  virtual void enterConstraintColumns(GpuSqlParser::ConstraintColumnsContext * /*ctx*/) override { }
  virtual void exitConstraintColumns(GpuSqlParser::ConstraintColumnsContext * /*ctx*/) override { }

  virtual void enterGroupByColumns(GpuSqlParser::GroupByColumnsContext * /*ctx*/) override { }
  virtual void exitGroupByColumns(GpuSqlParser::GroupByColumnsContext * /*ctx*/) override { }

  virtual void enterGroupByColumn(GpuSqlParser::GroupByColumnContext * /*ctx*/) override { }
  virtual void exitGroupByColumn(GpuSqlParser::GroupByColumnContext * /*ctx*/) override { }

  virtual void enterFromTables(GpuSqlParser::FromTablesContext * /*ctx*/) override { }
  virtual void exitFromTables(GpuSqlParser::FromTablesContext * /*ctx*/) override { }

  virtual void enterJoinClauses(GpuSqlParser::JoinClausesContext * /*ctx*/) override { }
  virtual void exitJoinClauses(GpuSqlParser::JoinClausesContext * /*ctx*/) override { }

  virtual void enterJoinClause(GpuSqlParser::JoinClauseContext * /*ctx*/) override { }
  virtual void exitJoinClause(GpuSqlParser::JoinClauseContext * /*ctx*/) override { }

  virtual void enterJoinTable(GpuSqlParser::JoinTableContext * /*ctx*/) override { }
  virtual void exitJoinTable(GpuSqlParser::JoinTableContext * /*ctx*/) override { }

  virtual void enterJoinColumnLeft(GpuSqlParser::JoinColumnLeftContext * /*ctx*/) override { }
  virtual void exitJoinColumnLeft(GpuSqlParser::JoinColumnLeftContext * /*ctx*/) override { }

  virtual void enterJoinColumnRight(GpuSqlParser::JoinColumnRightContext * /*ctx*/) override { }
  virtual void exitJoinColumnRight(GpuSqlParser::JoinColumnRightContext * /*ctx*/) override { }

  virtual void enterJoinOperator(GpuSqlParser::JoinOperatorContext * /*ctx*/) override { }
  virtual void exitJoinOperator(GpuSqlParser::JoinOperatorContext * /*ctx*/) override { }

  virtual void enterJoinType(GpuSqlParser::JoinTypeContext * /*ctx*/) override { }
  virtual void exitJoinType(GpuSqlParser::JoinTypeContext * /*ctx*/) override { }

  virtual void enterFromTable(GpuSqlParser::FromTableContext * /*ctx*/) override { }
  virtual void exitFromTable(GpuSqlParser::FromTableContext * /*ctx*/) override { }

  virtual void enterColumnId(GpuSqlParser::ColumnIdContext * /*ctx*/) override { }
  virtual void exitColumnId(GpuSqlParser::ColumnIdContext * /*ctx*/) override { }

  virtual void enterTable(GpuSqlParser::TableContext * /*ctx*/) override { }
  virtual void exitTable(GpuSqlParser::TableContext * /*ctx*/) override { }

  virtual void enterColumn(GpuSqlParser::ColumnContext * /*ctx*/) override { }
  virtual void exitColumn(GpuSqlParser::ColumnContext * /*ctx*/) override { }

  virtual void enterDatabase(GpuSqlParser::DatabaseContext * /*ctx*/) override { }
  virtual void exitDatabase(GpuSqlParser::DatabaseContext * /*ctx*/) override { }

  virtual void enterAlias(GpuSqlParser::AliasContext * /*ctx*/) override { }
  virtual void exitAlias(GpuSqlParser::AliasContext * /*ctx*/) override { }

  virtual void enterIndexName(GpuSqlParser::IndexNameContext * /*ctx*/) override { }
  virtual void exitIndexName(GpuSqlParser::IndexNameContext * /*ctx*/) override { }

  virtual void enterConstraintName(GpuSqlParser::ConstraintNameContext * /*ctx*/) override { }
  virtual void exitConstraintName(GpuSqlParser::ConstraintNameContext * /*ctx*/) override { }

  virtual void enterLimit(GpuSqlParser::LimitContext * /*ctx*/) override { }
  virtual void exitLimit(GpuSqlParser::LimitContext * /*ctx*/) override { }

  virtual void enterOffset(GpuSqlParser::OffsetContext * /*ctx*/) override { }
  virtual void exitOffset(GpuSqlParser::OffsetContext * /*ctx*/) override { }

  virtual void enterBlockSize(GpuSqlParser::BlockSizeContext * /*ctx*/) override { }
  virtual void exitBlockSize(GpuSqlParser::BlockSizeContext * /*ctx*/) override { }

  virtual void enterColumnValue(GpuSqlParser::ColumnValueContext * /*ctx*/) override { }
  virtual void exitColumnValue(GpuSqlParser::ColumnValueContext * /*ctx*/) override { }

  virtual void enterConstraint(GpuSqlParser::ConstraintContext * /*ctx*/) override { }
  virtual void exitConstraint(GpuSqlParser::ConstraintContext * /*ctx*/) override { }

  virtual void enterDecimalLiteral(GpuSqlParser::DecimalLiteralContext * /*ctx*/) override { }
  virtual void exitDecimalLiteral(GpuSqlParser::DecimalLiteralContext * /*ctx*/) override { }

  virtual void enterCastOperation(GpuSqlParser::CastOperationContext * /*ctx*/) override { }
  virtual void exitCastOperation(GpuSqlParser::CastOperationContext * /*ctx*/) override { }

  virtual void enterGeoReference(GpuSqlParser::GeoReferenceContext * /*ctx*/) override { }
  virtual void exitGeoReference(GpuSqlParser::GeoReferenceContext * /*ctx*/) override { }

  virtual void enterDateTimeLiteral(GpuSqlParser::DateTimeLiteralContext * /*ctx*/) override { }
  virtual void exitDateTimeLiteral(GpuSqlParser::DateTimeLiteralContext * /*ctx*/) override { }

  virtual void enterNowLiteral(GpuSqlParser::NowLiteralContext * /*ctx*/) override { }
  virtual void exitNowLiteral(GpuSqlParser::NowLiteralContext * /*ctx*/) override { }

  virtual void enterIntLiteral(GpuSqlParser::IntLiteralContext * /*ctx*/) override { }
  virtual void exitIntLiteral(GpuSqlParser::IntLiteralContext * /*ctx*/) override { }

  virtual void enterTernaryOperation(GpuSqlParser::TernaryOperationContext * /*ctx*/) override { }
  virtual void exitTernaryOperation(GpuSqlParser::TernaryOperationContext * /*ctx*/) override { }

  virtual void enterAggregation(GpuSqlParser::AggregationContext * /*ctx*/) override { }
  virtual void exitAggregation(GpuSqlParser::AggregationContext * /*ctx*/) override { }

  virtual void enterBinaryOperation(GpuSqlParser::BinaryOperationContext * /*ctx*/) override { }
  virtual void exitBinaryOperation(GpuSqlParser::BinaryOperationContext * /*ctx*/) override { }

  virtual void enterParenExpression(GpuSqlParser::ParenExpressionContext * /*ctx*/) override { }
  virtual void exitParenExpression(GpuSqlParser::ParenExpressionContext * /*ctx*/) override { }

  virtual void enterUnaryOperation(GpuSqlParser::UnaryOperationContext * /*ctx*/) override { }
  virtual void exitUnaryOperation(GpuSqlParser::UnaryOperationContext * /*ctx*/) override { }

  virtual void enterVarReference(GpuSqlParser::VarReferenceContext * /*ctx*/) override { }
  virtual void exitVarReference(GpuSqlParser::VarReferenceContext * /*ctx*/) override { }

  virtual void enterPiLiteral(GpuSqlParser::PiLiteralContext * /*ctx*/) override { }
  virtual void exitPiLiteral(GpuSqlParser::PiLiteralContext * /*ctx*/) override { }

  virtual void enterStringLiteral(GpuSqlParser::StringLiteralContext * /*ctx*/) override { }
  virtual void exitStringLiteral(GpuSqlParser::StringLiteralContext * /*ctx*/) override { }

  virtual void enterBooleanLiteral(GpuSqlParser::BooleanLiteralContext * /*ctx*/) override { }
  virtual void exitBooleanLiteral(GpuSqlParser::BooleanLiteralContext * /*ctx*/) override { }

  virtual void enterDatatype(GpuSqlParser::DatatypeContext * /*ctx*/) override { }
  virtual void exitDatatype(GpuSqlParser::DatatypeContext * /*ctx*/) override { }

  virtual void enterGeometry(GpuSqlParser::GeometryContext * /*ctx*/) override { }
  virtual void exitGeometry(GpuSqlParser::GeometryContext * /*ctx*/) override { }

  virtual void enterPointGeometry(GpuSqlParser::PointGeometryContext * /*ctx*/) override { }
  virtual void exitPointGeometry(GpuSqlParser::PointGeometryContext * /*ctx*/) override { }

  virtual void enterLineStringGeometry(GpuSqlParser::LineStringGeometryContext * /*ctx*/) override { }
  virtual void exitLineStringGeometry(GpuSqlParser::LineStringGeometryContext * /*ctx*/) override { }

  virtual void enterPolygonGeometry(GpuSqlParser::PolygonGeometryContext * /*ctx*/) override { }
  virtual void exitPolygonGeometry(GpuSqlParser::PolygonGeometryContext * /*ctx*/) override { }

  virtual void enterMultiPointGeometry(GpuSqlParser::MultiPointGeometryContext * /*ctx*/) override { }
  virtual void exitMultiPointGeometry(GpuSqlParser::MultiPointGeometryContext * /*ctx*/) override { }

  virtual void enterMultiLineStringGeometry(GpuSqlParser::MultiLineStringGeometryContext * /*ctx*/) override { }
  virtual void exitMultiLineStringGeometry(GpuSqlParser::MultiLineStringGeometryContext * /*ctx*/) override { }

  virtual void enterMultiPolygonGeometry(GpuSqlParser::MultiPolygonGeometryContext * /*ctx*/) override { }
  virtual void exitMultiPolygonGeometry(GpuSqlParser::MultiPolygonGeometryContext * /*ctx*/) override { }

  virtual void enterPointOrClosedPoint(GpuSqlParser::PointOrClosedPointContext * /*ctx*/) override { }
  virtual void exitPointOrClosedPoint(GpuSqlParser::PointOrClosedPointContext * /*ctx*/) override { }

  virtual void enterPolygon(GpuSqlParser::PolygonContext * /*ctx*/) override { }
  virtual void exitPolygon(GpuSqlParser::PolygonContext * /*ctx*/) override { }

  virtual void enterLineString(GpuSqlParser::LineStringContext * /*ctx*/) override { }
  virtual void exitLineString(GpuSqlParser::LineStringContext * /*ctx*/) override { }

  virtual void enterPoint(GpuSqlParser::PointContext * /*ctx*/) override { }
  virtual void exitPoint(GpuSqlParser::PointContext * /*ctx*/) override { }


  virtual void enterEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void exitEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void visitTerminal(antlr4::tree::TerminalNode * /*node*/) override { }
  virtual void visitErrorNode(antlr4::tree::ErrorNode * /*node*/) override { }

};

