
// Generated from /Users/ms/dropdbase_instarea/dropdbase/GpuSqlParser/GpuSqlParser.g4 by ANTLR 4.7.2

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

  virtual void enterSqlCreateTable(GpuSqlParser::SqlCreateTableContext * /*ctx*/) override { }
  virtual void exitSqlCreateTable(GpuSqlParser::SqlCreateTableContext * /*ctx*/) override { }

  virtual void enterSqlInsertInto(GpuSqlParser::SqlInsertIntoContext * /*ctx*/) override { }
  virtual void exitSqlInsertInto(GpuSqlParser::SqlInsertIntoContext * /*ctx*/) override { }

  virtual void enterNewTableColumns(GpuSqlParser::NewTableColumnsContext * /*ctx*/) override { }
  virtual void exitNewTableColumns(GpuSqlParser::NewTableColumnsContext * /*ctx*/) override { }

  virtual void enterNewTableColumn(GpuSqlParser::NewTableColumnContext * /*ctx*/) override { }
  virtual void exitNewTableColumn(GpuSqlParser::NewTableColumnContext * /*ctx*/) override { }

  virtual void enterSelectColumns(GpuSqlParser::SelectColumnsContext * /*ctx*/) override { }
  virtual void exitSelectColumns(GpuSqlParser::SelectColumnsContext * /*ctx*/) override { }

  virtual void enterSelectColumn(GpuSqlParser::SelectColumnContext * /*ctx*/) override { }
  virtual void exitSelectColumn(GpuSqlParser::SelectColumnContext * /*ctx*/) override { }

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

  virtual void enterGroupByColumns(GpuSqlParser::GroupByColumnsContext * /*ctx*/) override { }
  virtual void exitGroupByColumns(GpuSqlParser::GroupByColumnsContext * /*ctx*/) override { }

  virtual void enterColumnId(GpuSqlParser::ColumnIdContext * /*ctx*/) override { }
  virtual void exitColumnId(GpuSqlParser::ColumnIdContext * /*ctx*/) override { }

  virtual void enterFromTables(GpuSqlParser::FromTablesContext * /*ctx*/) override { }
  virtual void exitFromTables(GpuSqlParser::FromTablesContext * /*ctx*/) override { }

  virtual void enterJoinClauses(GpuSqlParser::JoinClausesContext * /*ctx*/) override { }
  virtual void exitJoinClauses(GpuSqlParser::JoinClausesContext * /*ctx*/) override { }

  virtual void enterJoinClause(GpuSqlParser::JoinClauseContext * /*ctx*/) override { }
  virtual void exitJoinClause(GpuSqlParser::JoinClauseContext * /*ctx*/) override { }

  virtual void enterJoinTable(GpuSqlParser::JoinTableContext * /*ctx*/) override { }
  virtual void exitJoinTable(GpuSqlParser::JoinTableContext * /*ctx*/) override { }

  virtual void enterTable(GpuSqlParser::TableContext * /*ctx*/) override { }
  virtual void exitTable(GpuSqlParser::TableContext * /*ctx*/) override { }

  virtual void enterColumn(GpuSqlParser::ColumnContext * /*ctx*/) override { }
  virtual void exitColumn(GpuSqlParser::ColumnContext * /*ctx*/) override { }

  virtual void enterDatabase(GpuSqlParser::DatabaseContext * /*ctx*/) override { }
  virtual void exitDatabase(GpuSqlParser::DatabaseContext * /*ctx*/) override { }

  virtual void enterLimit(GpuSqlParser::LimitContext * /*ctx*/) override { }
  virtual void exitLimit(GpuSqlParser::LimitContext * /*ctx*/) override { }

  virtual void enterOffset(GpuSqlParser::OffsetContext * /*ctx*/) override { }
  virtual void exitOffset(GpuSqlParser::OffsetContext * /*ctx*/) override { }

  virtual void enterColumnValue(GpuSqlParser::ColumnValueContext * /*ctx*/) override { }
  virtual void exitColumnValue(GpuSqlParser::ColumnValueContext * /*ctx*/) override { }

  virtual void enterDecimalLiteral(GpuSqlParser::DecimalLiteralContext * /*ctx*/) override { }
  virtual void exitDecimalLiteral(GpuSqlParser::DecimalLiteralContext * /*ctx*/) override { }

  virtual void enterGeoReference(GpuSqlParser::GeoReferenceContext * /*ctx*/) override { }
  virtual void exitGeoReference(GpuSqlParser::GeoReferenceContext * /*ctx*/) override { }

  virtual void enterMinusExpression(GpuSqlParser::MinusExpressionContext * /*ctx*/) override { }
  virtual void exitMinusExpression(GpuSqlParser::MinusExpressionContext * /*ctx*/) override { }

  virtual void enterStringLiteral(GpuSqlParser::StringLiteralContext * /*ctx*/) override { }
  virtual void exitStringLiteral(GpuSqlParser::StringLiteralContext * /*ctx*/) override { }

  virtual void enterIntLiteral(GpuSqlParser::IntLiteralContext * /*ctx*/) override { }
  virtual void exitIntLiteral(GpuSqlParser::IntLiteralContext * /*ctx*/) override { }

  virtual void enterTernaryOperation(GpuSqlParser::TernaryOperationContext * /*ctx*/) override { }
  virtual void exitTernaryOperation(GpuSqlParser::TernaryOperationContext * /*ctx*/) override { }

  virtual void enterAggregation(GpuSqlParser::AggregationContext * /*ctx*/) override { }
  virtual void exitAggregation(GpuSqlParser::AggregationContext * /*ctx*/) override { }

  virtual void enterParenExpression(GpuSqlParser::ParenExpressionContext * /*ctx*/) override { }
  virtual void exitParenExpression(GpuSqlParser::ParenExpressionContext * /*ctx*/) override { }

  virtual void enterBinaryOperation(GpuSqlParser::BinaryOperationContext * /*ctx*/) override { }
  virtual void exitBinaryOperation(GpuSqlParser::BinaryOperationContext * /*ctx*/) override { }

  virtual void enterUnaryOperation(GpuSqlParser::UnaryOperationContext * /*ctx*/) override { }
  virtual void exitUnaryOperation(GpuSqlParser::UnaryOperationContext * /*ctx*/) override { }

  virtual void enterBooleanLiteral(GpuSqlParser::BooleanLiteralContext * /*ctx*/) override { }
  virtual void exitBooleanLiteral(GpuSqlParser::BooleanLiteralContext * /*ctx*/) override { }

  virtual void enterVarReference(GpuSqlParser::VarReferenceContext * /*ctx*/) override { }
  virtual void exitVarReference(GpuSqlParser::VarReferenceContext * /*ctx*/) override { }

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
