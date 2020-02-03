//
// Created by Martin Staňo on 2019-01-15.
//

#ifndef DROPDBASE_INSTAREA_PARSEREXCEPTIONS_H
#define DROPDBASE_INSTAREA_PARSEREXCEPTIONS_H

#include <exception>
#include <string>
#include "DataType.h"

struct DatabaseNotUsedException : public std::runtime_error
{
    DatabaseNotUsedException() : std::runtime_error("No database is currently being used.")
    {
    }
};

struct DatabaseNotFoundException : public std::runtime_error
{
    DatabaseNotFoundException(const std::string& database)
    : std::runtime_error(("Database: \"" + database + "\" was not found.").c_str())
    {
    }
};

struct DatabaseAlreadyExistsException : public std::runtime_error
{
    DatabaseAlreadyExistsException(const std::string& database)
    : std::runtime_error(("Database: \"" + database + "\" already exists.").c_str())
    {
    }
};

struct TableNotFoundFromException : public std::runtime_error
{
    TableNotFoundFromException(const std::string& table)
    : std::runtime_error(("Table: \"" + table + "\" was not found in the FROM clause.").c_str())
    {
    }
};

struct TableAlreadyExistsException : public std::runtime_error
{
    TableAlreadyExistsException(const std::string& table)
    : std::runtime_error(("Table: \"" + table + "\" already exists.").c_str())
    {
    }
};

struct TableIsFilledException : public std::runtime_error
{
    TableIsFilledException() : std::runtime_error("Index cannot be created on filled table.")
    {
    }
};

struct ColumnAmbiguityException : public std::runtime_error
{
    ColumnAmbiguityException(const std::string& column)
    : std::runtime_error(("Column: \"" + column + "\" was found in more than one table.").c_str())
    {
    }
};

struct ColumnAlreadyExistsException : public std::runtime_error
{
    ColumnAlreadyExistsException(const std::string& column)
    : std::runtime_error(("Column: \"" + column + "\" already exists.").c_str())
    {
    }
};

struct ColumnNotFoundException : public std::runtime_error
{
    ColumnNotFoundException(const std::string& column)
    : std::runtime_error(("Column: \"" + column + "\" was not found in table.").c_str())
    {
    }
};

struct AlreadyModifiedColumnException : public std::runtime_error
{
    AlreadyModifiedColumnException(const std::string& column)
    : std::runtime_error(("Column: \"" + column + "\" was already modified in this command.").c_str())
    {
    }
};

struct IndexColumnDataTypeException : public std::runtime_error
{
    IndexColumnDataTypeException(const std::string& column, DataType dataType)
    : std::runtime_error(("Column: \"" + column + "\" of type: " + ::GetStringFromColumnDataType(dataType) + " cannot be used as an index.")
                         .c_str())
    {
    }
};

struct ConstraintAlreadyReferencedException : public std::runtime_error
{
    ConstraintAlreadyReferencedException(const std::string& constraint)
    : std::runtime_error(("Constraint: \"" + constraint + "\" was already referenced in query.").c_str())
    {
    }
};

struct ConstraintAlreadyExistsException : public std::runtime_error
{
    ConstraintAlreadyExistsException(const std::string& constraint)
    : std::runtime_error(("Constraint: \"" + constraint + "\" already exists in table.").c_str())
    {
    }
};

struct ColumnAlreadyConstrainedException : public std::runtime_error
{
    ColumnAlreadyConstrainedException(const std::string& column)
    : std::runtime_error(
          ("Column: \"" + column + "\" is already constrained with a constraint of the same type.").c_str())
    {
    }
};

struct ConstraintNotFound : public std::runtime_error
{
    ConstraintNotFound(const std::string& constraint)
    : std::runtime_error(("Constraint: \"" + constraint + "\" was notfound in table.").c_str())
    {
    }
};

struct ConstraintCannotBeRemovedException : public std::runtime_error
{
    ConstraintCannotBeRemovedException(const std::string& constraint, const std::string& column)
    : std::runtime_error(("Constraint: \"" + constraint + "\" cannot be removed becasue its dependancy of another constraint of column: \"" +
                      column + "\" .")
                         .c_str())
    {
    }
};

struct ColumnGroupByException : public std::runtime_error
{
    ColumnGroupByException(const std::string& column)
    : std::runtime_error(("Column: \"" + column + "\" must appear in GROUP BY clause or be used in AGGREGATE function.")
                         .c_str())
    {
    }
};

struct InsertIntoException : public std::runtime_error
{
    InsertIntoException(const std::string& column)
    : std::runtime_error(
          ("Column: \"" + column + "\" was referenced multiple times in the INSERT INTO command.").c_str())
    {
    }
};

struct NotSameAmoutOfValuesException : public std::runtime_error
{
    NotSameAmoutOfValuesException()
    : std::runtime_error(
          "Number of values provided in the INSERT INTO must be the same as number of columns")
    {
    }
};

struct NestedAggregationException : public std::runtime_error
{
    NestedAggregationException()
    : std::runtime_error("Use of nested aggregation functions is not allowed.")
    {
    }
};

struct AggregationGroupByException : public std::runtime_error
{
    AggregationGroupByException()
    : std::runtime_error("Use of aggregation functions in GROUP BY clause is not allowed.")
    {
    }
};

struct AggregationWhereException : public std::runtime_error
{
    AggregationWhereException()
    : std::runtime_error("Use of aggregation functions in WHERE clause is not allowed.")
    {
    }
};

struct OrderByColumnAlreadyReferencedException : public std::runtime_error
{
    OrderByColumnAlreadyReferencedException(const std::string& column)
    : std::runtime_error(
          ("Column: \"" + column + "\" was referenced multiple times in the ORDER BY clause.").c_str())
    {
    }
};

struct OrderByInvalidColumnException : public std::runtime_error
{
    OrderByInvalidColumnException(const std::string& column)
    : std::runtime_error(("Column: \"" + column + "\" is not a valid ORDER BY column.").c_str())
    {
    }
};

struct GroupByInvalidColumnException : public std::runtime_error
{
    GroupByInvalidColumnException(const std::string& column)
    : std::runtime_error(("Column: \"" + column + "\" is not a valid GROUP BY column.").c_str())
    {
    }
};

struct RetPolygonGroupByException : public std::runtime_error
{
    RetPolygonGroupByException()
    : std::runtime_error("Return of complex polygon WKT is not allowed while using group by.")
    {
    }
};

struct RetPointGroupByException : public std::runtime_error
{
    RetPointGroupByException()
    : std::runtime_error("Return of point WKT is not allowed while using group by.")
    {
    }
};

struct AliasRedefinitionException : public std::runtime_error
{
    AliasRedefinitionException(const std::string& alias)
    : std::runtime_error(("Attempt to redefine alias: \"" + alias + "\" has occured.").c_str())
    {
    }
};

struct NullMaskOperationInvalidOperandException : public std::runtime_error
{
    NullMaskOperationInvalidOperandException()
    : std::runtime_error("Null mask operation can only be called with a column operand.")
    {
    }
};

struct JoinColumnTypeException : public std::runtime_error
{
    JoinColumnTypeException(const std::string& colA, const std::string& colB)
    : std::runtime_error(("Attempt to join columns: \"" + colA + "\" and \"" + colB + "\" which have different data types.")
                         .c_str())
    {
    }
};

struct InvalidOperandsException : public std::runtime_error
{
    InvalidOperandsException(const std::string& left, const std::string& right, const std::string& op)
    : std::runtime_error(("Invalid operands: " + left + " " + right + " for operation: " + op).c_str())
    {
    }
};


#endif // DROPDBASE_INSTAREA_PARSEREXCEPTIONS_H
