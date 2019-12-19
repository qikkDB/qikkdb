//
// Created by Martin Staňo on 2019-01-15.
//

#ifndef DROPDBASE_INSTAREA_PARSEREXCEPTIONS_H
#define DROPDBASE_INSTAREA_PARSEREXCEPTIONS_H

#include <exception>
#include <string>
#include "DataType.h"

struct DatabaseNotUsedException : public std::exception
{
    DatabaseNotUsedException() : std::exception("No database is currently being used.")
    {
    }
};

struct DatabaseNotFoundException : public std::exception
{
    DatabaseNotFoundException(const std::string& database)
    : std::exception(("Database: \"" + database + "\" was not found.").c_str())
    {
    }
};

struct DatabaseAlreadyExistsException : public std::exception
{
    DatabaseAlreadyExistsException(const std::string& database)
    : std::exception(("Database: \"" + database + "\" already exists.").c_str())
    {
    }
};

struct TableNotFoundFromException : public std::exception
{
    TableNotFoundFromException(const std::string& table)
    : std::exception(("Table: \"" + table + "\" was not found in the FROM clause.").c_str())
    {
    }
};

struct TableAlreadyExistsException : public std::exception
{
    TableAlreadyExistsException(const std::string& table)
    : std::exception(("Table: \"" + table + "\" already exists.").c_str())
    {
    }
};

struct TableIsFilledException : public std::exception
{
    TableIsFilledException() : std::exception("Index cannot be created on filled table.")
    {
    }
};

struct ColumnAmbiguityException : public std::exception
{
    ColumnAmbiguityException(const std::string& column)
    : std::exception(("Column: \"" + column + "\" was found in more than one table.").c_str())
    {
    }
};

struct ColumnAlreadyExistsException : public std::exception
{
    ColumnAlreadyExistsException(const std::string& column)
    : std::exception(("Column: \"" + column + "\" already exists.").c_str())
    {
    }
};

struct ColumnNotFoundException : public std::exception
{
    ColumnNotFoundException(const std::string& column)
    : std::exception(("Column: \"" + column + "\" was not found in table.").c_str())
    {
    }
};

struct AlreadyModifiedColumnException : public std::exception
{
    AlreadyModifiedColumnException(const std::string& column)
    : std::exception(("Column: \"" + column + "\" was already modified in this command.").c_str())
    {
    }
};

struct IndexColumnDataTypeException : public std::exception
{
    IndexColumnDataTypeException(const std::string& column, DataType dataType)
    : std::exception(("Column: \"" + column + "\" of type: " + ::GetStringFromColumnDataType(dataType) + " cannot be used as an index.")
                         .c_str())
    {
    }
};

struct ConstraintAlreadyReferencedException : public std::exception
{
    ConstraintAlreadyReferencedException(const std::string& constraint)
    : std::exception(("Constraint: \"" + constraint + "\" was already referenced in query.").c_str())
    {
    }
};

struct ConstraintAlreadyExistsException : public std::exception
{
    ConstraintAlreadyExistsException(const std::string& constraint)
    : std::exception(("Constraint: \"" + constraint + "\" already exists in table.").c_str())
    {
    }
};

struct ColumnAlreadyConstrainedException : public std::exception
{
    ColumnAlreadyConstrainedException(const std::string& column)
    : std::exception(
          ("Column: \"" + column + "\" is already constrained with a constraint of the same type.").c_str())
    {
    }
};

struct ConstraintNotFound : public std::exception
{
    ConstraintNotFound(const std::string& constraint)
    : std::exception(("Constraint: \"" + constraint + "\" was notfound in table.").c_str())
    {
    }
};

struct ColumnGroupByException : public std::exception
{
    ColumnGroupByException(const std::string& column)
    : std::exception(("Column: \"" + column + "\" must appear in GROUP BY clause or be used in AGGREGATE function.")
                         .c_str())
    {
    }
};

struct InsertIntoException : public std::exception
{
    InsertIntoException(const std::string& column)
    : std::exception(
          ("Column: \"" + column + "\" was referenced multiple times in the INSERT INTO command.").c_str())
    {
    }
};

struct NotSameAmoutOfValuesException : public std::exception
{
    NotSameAmoutOfValuesException()
    : std::exception(
          "Number of values provided in the INSERT INTO must be the same as number of columns")
    {
    }
};

struct NestedAggregationException : public std::exception
{
    NestedAggregationException()
    : std::exception("Use of nested aggregation functions is not allowed.")
    {
    }
};

struct AggregationGroupByException : public std::exception
{
    AggregationGroupByException()
    : std::exception("Use of aggregation functions in GROUP BY clause is not allowed.")
    {
    }
};

struct AggregationWhereException : public std::exception
{
    AggregationWhereException()
    : std::exception("Use of aggregation functions in WHERE clause is not allowed.")
    {
    }
};

struct OrderByColumnAlreadyReferencedException : public std::exception
{
    OrderByColumnAlreadyReferencedException(const std::string& column)
    : std::exception(
          ("Column: \"" + column + "\" was referenced multiple times in the ORDER BY clause.").c_str())
    {
    }
};

struct OrderByInvalidColumnException : public std::exception
{
    OrderByInvalidColumnException(const std::string& column)
    : std::exception(("Column: \"" + column + "\" is not a valid ORDER BY column.").c_str())
    {
    }
};

struct GroupByInvalidColumnException : public std::exception
{
    GroupByInvalidColumnException(const std::string& column)
    : std::exception(("Column: \"" + column + "\" is not a valid GROUP BY column.").c_str())
    {
    }
};

struct RetPolygonGroupByException : public std::exception
{
    RetPolygonGroupByException()
    : std::exception("Return of complex polygon WKT is not allowed while using group by.")
    {
    }
};

struct RetPointGroupByException : public std::exception
{
    RetPointGroupByException()
    : std::exception("Return of point WKT is not allowed while using group by.")
    {
    }
};

struct AliasRedefinitionException : public std::exception
{
    AliasRedefinitionException(const std::string& alias)
    : std::exception(("Attempt to redefine alias: \"" + alias + "\" has occured.").c_str())
    {
    }
};

struct NullMaskOperationInvalidOperandException : public std::exception
{
    NullMaskOperationInvalidOperandException()
    : std::exception("Null mask operation can only be called with a column operand.")
    {
    }
};

struct JoinColumnTypeException : public std::exception
{
    JoinColumnTypeException(const std::string& colA, const std::string& colB)
    : std::exception(("Attempt to join columns: \"" + colA + "\" and \"" + colB + "\" which have different data types.")
                         .c_str())
    {
    }
};

struct InvalidOperandsException : public std::exception
{
    InvalidOperandsException(const std::string& left, const std::string& right, const std::string& op)
    : std::exception(("Invalid operands: " + left + " " + right + " for operation: " + op).c_str())
    {
    }
};


#endif // DROPDBASE_INSTAREA_PARSEREXCEPTIONS_H
