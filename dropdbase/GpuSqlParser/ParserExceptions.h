//
// Created by Martin Staňo on 2019-01-15.
//

#ifndef DROPDBASE_INSTAREA_PARSEREXCEPTIONS_H
#define DROPDBASE_INSTAREA_PARSEREXCEPTIONS_H

#include <exception>
#include <string>

struct DatabaseNotUsedException : public std::exception
{
    const char* what() const noexcept override
    {
        return "No database is currently being used.";
    }
};

struct DatabaseNotFoundException : public std::exception
{
    DatabaseNotFoundException(const std::string& database)
    : message_("Database: \"" + database + "\" was not found.")
    {
    }

    const char* what() const noexcept override
    {
        return message_.c_str();
    }

private:
    std::string message_;
};

struct DatabaseAlreadyExistsException : public std::exception
{
    DatabaseAlreadyExistsException(const std::string& database)
    : message_("Database: \"" + database + "\" already exists.")
    {
    }

    const char* what() const noexcept override
    {
        return message_.c_str();
    }

private:
    std::string message_;
};

struct TableNotFoundFromException : public std::exception
{
    TableNotFoundFromException(const std::string& table)
    : message_("Table: \"" + table + "\" was not found in the FROM clause.")
    {
    }

    const char* what() const noexcept override
    {
        return message_.c_str();
    }

private:
    std::string message_;
};

struct TableAlreadyExistsException : public std::exception
{
    TableAlreadyExistsException(const std::string& table)
    : message_("Table: \"" + table + "\" already exists.")
    {
    }

    const char* what() const noexcept override
    {
        return message_.c_str();
    }

private:
    std::string message_;
};

struct TableIsFilledException : public std::exception
{
    const char* what() const noexcept override
    {
        return "Index cannot be created on filled table.";
    }
};

struct ColumnAmbiguityException : public std::exception
{
    ColumnAmbiguityException(const std::string& column)
    : message_("Column: \"" + column + "\" was found in more than one table.")
    {
    }

    const char* what() const noexcept override
    {
        return message_.c_str();
    }

private:
    std::string message_;
};

struct ColumnAlreadyExistsException : public std::exception
{
    ColumnAlreadyExistsException(const std::string& column)
    : message_("Column: \"" + column + "\" already exists.")
    {
    }

    const char* what() const noexcept override
    {
        return message_.c_str();
    }

private:
    std::string message_;
};


struct ColumnAlreadyExistsInIndexException : public std::exception
{
    ColumnAlreadyExistsInIndexException(const std::string& column)
    : message_("Column: \"" + column + "\" already referenced multiple times in single index.")
    {
    }

    const char* what() const noexcept override
    {
        return message_.c_str();
    }

private:
    std::string message_;
};

struct ColumnNotFoundException : public std::exception
{
    ColumnNotFoundException(const std::string& column)
    : message_("Column: \"" + column + "\" was not found in table.")
    {
    }

    const char* what() const noexcept override
    {
        return message_.c_str();
    }

private:
    std::string message_;
};

struct IndexAlreadyExistsException : public std::exception
{
    IndexAlreadyExistsException(const std::string& index)
    : message_("Index: \"" + index + "\" already exists in table.")
    {
    }

    const char* what() const noexcept override
    {
        return message_.c_str();
    }

private:
    std::string message_;
};

struct ColumnGroupByException : public std::exception
{
    ColumnGroupByException(const std::string& column)
    : message_("Column: \"" + column + "\" must appear in GROUP BY clause or be used in AGGREGATE function.")
    {
    }

    const char* what() const noexcept override
    {
        return message_.c_str();
    }

private:
    std::string message_;
};

struct InsertIntoException : public std::exception
{
    const char* what() const noexcept override
    {
        return "There are several same referenced columns";
    }
};

struct NotSameAmoutOfValuesException : public std::exception
{
    const char* what() const noexcept override
    {
        return "Number of values provided must be the same as number of columns";
    }
};

struct NestedAggregationException : public std::exception
{
    const char* what() const noexcept override
    {
        return "Use of nested aggregation functions is not allowed.";
    }
};

struct AggregationOrderByException : public std::exception
{
    const char* what() const noexcept override
    {
        return "Use of aggregation functions in ORDER BY clause is not allowed.";
    }
};

struct AggregationWhereException : public std::exception
{
    const char* what() const noexcept override
    {
        return "Use of aggregation functions in WHERE clause is not allowed.";
    }
};


struct OrderByColumnAlreadyReferencedException : public std::exception
{
    const char* what() const noexcept override
    {
        return "The same column was referenced multiple times in ORDER BY clause.";
    }
};

struct RetPolygonGroupByException : public std::exception
{
    const char* what() const noexcept override
    {
        return "Return of complex polygon WKT is not allowed while using group by.";
    }
};

struct RetPointGroupByException : public std::exception
{
    const char* what() const noexcept override
    {
        return "Return of point WKT is not allowed while using group by.";
    }
};

struct RetStringGroupByException : public std::exception
{
    const char* what() const noexcept override
    {
        return "Return of string is not allowed while using group by.";
    }
};

struct StringGroupByException : public std::exception
{
    const char* what() const noexcept override
    {
        return "String is not allowed as a key while using group by.";
    }
};

struct AliasRedefinitionException : public std::exception
{
    const char* what() const noexcept override
    {
        return "Attempt to redefine an allias has occured.";
    }
};

struct NullMaskOperationInvalidOperandException : public std::exception
{
    const char* what() const noexcept override
    {
        return "Null mask operation can only be called with a column operand.";
    }
};

struct JoinColumnTypeException : public std::exception
{
    const char* what() const noexcept override
    {
        return "Attempt to join columns of different data types.";
    }
};

struct InvalidOperandsException : public std::exception
{
    InvalidOperandsException(const std::string& left, const std::string& right, const std::string& op)
    : message_("Invalid operands: " + left + " " + right + " for operation: " + op)
    {
    }

    const char* what() const noexcept override
    {
        return message_.c_str();
    }

private:
    std::string message_;
};


#endif // DROPDBASE_INSTAREA_PARSEREXCEPTIONS_H
