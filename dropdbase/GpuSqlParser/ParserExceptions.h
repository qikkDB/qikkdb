//
// Created by Martin Staňo on 2019-01-15.
//

#ifndef DROPDBASE_INSTAREA_PARSEREXCEPTIONS_H
#define DROPDBASE_INSTAREA_PARSEREXCEPTIONS_H

#include <exception>
#include <string>

struct DatabaseNotFoundException : public std::exception
{
    const char* what() const noexcept override
    {
        return "Database was not found.";
    }
};

struct DatabaseAlreadyExistsException : public std::exception
{
	const char* what() const noexcept override
	{
		return "Database already exists.";
	}
};

struct TableNotFoundFromException : public std::exception
{
    const char* what() const noexcept override
    {
        return "Table was not found in FROM clause.";
    }
};

struct TableAlreadyExistsException : public std::exception
{
	const char* what() const noexcept override
	{
		return "Table already exists.";
	}
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
    const char* what() const noexcept override
    {
        return "Column was found in more than one table.";
    }
};

struct ColumnAlreadyExistsException : public std::exception
{
	const char* what() const noexcept override
	{
		return "Column already exists.";
	}
};

struct ColumnAlreadyExistsInIndexException : public std::exception
{
	const char* what() const noexcept override
	{
		return "Column already referenced multiple times in single index.";
	}
};

struct ColumnNotFoundException : public std::exception
{
    const char* what() const noexcept override
    {
        return "Column was not found in table.";
    }
};

struct IndexAlreadyExistsException : public std::exception
{
	const char* what() const noexcept override
	{
		return "Index already exists in table.";
	}
};

struct ColumnGroupByException : public std::exception
{
    const char* what() const noexcept override
    {
        return "Column must appear in GROUP BY clause or be used in AGGREGATE function.";
    }
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

struct InvalidOperandsException : public std::exception
{
	InvalidOperandsException(const std::string& left, const std::string& right, const std::string &op) : 
		message_("Invalid operands: " + left + " " + right + " for operation: " + op)
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
