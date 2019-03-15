//
// Created by Martin Staňo on 2019-01-15.
//

#ifndef DROPDBASE_INSTAREA_PARSEREXCEPTIONS_H
#define DROPDBASE_INSTAREA_PARSEREXCEPTIONS_H

#include <exception>

struct DatabaseNotFoundException : public std::exception
{
    const char *what() const noexcept override
    {
        return "Database was not found.";
    }
};

struct TableNotFoundFromException : public std::exception
{
    const char *what() const noexcept override
    {
        return "Table was not found in FROM clause.";
    }
};

struct ColumnAmbiguityException : public std::exception
{
    const char *what() const noexcept override
    {
        return "Column was found in more than one table.";
    }
};

struct ColumnNotFoundException : public std::exception
{
    const char *what() const noexcept override
    {
        return "Column was not found in table.";
    }
};

struct ColumnGroupByException : public std::exception
{
    const char *what() const noexcept override
    {
        return "Column must appear in GROUP BY clause or be used in AGGREGATE function.";
    }
};

struct InsertIntoException : public std::exception
{
	const char *what() const noexcept override
	{
		return "There are several same referenced columns";
	}
};

struct NotSameAmoutOfValuesException : public std::exception
{
	const char *what() const noexcept override
	{
		return "Number of values provided must be the same as number of columns";
	}
};

struct NestedAggregationException : public std::exception
{
	const char *what() const noexcept override
	{
		return "Use of nested aggregation functions is not allowed.";
	}
};

struct RetPolygonGroupByException : public std::exception
{
	const char *what() const noexcept override
	{
		return "Return of complex polygon WKT is not allowed while using group by.";
	}
};

struct RetPointGroupByException : public std::exception
{
	const char *what() const noexcept override
	{
		return "Return of point WKT is not allowed while using group by.";
	}
};

struct RetStringGroupByException : public std::exception
{
	const char *what() const noexcept override
	{
		return "Return of string is not allowed while using group by.";
	}
};

#endif //DROPDBASE_INSTAREA_PARSEREXCEPTIONS_H
