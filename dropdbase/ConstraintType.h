#pragma once
#include <string>

enum ConstraintType
{
    CONSTRAINT_NONE = 0,
    CONSTRAINT_INDEX = 1,
    CONSTRAINT_UNIQUE = 2,
    CONSTRAINT_NOT_NULL = 3
};

ConstraintType GetConstraintType(const std::string& constraintTypeName);