#pragma once
#include <string>

enum ConstraintType
{
    CONSTRAINT_NONE = 0,
    CONSTRAINT_INDEX = 1,
    CONSTRAINT_NOT_NULL = 2,
    CONSTRAINT_UNIQUE = 3,
};

ConstraintType GetConstraintType(const std::string& constraintTypeName);
std::string GetConstraintTypeSuffix(ConstraintType constraintType);