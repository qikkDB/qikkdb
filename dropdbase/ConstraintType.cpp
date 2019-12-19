#include "ConstraintType.h"
#include <unordered_map>

ConstraintType GetConstraintType(const std::string& constraintTypeName)
{
    std::string constraintTypeNameUpper = constraintTypeName;

    for (auto& c : constraintTypeNameUpper)
    {
        c = toupper(c);
    }

    const std::unordered_map<std::string, ConstraintType> constraintTypes = {
        {"INDEX", ConstraintType::CONSTRAINT_INDEX},
        {"NOT NULL", ConstraintType::CONSTRAINT_NOT_NULL},
        {"UNIQUE", ConstraintType::CONSTRAINT_UNIQUE}};

    if (constraintTypes.find(constraintTypeNameUpper) == constraintTypes.end())
    {
        return ConstraintType::CONSTRAINT_NONE;
    }
    return constraintTypes.at(constraintTypeNameUpper);
}

std::string GetConstraintTypeSuffix(ConstraintType constraintType)
{
    const std::unordered_map<ConstraintType, std::string> constraintSuffixes = {
        {ConstraintType::CONSTRAINT_NONE, ""},
        {ConstraintType::CONSTRAINT_INDEX, "_IC"},
        {ConstraintType::CONSTRAINT_NOT_NULL, "_NC"},
        {ConstraintType::CONSTRAINT_UNIQUE, "_UC"}};
    return constraintSuffixes.at(constraintType);
}
