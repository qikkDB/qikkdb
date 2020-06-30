#pragma once
#include <string>
#include <stdexcept>

enum ConstraintViolationErrorType
{
    UNIQUE_CONSTRAINT_INSERT_NULL_VALUE,
    UNIQUE_CONSTRAINT_INSERT_DUPLICATE_VALUE,
    UNIQUE_CONSTRAINT_DROP_NOT_NULL
};

class constraint_error : public std::runtime_error
{
public:
    explicit constraint_error(const std::string& what_arg);
};


class constraint_violation_error : public constraint_error
{
private:
    ConstraintViolationErrorType constraintViolationErrorType_;

public:
    explicit constraint_violation_error(ConstraintViolationErrorType constraintViolationErrorType,
                                        const std::string& message);

    ConstraintViolationErrorType GetConstraintViolationError() const;
};