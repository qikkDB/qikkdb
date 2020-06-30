#include "ConstraintViolationError.h"

constraint_violation_error::constraint_violation_error(ConstraintViolationErrorType constraintViolationErrorType,
                                                       const std::string& message)
: constraint_error("Constraint Error " + std::to_string(constraintViolationErrorType) +
                   (message.size() > 0 ? (": " + message) : "")),
  constraintViolationErrorType_(constraintViolationErrorType)
{
}

ConstraintViolationErrorType constraint_violation_error::GetConstraintViolationError() const
{
    return constraintViolationErrorType_;
}

constraint_error::constraint_error(const std::string& what_arg) : runtime_error(what_arg)
{
}
