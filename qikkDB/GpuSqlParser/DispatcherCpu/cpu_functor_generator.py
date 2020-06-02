INT = "int32_t"
LONG = "int64_t"
FLOAT = "float"
DOUBLE = "double"
POINT = "ColmnarDB::Types::Point"
POLYGON = "ColmnarDB::Types::ComplexPolygon"
STRING = "std::string"
BOOL = "int8_t"

types = [INT,
         LONG,
         FLOAT,
         DOUBLE,
         POINT,
         POLYGON,
         STRING,
         BOOL]
all_types = [INT,
             LONG,
             FLOAT,
             DOUBLE,
             POINT,
             POLYGON,
             STRING,
             BOOL,
             INT,
             LONG,
             FLOAT,
             DOUBLE,
             POINT,
             POLYGON,
             STRING,
             BOOL]

bitwise_operations = ["bitwiseOr", "bitwiseAnd", "bitwiseXor", "bitwiseLeftShift", "bitwiseRightShift"]
arithmetic_operations = ["mul", "div", "add", "sub", "mod", "logarithm", "power"]
unary_arithmetic_operations = ['minus', 'absolute', 'sine', 'cosine', 'tangent', 'cotangent', 'arcsine', 'arccosine',
                               'arctangent',
                               'logarithm10', 'logarithmNatural', 'exponential', 'squareRoot', 'square', 'sign',
                               'round', 'floor', 'ceil']
geo_operations = ["contains"]
polygon_operations = ["intersect", "union"]
filter_operations = ["greater", "less", "greaterEqual", "lessEqual", "equal", "notEqual"]
logical_operations = ["logicalAnd", "logicalOr"]

numeric_types = [INT, LONG, FLOAT, DOUBLE]
floating_types = [FLOAT, DOUBLE]
geo_types = [POINT, POLYGON]
bool_types = [BOOL]

operations_binary = ["greater", "less", "greaterEqual", "lessEqual", "equal", "notEqual", "logicalAnd", "logicalOr",
                     "mul", "div", "add", "sub", "mod", "contains", "intersect", "union"]
operations_filter = ["greater", "less", "greaterEqual", "lessEqual", "equal", "notEqual"]
operations_logical = ["logicalAnd", "logicalOr"]
operations_arithmetic = ["mul", "div", "add", "sub", "mod", "bitwiseOr", "bitwiseAnd", "bitwiseXor", "bitwiseLeftShift",
                         "bitwiseRightShift", "power", "logarithm", "arctangent2", "root", "roundDecimal"]
operations_unary = ["logicalNot", "minus", "min", "max", "sum", "count", "avg", "year", "month", "day", "hour",
                    "minute", "second"]
operations_aggregation = ["min", "max", "sum", "count", "avg"]
operations_date = ["year", "month", "day", "hour", "minute", "second"]
operations_move = ["load", "ret", "groupBy"]
operations_ternary = ["between"]

operation_binary_monotonous = ["greater", "less", "greaterEqual", "lessEqual", "equal", "notEqual"]
operation_arithmetic_monotonous = ["mul", "div", "add", "sub", "mod", "logarithm", "arctangent2"]
operation_arithmetic_non_monotonous = ["bitwiseOr", "bitwiseAnd", "bitwiseXor", "bitwiseLeftShift", "bitwiseRightShift",
                                       "power"]

operations_string_unary = ['ltrim', 'rtrim', 'lower', 'upper', 'reverse']

operations_string_binary = ['left', 'right']

for operation in ["whereResult"]:
    declaration = "std::array<CpuSqlDispatcher::CpuDispatchFunction," \
                  "DataType::DATA_TYPE_SIZE> CpuSqlDispatcher::" + operation + "Functions = {"

    for colIdx, colVal in enumerate(all_types):

        if colIdx < len(types):
            col = "Const"
        elif colIdx >= len(types):
            col = "Col"

        if colVal in [STRING, POINT, POLYGON]:
            op = "InvalidOperandTypesErrorHandler"

        else:
            op = operation
        function = "CpuSqlDispatcher::" + op + col + "<" + colVal + ">"

        if colIdx == len(all_types) - 1:
            declaration += ("&" + function + "};")
        else:
            declaration += ("&" + function + ", ")

    print(declaration)
print()

for operation in operations_binary:
    declaration = "std::array<CpuSqlDispatcher::CpuDispatchFunction," \
                  "DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> CpuSqlDispatcher::" + operation + "Functions = {"

    for colIdx, colVal in enumerate(all_types):
        for rowIdx, rowVal in enumerate(all_types):

            if colIdx < len(types):
                col = "Const"
            elif colIdx >= len(types):
                col = "Col"

            if rowIdx < len(types):
                row = "Const"
            elif rowIdx >= len(types):
                row = "Col"

            if operation != 'contains' and (colVal in geo_types or rowVal in geo_types):
                op = "InvalidOperandTypesErrorHandler"

            elif colVal == STRING or rowVal == STRING:
                op = "InvalidOperandTypesErrorHandler"

            elif operation in arithmetic_operations and (colVal == BOOL or rowVal == BOOL):
                op = "InvalidOperandTypesErrorHandler"

            elif operation == "mod" and (colVal in floating_types or rowVal in floating_types):
                op = "InvalidOperandTypesErrorHandler"

            elif operation == "contains" and (colVal != POLYGON or rowVal != POINT):
                op = "InvalidOperandTypesErrorHandler"

            else:
                op = operation
            function = "CpuSqlDispatcher::" + op + col + row + "<" + colVal + ", " + rowVal + ">"

            if colIdx == len(all_types) - 1 and rowIdx == len(all_types) - 1:
                declaration += ("&" + function + "};")
            else:
                declaration += ("&" + function + ", ")

    print(declaration)
print()
#
# for operation in operations_binary:
#     print("void add" + operation[0].upper() + operation[1:] + "Function(DataType left, DataType right);")
#
# for operation in operations_binary:
#     print(
#         "void CpuSqlDispatcher::add" + operation[0].upper() + operation[1:] + "Function(DataType left, DataType right)")
#     print('{')
#     print("\tdispatcherFunctions.push_back(" + operation + "Functions[DataType::DATA_TYPE_SIZE * left + right]);")
#     print('}')
#     print('\n')
#
# for operation in operations_unary:
#     print("static std::array<CpuSqlDispatcher::CpuDispatchFunction, " \
#           "DataType::DATA_TYPE_SIZE> " + operation + "Functions;")
#
# print('\n')
#
# for operation in operations_unary:
#     declaration = "std::array<CpuSqlDispatcher::CpuDispatchFunction, " \
#                   "DataType::DATA_TYPE_SIZE> CpuSqlDispatcher::" + operation + "Functions = {"
#
#     for colIdx, colVal in enumerate(all_types):
#
#         if colIdx < len(types):
#             col = "Const"
#         elif colIdx >= len(types):
#             col = "Col"
#
#         if colVal in geo_types or colVal == STRING:
#             op = "InvalidOperandTypesErrorHandler"
#         else:
#             op = operation
#         function = "CpuSqlDispatcher::" + op + col + "<" + colVal + ">"
#
#         if colIdx == len(all_types) - 1:
#             declaration += ("&" + function + "};")
#         else:
#             declaration += ("&" + function + ", ")
#
#     print(declaration)
# print()
#
# for operation in operations_unary:
#     print("void add" + operation[0].upper() + operation[1:] + "Function(DataType type);")
#
# for operation in operations_unary:
#     print(
#         "void CpuSqlDispatcher::add" + operation[0].upper() + operation[1:] + "Function(DataType type)")
#     print('{')
#     print("\tdispatcherFunctions.push_back(" + operation + "Functions[type]);")
#     print('}')
#     print('\n')
#
# print('\n')
#
# for operation in operations_move:
#     print("static std::array<CpuSqlDispatcher::CpuDispatchFunction, " \
#           "DataType::DATA_TYPE_SIZE> " + operation + "Functions;")
#
# print('\n')
#
# for operation in operations_move:
#     declaration = "std::array<CpuSqlDispatcher::CpuDispatchFunction, " \
#                   "DataType::DATA_TYPE_SIZE> CpuSqlDispatcher::" + operation + "Functions = {"
#
#     for colIdx, colVal in enumerate(all_types):
#
#         if colIdx < len(types):
#             col = "Const"
#         elif colIdx >= len(types):
#             col = "Col"
#
#         if (operation == 'groupBy') and (colVal == STRING or colVal == BOOL or colVal in geo_types):
#             function = "CpuSqlDispatcher::" + "InvalidOperandTypesErrorHandler" + col + "<" + colVal + ">"
#         elif (operation == 'ret') and (colVal == BOOL):
#             function = "CpuSqlDispatcher::" + "InvalidOperandTypesErrorHandler" + col + "<" + colVal + ">"
#         else:
#             function = "CpuSqlDispatcher::" + operation + col + "<" + colVal + ">"
#
#         if colIdx == len(all_types) - 1:
#             declaration += ("&" + function + "};")
#         else:
#             declaration += ("&" + function + ", ")
#
#     print(declaration)
# print()
#
# for operation in operations_move:
#     print("void add" + operation[0].upper() + operation[1:] + "Function(DataType type);")
#
# for operation in operations_move:
#     print(
#         "void CpuSqlDispatcher::add" + operation[0].upper() + operation[1:] + "Function(DataType type)")
#     print('{')
#     print("\tdispatcherFunctions.push_back(" + operation + "Functions[type]);")
#     print('}')
#     print('\n')

for operation in operations_filter:
    declaration = "std::array<CpuSqlDispatcher::CpuDispatchFunction," \
                  "DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> CpuSqlDispatcher::" + operation + "Functions = {"

    for colIdx, colVal in enumerate(all_types):
        for rowIdx, rowVal in enumerate(all_types):

            if colIdx < len(types):
                col = "Const"
            elif colIdx >= len(types):
                col = "Col"

            if rowIdx < len(types):
                row = "Const"
            elif rowIdx >= len(types):
                row = "Col"

            if colVal in geo_types or rowVal in geo_types:
                op = "InvalidOperandTypesErrorHandler"

            elif colVal == STRING and rowVal != STRING:
                op = "InvalidOperandTypesErrorHandler"

            elif colVal != STRING and rowVal == STRING:

                op = "InvalidOperandTypesErrorHandler"

            elif operation in arithmetic_operations and (colVal == BOOL or rowVal == BOOL):
                op = "InvalidOperandTypesErrorHandler"

            elif operation == "mod" and (colVal in floating_types or rowVal in floating_types):
                op = "InvalidOperandTypesErrorHandler"

            elif colVal == STRING and rowVal == STRING:
                op = "filterString"
            else:
                op = "filter"

            if op == "filterString":
                function = "CpuSqlDispatcher::" + op + col + row + "<FilterConditions::" + operation + ">"
            else:
                function = "CpuSqlDispatcher::" + op + col + row + "<FilterConditions::" + operation + ", " + colVal + ", " + rowVal + ">"

            if colIdx == len(all_types) - 1 and rowIdx == len(all_types) - 1:
                declaration += ("&" + function + "};")
            else:
                declaration += ("&" + function + ", ")

    print(declaration)
print()

for operation in operations_logical:
    declaration = "std::array<CpuSqlDispatcher::CpuDispatchFunction," \
                  "DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> CpuSqlDispatcher::" + operation + "Functions = {"

    for colIdx, colVal in enumerate(all_types):
        for rowIdx, rowVal in enumerate(all_types):

            if colIdx < len(types):
                col = "Const"
            elif colIdx >= len(types):
                col = "Col"

            if rowIdx < len(types):
                row = "Const"
            elif rowIdx >= len(types):
                row = "Col"

            if colVal in geo_types or rowVal in geo_types:
                op = "InvalidOperandTypesErrorHandler"

            elif colVal == STRING or rowVal == STRING:
                op = "InvalidOperandTypesErrorHandler"

            elif operation in arithmetic_operations and (colVal == BOOL or rowVal == BOOL):
                op = "InvalidOperandTypesErrorHandler"

            else:
                op = "logical"
            function = "CpuSqlDispatcher::" + op + col + row + "<LogicOperations::" + operation + ", " + colVal + ", " + rowVal + ">"

            if colIdx == len(all_types) - 1 and rowIdx == len(all_types) - 1:
                declaration += ("&" + function + "};")
            else:
                declaration += ("&" + function + ", ")

    print(declaration)
print()

for operation in operations_arithmetic:
    declaration = "std::array<CpuSqlDispatcher::CpuDispatchFunction," \
                  "DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> CpuSqlDispatcher::" + operation + "Functions_ = {"

    for colIdx, colVal in enumerate(all_types):
        for rowIdx, rowVal in enumerate(all_types):

            if colIdx < len(types):
                col = "Const"
            elif colIdx >= len(types):
                col = "Col"

            if rowIdx < len(types):
                row = "Const"
            elif rowIdx >= len(types):
                row = "Col"

            if colVal in geo_types or rowVal in geo_types:
                op = "InvalidOperandTypesErrorHandler"

            elif colVal == STRING or rowVal == STRING:
                op = "InvalidOperandTypesErrorHandler"

            elif colVal == BOOL or rowVal == BOOL:
                op = "InvalidOperandTypesErrorHandler"

            elif (operation == "mod" or operation in bitwise_operations) and (
                    colVal in floating_types or rowVal in floating_types):
                op = "InvalidOperandTypesErrorHandler"

            elif (operation == "roundDecimal") and (rowVal in floating_types):
                op = "InvalidOperandTypesErrorHandler"

            else:
                op = "arithmetic"

            function = "CpuSqlDispatcher::" + op + col + row + "<ArithmeticOperations::" + operation + "NoCheck, " + colVal + ", " + rowVal + ">"

            if colIdx == len(all_types) - 1 and rowIdx == len(all_types) - 1:
                declaration += ("&" + function + "};")
            else:
                declaration += ("&" + function + ", ")

    print(declaration)
print()

for operation in operations_date:
    declaration = "std::array<CpuSqlDispatcher::CpuDispatchFunction, " \
                  "DataType::DATA_TYPE_SIZE> CpuSqlDispatcher::" + operation + "Functions = {"

    for colIdx, colVal in enumerate(all_types):

        if colIdx < len(types):
            col = "Const"
        elif colIdx >= len(types):
            col = "Col"

        if colVal != LONG:
            op = "InvalidOperandTypesErrorHandler"
            function = "CpuSqlDispatcher::" + op + col + "<DateOperations::" + operation + ", " + colVal + ">"
        else:
            op = "dateExtract"
            function = "CpuSqlDispatcher::" + op + col + "<DateOperations::" + operation + ">"

        if colIdx == len(all_types) - 1:
            declaration += ("&" + function + "};")
        else:
            declaration += ("&" + function + ", ")

    print(declaration)
print()

for operation in unary_arithmetic_operations:
    declaration = "std::array<CpuSqlDispatcher::CpuDispatchFunction," \
                  "DataType::DATA_TYPE_SIZE> CpuSqlDispatcher::" + operation + "Functions = {"

    for colIdx, colVal in enumerate(all_types):

        if colIdx < len(types):
            col = "Const"
        elif colIdx >= len(types):
            col = "Col"

        if colVal in geo_types:
            op = "InvalidOperandTypesErrorHandler"

        elif colVal == STRING:
            op = "InvalidOperandTypesErrorHandler"

        elif colVal == BOOL:
            op = "InvalidOperandTypesErrorHandler"

        else:
            op = "arithmeticUnary"

        function = "CpuSqlDispatcher::" + op + col + "<ArithmeticUnaryOperations::" + operation + ", " + colVal + ">"

        if colIdx == len(all_types) - 1:
            declaration += ("&" + function + "};")
        else:
            declaration += ("&" + function + ", ")

    print(declaration)
print()

for opIdx, operation in enumerate(["castToInt", "castToLong", "castToFloat", "castToDouble", "castToPoint", "castToPolygon", "castToString", "castToInt8t"]):
    declaration = "std::array<CpuSqlDispatcher::DispatchFunction," \
                  "DataType::DATA_TYPE_SIZE> CpuSqlDispatcher::" + operation + "Functions = {"

    for colIdx, colVal in enumerate(all_types):

        if colIdx < len(types):
            col = "Const"
        elif colIdx >= len(types):
            col = "Col"

        if colVal in numeric_types and types[opIdx] in numeric_types:
            op = "castNumeric"
        else:
            op = "InvalidOperandTypesErrorHandler"

        function = "CpuSqlDispatcher::" + op + col + "<" + types[opIdx] + ", " + colVal + ">"

        if colIdx == len(all_types) - 1:
            declaration += ("&" + function + "};")
        else:
            declaration += ("&" + function + ", ")

    print(declaration)
print()

for operation in operations_string_unary:
    declaration = "std::array<CpuSqlDispatcher::CpuDispatchFunction," \
                  "DataType::DATA_TYPE_SIZE> CpuSqlDispatcher::" + operation + "Functions = {"

    for colIdx, colVal in enumerate(all_types):

        if colIdx < len(types):
            col = "Const"
        elif colIdx >= len(types):
            col = "Col"

        if colVal != STRING:
            op = "InvalidOperandTypesErrorHandler"
        else:
            op = "stringUnary"

        if op == "stringUnary":
            function = "CpuSqlDispatcher::" + op + col + "<StringUnaryOperationsCpu::" + operation + ">"

        else:
            function = "CpuSqlDispatcher::" + op + col + "<StringUnaryOperationsCpu::" + operation + ", " + colVal + ">"

        if colIdx == len(all_types) - 1:
            declaration += ("&" + function + "};")
        else:
            declaration += ("&" + function + ", ")

    print(declaration)
print()

for operation in ['len']:
    declaration = "std::array<CpuSqlDispatcher::CpuDispatchFunction," \
                  "DataType::DATA_TYPE_SIZE> CpuSqlDispatcher::" + operation + "Functions = {"

    for colIdx, colVal in enumerate(all_types):

        if colIdx < len(types):
            col = "Const"
        elif colIdx >= len(types):
            col = "Col"

        if colVal != STRING:
            op = "InvalidOperandTypesErrorHandler"
        else:
            op = "stringUnaryNumeric"

        if op == "stringUnaryNumeric":
            function = "CpuSqlDispatcher::" + op + col + "<StringUnaryOperationsCpu::" + operation + ">"

        else:
            function = "CpuSqlDispatcher::" + op + col + "<StringUnaryOperationsCpu::" + operation + ", " + colVal + ">"

        if colIdx == len(all_types) - 1:
            declaration += ("&" + function + "};")
        else:
            declaration += ("&" + function + ", ")

    print(declaration)
print()

for operation in operations_string_binary:
    declaration = "std::array<CpuSqlDispatcher::CpuDispatchFunction," \
                  "DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> CpuSqlDispatcher::" + operation + "Functions = {"

    for colIdx, colVal in enumerate(all_types):
        for rowIdx, rowVal in enumerate(all_types):

            if colIdx < len(types):
                col = "Const"
            elif colIdx >= len(types):
                col = "Col"

            if rowIdx < len(types):
                row = "Const"
            elif rowIdx >= len(types):
                row = "Col"

            if (operation == 'left' or operation == 'right') and (colVal != STRING or rowVal not in [INT, LONG]):
                op = "InvalidOperandTypesErrorHandler"

            else:
                op = "stringBinaryNumeric"

            if op == "stringBinaryNumeric":
                function = "CpuSqlDispatcher::" + op + col + row + "<StringBinaryOperationsCpu::" + operation + ", " + rowVal + ">"
            else:
                function = "CpuSqlDispatcher::" + op + col + row + "<StringBinaryOperationsCpu::" + operation + ", " + colVal + ", " + rowVal + ">"

            if colIdx == len(all_types) - 1 and rowIdx == len(all_types) - 1:
                declaration += ("&" + function + "};")
            else:
                declaration += ("&" + function + ", ")

    print(declaration)
print()

for operation in ["concat"]:
    declaration = "std::array<CpuSqlDispatcher::CpuDispatchFunction," \
                  "DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> CpuSqlDispatcher::" + operation + "Functions = {"

    for colIdx, colVal in enumerate(all_types):
        for rowIdx, rowVal in enumerate(all_types):

            if colIdx < len(types):
                col = "Const"
            elif colIdx >= len(types):
                col = "Col"

            if rowIdx < len(types):
                row = "Const"
            elif rowIdx >= len(types):
                row = "Col"

            if colVal != STRING or rowVal != STRING:
                op = "InvalidOperandTypesErrorHandler"

            else:
                op = "stringBinary"

            if op == "stringBinary":
                function = "CpuSqlDispatcher::" + op + col + row + "<StringBinaryOperationsCpu::" + operation + ">"
            else:
                function = "CpuSqlDispatcher::" + op + col + row + "<StringBinaryOperationsCpu::" + operation + ", " + colVal + ", " + rowVal + ">"

            if colIdx == len(all_types) - 1 and rowIdx == len(all_types) - 1:
                declaration += ("&" + function + "};")
            else:
                declaration += ("&" + function + ", ")

    print(declaration)
print()

#
# operation = "insertInto"
#
# declaration = "std::array<CpuSqlDispatcher::CpuDispatchFunction," \
#               "DataType::DATA_TYPE_SIZE> CpuSqlDispatcher::" + operation + "Functions = {"
#
# for colIdx, colVal in enumerate(all_types):
#     function = "CpuSqlDispatcher::" + operation + "<" + colVal + ">"
#
#     if colIdx == len(all_types) - 1:
#         declaration += ("&" + function + "};")
#     else:
#         declaration += ("&" + function + ", ")
#
# print(declaration)
# print()
#
# # Aggregations (without group by)
# for operation in operations_aggregation:
#     declaration = "std::array<CpuSqlDispatcher::CpuDispatchFunction, " \
#                   "DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> CpuSqlDispatcher::" + \
#                   operation + "AggregationFunctions = {"
#
#     for colIdx, colVal in enumerate(all_types):
#
#         if colIdx < len(types):
#             col = "Const"
#         elif colIdx >= len(types):
#             col = "Col"
#
#         if (colVal in geo_types and operation != "count") or (colVal == STRING) or (colVal == BOOL):
#             op = "InvalidOperandTypesErrorHandler"
#         else:
#             op = "aggregation"
#         retVal = colVal
#         if operation == "count":
#             retVal = LONG
#         # TODO: for avg FLOAT/DOUBLE
#         if op != "InvalidOperandTypesErrorHandler":
#             function = "CpuSqlDispatcher::" + op + col + "<AggregationFunctions::" + operation + ", " + retVal + ", " + colVal + ">"
#         else:
#             function = "CpuSqlDispatcher::" + op + col + "<AggregationFunctions::" + operation + ", " + colVal + ">"
#
#         if colIdx == len(all_types) - 1:
#             declaration += ("&" + function + "};")
#         else:
#             declaration += ("&" + function + ", ")
#
#     print(declaration)
# print()
#
# # Aggregations with group by
# for operation in operations_aggregation:
#     declaration = "std::array<CpuSqlDispatcher::CpuDispatchFunction, " \
#                   "DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> CpuSqlDispatcher::" + \
#                   operation + "GroupByFunctions = {"
#
#     for colIdx, colVal in enumerate(all_types):
#         for rowIdx, rowVal in enumerate(all_types):
#
#             if colIdx < len(types):
#                 col = "Const"
#             elif colIdx >= len(types):
#                 col = "Col"
#
#             if rowIdx < len(types):
#                 row = "Const"
#             elif rowIdx >= len(types):
#                 row = "Col"
#
#             if (col != "Col" or row != "Col") or \
#                     (colVal in geo_types or colVal == STRING) or \
#                     (rowVal in geo_types or rowVal == STRING) or (
#                     rowVal == BOOL or colVal == BOOL):
#                 op = "InvalidOperandTypesErrorHandler"
#             else:
#                 op = "aggregationGroupBy"
#             retVal = colVal
#             if operation == "count":
#                 retVal = LONG
#             # TODO: for avg FLOAT/DOUBLE
#             if op != "InvalidOperandTypesErrorHandler":
#                 function = "CpuSqlDispatcher::" + op + "<AggregationFunctions::" + operation + ", " + retVal + ", " + colVal + ", " + rowVal + ">"
#             else:
#                 function = "CpuSqlDispatcher::" + op + col + row + "<AggregationFunctions::" + operation + ", " + colVal + ", " + rowVal + ">"
#
#             if colIdx == len(all_types) - 1 and rowIdx == len(all_types) - 1:
#                 declaration += ("&" + function + "};")
#             else:
#                 declaration += ("&" + function + ", ")
#
#     print(declaration)
# print()
#

#
# for operation in polygon_operations:
#     declaration = "std::array<CpuSqlDispatcher::CpuDispatchFunction," \
#                   "DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> CpuSqlDispatcher::" + operation + "Functions = {"
#
#     for colIdx, colVal in enumerate(all_types):
#         for rowIdx, rowVal in enumerate(all_types):
#
#             if colIdx < len(types):
#                 col = "Const"
#             elif colIdx >= len(types):
#                 col = "Col"
#
#             if rowIdx < len(types):
#                 row = "Const"
#             elif rowIdx >= len(types):
#                 row = "Col"
#
#             if colVal != POLYGON or rowVal != POLYGON:
#                 op = "InvalidOperandTypesErrorHandler"
#
#             else:
#                 op = "polygonOperation"
#             function = "CpuSqlDispatcher::" + op + col + row + "<PolygonFunctions::poly" + operation.capitalize() + ", " + colVal + ", " + rowVal + ">"
#
#             if colIdx == len(all_types) - 1 and rowIdx == len(all_types) - 1:
#                 declaration += ("&" + function + "};")
#             else:
#                 declaration += ("&" + function + ", ")
#
#     print(declaration)
# print()
#
# operation = "point"
# declaration = "std::array<CpuSqlDispatcher::CpuDispatchFunction," \
#               "DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> CpuSqlDispatcher::" + operation + "Functions = {"
#
# for colIdx, colVal in enumerate(all_types):
#     for rowIdx, rowVal in enumerate(all_types):
#
#         if colIdx < len(types):
#             col = "Const"
#         elif colIdx >= len(types):
#             col = "Col"
#
#         if rowIdx < len(types):
#             row = "Const"
#         elif rowIdx >= len(types):
#             row = "Col"
#
#         if col == "Const" and row == "Const":
#             op = "InvalidOperandTypesErrorHandler"
#         elif colVal not in numeric_types or rowVal not in numeric_types:
#             op = "InvalidOperandTypesErrorHandler"
#         else:
#             op = operation
#
#         function = "CpuSqlDispatcher::" + op + col + row + "<" + colVal + ", " + rowVal + ">"
#
#         if colIdx == len(all_types) - 1 and rowIdx == len(all_types) - 1:
#             declaration += ("&" + function + "};")
#         else:
#             declaration += ("&" + function + ", ")
#
# print(declaration)
# print()
