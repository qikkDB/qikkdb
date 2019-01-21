types = ["int",
         "long",
         "float",
         "double",
         "ColmnarDB::Types::Point",
         "ColmnarDB::Types::Polygon",
         "std::string",
         "bool"]
all_types = ["int",
             "long",
             "float",
             "double",
             "ColmnarDB::Types::Point",
             "ColmnarDB::Types::Polygon",
             "std::string",
             "bool",
             "int",
             "long",
             "float",
             "double",
             "ColmnarDB::Types::Point",
             "ColmnarDB::Types::Polygon",
             "std::string",
             "bool",
             "unsigned char"]

arithmetic_operations = ["mul", "div", "add", "sub", "mod"]
geo_operations = ["contains"]
filter_operations = ["greater", "less", "greaterEqual", "lessEqual", "equal", "notEqual"]
logical_operations = ["logicalAnd", "logicalOr"]

numeric_types = ["int", "long", "float", "double"]
geo_types = ["ColmnarDB::Types::Point", "ColmnarDB::Types::Polygon"]
bool_types = ["unsigned char", "bool"]

operations = ["greater", "less", "greaterEqual", "lessEqual", "equal", "notEqual", "logicalAnd", "logicalOr", "mul",
              "div", "add", "sub", "mod", "contains"]

for operation in operations:
    declaration = "std::array<std::function<void(GpuSqlDispatcher &)>," \
                  "DataType::DATA_TYPE_SIZE * DATA_TYPE_SIZE> GpuSqlDispatcher::" + operation + "Functions = {"

    for colIdx, colVal in enumerate(all_types):
        for rowIdx, rowVal in enumerate(all_types):

            if colIdx < len(types):
                col = "Const"
            elif colIdx >= len(types) and colVal != "unsigned char":
                col = "Col"
            else:
                col = "Reg"

            if rowIdx < len(types):
                row = "Const"
            elif rowIdx >= len(types) and rowVal != "unsigned char":
                row = "Col"
            else:
                row = "Reg"

            if row == "Reg" and col == "Reg":
                function = operation + col + row
            else:
                if row == "Reg" or col == "Reg":
                    op = "invalidOperandTypesErrorHandler"
                else:
                    op = operation
                function = op + col + row + "<" + colVal + ", " + rowVal + ">"

            if colIdx == len(all_types) - 1 and rowIdx == len(all_types) - 1:
                declaration += ("&" + function + "};")
            else:
                declaration += ("&" + function + ", ")

    print(declaration)
