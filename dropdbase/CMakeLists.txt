cmake_minimum_required(VERSION 3.13)

set(PROJECT_NAME dropdbase_instarea)
project(${PROJECT_NAME})
#enable_language(CUDA)

set(CMAKE_CXX_STANDARD 14)

find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS Types/Point.proto)

find_package(gtest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

find_package(spdlog REQUIRED)
include_directories(${SPDLOG_INCLUDE_DIRS})


set(CMAKE_PREFIX_PATH /usr/local/lib/cmake/antlr4)
set(ANTLR4_JAR_LOCATION /usr/local/lib/antlr-4.7.2-complete.jar)

find_package(antlr4-runtime REQUIRED)
include_directories(${ANTLR4_INCLUDE_DIR})
find_package(antlr4-generator REQUIRED)

antlr4_generate(
        GpuSqlLexer
        GpuSqlParser/GpuSqlLexer.g4
        LEXER
        FALSE
        FALSE
        "gpuSql"
)

antlr4_generate(
        GpuSqlParser
        GpuSqlParser/GpuSqlParser.g4
        PARSER
        FALSE
        TRUE
        "gpuSql"
        "${ANTLR4_TOKEN_FILES_GpuSqlLexer}"
        "${ANTLR4_TOKEN_DIRECTORY_GpuSqlLexer}"
)

set(ANTLR_GENERATED
        GpuSqlParser/GpuSqlLexer.h
        GpuSqlParser/GpuSqlLexer.cpp
        GpuSqlParser/GpuSqlParser.h
        GpuSqlParser/GpuSqlParser.cpp
        GpuSqlParser/GpuSqlParserBaseListener.h
        GpuSqlParser/GpuSqlParserBaseListener.cpp
        GpuSqlParser/GpuSqlParserListener.h
        GpuSqlParser/GpuSqlParserListener.cpp)

include_directories(.)
include_directories(messages)
include_directories(Types)



add_executable(${PROJECT_NAME}
        messages/CSVImportMessage.pb.cc
        messages/CSVImportMessage.pb.h
        messages/InfoMessage.pb.cc
        messages/InfoMessage.pb.h
        messages/QueryMessage.pb.cc
        messages/QueryMessage.pb.h
        messages/QueryResponseMessage.pb.cc
        messages/QueryResponseMessage.pb.h
        messages/SetDatabaseMessage.pb.cc
        messages/SetDatabaseMessage.pb.h
        Types/ComplexPolygon.pb.cc
        Types/ComplexPolygon.pb.h
        Types/Point.pb.cc
        Types/Point.pb.h
        BlockBase.h
        ColumnBase.h
        ComplexPolygonFactory.cpp
        ComplexPolygonFactory.h
        IBlock.h
        main.cpp
        NativeGeoPoint.h
        PointFactory.cpp
        PointFactory.h
        GpuSqlParser/GpuSqlCustomParser.cpp
        GpuSqlParser/GpuSqlCustomParser.h
        Database.cpp
        Database.h
        GpuSqlParser/ParserExceptions.h
        GpuSqlParser/GpuSqlListener.cpp
        GpuSqlParser/GpuSqlListener.h
        GpuSqlParser/GpuSqlDispatcher.cpp
        GpuSqlParser/GpuSqlDispatcher.h
        GpuSqlParser/MemoryStream.h
        ${ANTLR_GENERATED} GpuSqlParser/MemoryStream.cpp)

target_link_libraries(${PROJECT_NAME} ${PROTOBUF_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${GTEST_BOTH_LIBRARIES})
target_link_libraries(${PROJECT_NAME} antlr4_static)
