cmake_minimum_required(VERSION 3.10)
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)
SET_DIRECTORY_PROPERTIES(PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}/third_party)
set(PROJECT_NAME dropdbase_instarea)
project(${PROJECT_NAME} LANGUAGES CXX CUDA)
enable_testing()
set(CMAKE_CXX_STANDARD 17)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(WIN32)
	add_definitions(-D_WIN32_WINNT=0x601)
endif(WIN32)

# Find include dirs
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

if(WIN32)
  set(prefix "lib")
  set(suffix ".lib")
elseif(APPLE)
  set(prefix "lib")
  set(suffix ".a")
else()
  set(prefix "lib")
  set(suffix ".a")
endif()


###### Google Protocol Buffers external project ########

set(PROTOBUF_TAR_GZ https://github.com/protocolbuffers/protobuf/archive/v3.6.1.tar.gz)

ExternalProject_Add(
  protobuf-external
  PREFIX protobuf
  URL ${PROTOBUF_TAR_GZ}
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/protobuf
  CMAKE_CACHE_ARGS
    "-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}"
    "-Dprotobuf_BUILD_TESTS:BOOL=OFF"
    "-Dprotobuf_BUILD_EXAMPLES:BOOL=OFF"
    "-Dprotobuf_WITH_ZLIB:BOOL=OFF"
	"-Dprotobuf_BUILD_PROTOC_BINARIES:BOOL=OFF"
	"-Dprotobuf_BUILD_SHARED_LIBS:BOOL=OFF"
	"-Dprotobuf_MSVC_STATIC_RUNTIME:BOOL=OFF"
    "-DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}"
    # other project specific parameters
  SOURCE_SUBDIR cmake
  BUILD_ALWAYS 1
  STEP_TARGETS build
  INSTALL_COMMAND ""
)

EXternalProject_Get_Property(protobuf-external source_dir)
include_directories(${source_dir}/src)
link_directories(${CMAKE_CURRENT_BINARY_DIR}/protobuf)

###### Google Protocol Buffers external project end ########

###### Antlr4 external project ########

add_definitions(-DANTLR4CPP_STATIC)
set(ANTLR4_WITH_STATIC_CRT OFF)
# add external build for antlrcpp
include(ExternalAntlr4Cpp)
# add antrl4cpp artifacts to project environment
include_directories(${ANTLR4_INCLUDE_DIRS})

###### Antlr4 external project ########

###### Google Test external project ########

ExternalProject_Add(
    googletest
    URL https://github.com/google/googletest/archive/release-1.8.1.zip
    # TIMEOUT 10
    # # Force separate output paths for debug and release builds to allow easy
    # # identification of correct lib in subsequent TARGET_LINK_LIBRARIES commands
    # CMAKE_ARGS -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG:PATH=DebugLibs
    #            -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE:PATH=ReleaseLibs
    #            -Dgtest_force_shared_crt=ON
    # Disable install step
	CMAKE_CACHE_ARGS
	  "-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}"
	  "-DBUILD_GMOCK:BOOL=OFF"
	  "-DINSTALL_GTEST:BOOL=OFF"
	  "-Dgtest_force_shared_crt:BOOL=ON"
    INSTALL_COMMAND ""
    # Wrap download, configure and build steps in a script to log output
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
LOG_BUILD ON)
ExternalProject_Get_Property(googletest source_dir)
set(GTEST_INCLUDE_DIR ${source_dir}/googletest/include)

# Library
ExternalProject_Get_Property(googletest binary_dir)
set(GTEST_MAIN_LIBRARY_PATH ${binary_dir}/googletest/${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main${suffix} )
set(GTEST_MAIN_LIBRARY gtest_main)
add_library(${GTEST_MAIN_LIBRARY} UNKNOWN IMPORTED)
set_property(TARGET ${GTEST_MAIN_LIBRARY} PROPERTY IMPORTED_LOCATION
                ${GTEST_MAIN_LIBRARY_PATH} )
add_dependencies(${GTEST_MAIN_LIBRARY} googletest)

set(GTEST_LIBRARY_PATH ${binary_dir}/googletest/${CMAKE_FIND_LIBRARY_PREFIXES}gtest${suffix} )
set(GTEST_LIBRARY gtest)
add_library(${GTEST_LIBRARY} UNKNOWN IMPORTED)
set_property(TARGET ${GTEST_LIBRARY} PROPERTY IMPORTED_LOCATION
                ${GTEST_LIBRARY_PATH} )
add_dependencies(${GTEST_LIBRARY} googletest)

###### Google Test external project ########


SET(Boost_USE_STATIC_LIBS ON)
find_package(Boost 1.66.0 COMPONENTS system filesystem log REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIRS})

include_directories(.)
include_directories(messages)
include_directories(Types)
include_directories(GpuSqlParser)

set(SOURCE_FILES messages/CSVImportMessage.pb.cc
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
	QueryEngine/CudaMemAllocator.cu
        BlockBase.h
        BlockBase.cpp
        ColumnBase.cpp
        ColumnBase.h
        ComplexPolygonFactory.cpp
        ComplexPolygonFactory.h
        NativeGeoPoint.h
        PointFactory.cpp
        PointFactory.h
        Database.cpp
        Database.h
        Configuration.cpp
        Configuration.h
        GpuSqlParser/GpuSqlCustomParser.cpp
        GpuSqlParser/GpuSqlCustomParser.h
        GpuSqlParser/ParserExceptions.h
        GpuSqlParser/GpuSqlListener.cpp
        GpuSqlParser/GpuSqlListener.h
        GpuSqlParser/GpuSqlDispatcher.cu
        GpuSqlParser/GpuSqlDispatcher.h
        GpuSqlParser/MemoryStream.h
        GpuSqlParser/MemoryStream.cpp
        GpuSqlParser/QueryType.h
        GpuSqlParser/GpuSqlLexer.cpp
        GpuSqlParser/GpuSqlLexer.h
        GpuSqlParser/GpuSqlParser.cpp
        GpuSqlParser/GpuSqlParser.h
        GpuSqlParser/GpuSqlParserBaseListener.cpp
        GpuSqlParser/GpuSqlParserBaseListener.h
        GpuSqlParser/GpuSqlParserListener.cpp
        GpuSqlParser/GpuSqlParserListener.h
        GpuSqlParser/GpuSqlParser.cpp
        GpuSqlParser/GpuSqlParser.h
        IColumn.h
        Table.cpp
        Table.h
        DatabaseGenerator.cpp
        DatabaseGenerator.h)

set(TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../dropdbase_test/BlockTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/../dropdbase_test/ColumnTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/../dropdbase_test/DatabaseTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/../dropdbase_test/DispatcherTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/../dropdbase_test/TableTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/../dropdbase_test/GPUFilterTests.cu
				${CMAKE_CURRENT_SOURCE_DIR}/../dropdbase_test/GPUArtithmeticTests.cu
				${CMAKE_CURRENT_SOURCE_DIR}/../dropdbase_test/GPUAggregationTests.cu)

add_executable(${PROJECT_NAME} ${SOURCE_FILES} main.cpp)
add_executable(${PROJECT_NAME}_test ${SOURCE_FILES} ${TEST_FILES})
target_include_directories(${PROJECT_NAME}_test PUBLIC ${GTEST_INCLUDE_DIR}) 

add_test(${PROJECT_NAME}_test ${PROJECT_NAME}_test)

target_link_libraries(${PROJECT_NAME} ${prefix}protobuf${suffix})
target_link_libraries(${PROJECT_NAME} ${YAML_CPP_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
target_link_libraries(${PROJECT_NAME} antlr4_static)

target_link_libraries(${PROJECT_NAME}_test ${prefix}protobuf${suffix})
target_link_libraries(${PROJECT_NAME}_test ${YAML_CPP_LIBRARIES})
target_link_libraries(${PROJECT_NAME}_test ${Boost_LIBRARIES})
target_link_libraries(${PROJECT_NAME}_test antlr4_static)
target_link_libraries(${PROJECT_NAME}_test gtest)
target_link_libraries(${PROJECT_NAME}_test gtest_main)

