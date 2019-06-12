
include_directories(../dropdbase)
set(TEST_FILES  ${CMAKE_CURRENT_SOURCE_DIR}/BlockTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/AllocatorTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/ColumnTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/CacheTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/DatabaseTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/TableTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/ClientHandlerTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/CSVDataImportTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/DispatcherTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/DispatcherGeoTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/DispatcherTestsRegression.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/ProtocolMessageTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/TCPServerTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/InsertIntoTests.cpp)

set(CUDA_TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/GPUFilterTests.cu
				${CMAKE_CURRENT_SOURCE_DIR}/GPUArtithmeticTests.cu
				${CMAKE_CURRENT_SOURCE_DIR}/GPUAggregationTests.cu
				${CMAKE_CURRENT_SOURCE_DIR}/GPUDateTests.cu
				${CMAKE_CURRENT_SOURCE_DIR}/GPUCompressionTests.cu
				${CMAKE_CURRENT_SOURCE_DIR}/GPUPolygonClippingTests.cu
				${CMAKE_CURRENT_SOURCE_DIR}/GPUReconstructTests.cu)

if(WIN32)
	set_source_files_properties(${SOURCE_FILES} main.cpp PROPERTIES COMPILE_FLAGS "/wd4251 /wd4275")
	set_source_files_properties(${CUDA_FILES} PROPERTIES COMPILE_FLAGS "-Xcudafe=\"--diag_suppress=field_without_dll_interface --diag_suppress=base_class_has_different_dll_interface\" -Xcompiler=\"/bigobj\"")
	set_source_files_properties(${TEST_FILES} PROPERTIES COMPILE_FLAGS "/wd4251 /wd4275")
	set_source_files_properties(${CUDA_TEST_FILES} PROPERTIES COMPILE_FLAGS "-Xcudafe=\"--diag_suppress=field_without_dll_interface --diag_suppress=base_class_has_different_dll_interface\" -Xcompiler=\"/bigobj\"")
endif(WIN32)
if(NOT WIN32)
	set_source_files_properties(${CUDA_FILES} PROPERTIES COMPILE_FLAGS "--compiler-bindir=${CMAKE_CXX_COMPILER}")
endif(NOT WIN32)
add_executable(${PROJECT_NAME}_test ${SOURCE_FILES} ${CUDA_FILES} ${TEST_FILES} ${CUDA_TEST_FILES})

target_include_directories(${PROJECT_NAME}_test PUBLIC ${GTEST_INCLUDE_DIR}) 

string( TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_lower )

target_link_libraries(${PROJECT_NAME}_test ${YAML_CPP_LIBRARIES})
target_link_libraries(${PROJECT_NAME}_test ${Boost_LIBRARIES})
target_link_libraries(${PROJECT_NAME}_test antlr4_static)




IF(CMAKE_BUILD_TYPE_lower STREQUAL "debug")
	target_link_libraries(${PROJECT_NAME}_test ${prefix}protobufd${suffix})
    target_link_libraries(${PROJECT_NAME}_test gtestd)
	target_link_libraries(${PROJECT_NAME}_test gtest_maind)
ELSE(CMAKE_BUILD_TYPE_lower STREQUAL "debug")
target_link_libraries(${PROJECT_NAME}_test ${prefix}protobuf${suffix})
	target_link_libraries(${PROJECT_NAME}_test gtest)
	target_link_libraries(${PROJECT_NAME}_test gtest_main)
ENDIF(CMAKE_BUILD_TYPE_lower STREQUAL "debug")

add_custom_command(
        TARGET ${PROJECT_NAME}_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_SOURCE_DIR}/csv_tests
                ${CMAKE_CURRENT_BINARY_DIR}/csv_tests)
