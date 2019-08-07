
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
				${CMAKE_CURRENT_SOURCE_DIR}/DispatcherNullTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/DispatcherGeoTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/DispatcherGroupByTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/DispatcherTestsRegression.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/ProtocolMessageTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/TCPServerTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/InsertIntoTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/GPUCompressionTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/OrderByMergeTests.cpp
				)

set(CUDA_TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/GPUFilterTests.cu
				${CMAKE_CURRENT_SOURCE_DIR}/GPUArtithmeticTests.cu
				${CMAKE_CURRENT_SOURCE_DIR}/GPUAggregationTests.cu
				${CMAKE_CURRENT_SOURCE_DIR}/GPUDateTests.cu
				${CMAKE_CURRENT_SOURCE_DIR}/GPUJoinTests.cu
				${CMAKE_CURRENT_SOURCE_DIR}/GPUGroupByTests.cu
				${CMAKE_CURRENT_SOURCE_DIR}/GPUPolygonClippingTests.cu
				${CMAKE_CURRENT_SOURCE_DIR}/GPUReconstructTests.cu
				${CMAKE_CURRENT_SOURCE_DIR}/GPUOrderByTests.cu)

if(WIN32)
	set_source_files_properties(${TEST_FILES} PROPERTIES COMPILE_FLAGS "/wd4251 /wd4275 /wd4244 /wd4267 /wd4305")
	set_source_files_properties(${CUDA_TEST_FILES} PROPERTIES COMPILE_FLAGS "-arch sm_60 -Xcudafe=\"--diag_suppress=field_without_dll_interface --diag_suppress=base_class_has_different_dll_interface\" -Xcompiler=\"/bigobj /wd4251 /wd4275 /wd4244 /wd4267 /wd4305\" --expt-relaxed-constexpr")
endif(WIN32)
if(NOT WIN32)
	set_source_files_properties(${CUDA_TEST_FILES} PROPERTIES COMPILE_FLAGS "-arch sm_60 --compiler-bindir=${CMAKE_CXX_COMPILER} --expt-relaxed-constexpr")
endif(NOT WIN32)
add_executable(${PROJECT_NAME}_test ${TEST_FILES} ${CUDA_TEST_FILES})
set_target_properties(${PROJECT_NAME}_test PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${PROJECT_NAME}_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
if(NOT WIN32)
	set_target_properties(${PROJECT_NAME}_test PROPERTIES LINK_FLAGS_DEBUG "-rdynamic")
endif(NOT WIN32)
target_include_directories(${PROJECT_NAME}_test PUBLIC ${GTEST_INCLUDE_DIR}) 

string( TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_lower )

IF(CMAKE_BUILD_TYPE_lower STREQUAL "debug")
    target_link_libraries(${PROJECT_NAME}_test gtestd)
	target_link_libraries(${PROJECT_NAME}_test gtest_maind)
ELSE(CMAKE_BUILD_TYPE_lower STREQUAL "debug")
	target_link_libraries(${PROJECT_NAME}_test gtest)
	target_link_libraries(${PROJECT_NAME}_test gtest_main)
ENDIF(CMAKE_BUILD_TYPE_lower STREQUAL "debug")

target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME}_obj)


add_custom_command(
        TARGET ${PROJECT_NAME}_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_SOURCE_DIR}/csv_tests
                ${CMAKE_CURRENT_BINARY_DIR}/csv_tests)
