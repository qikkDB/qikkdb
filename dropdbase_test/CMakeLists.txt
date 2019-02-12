
set(TEST_FILES  ${CMAKE_CURRENT_SOURCE_DIR}/BlockTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/ColumnTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/DatabaseTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/DispatcherTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/TableTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/ClientHandlerTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/ColumnTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/DatabaseTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/DispatcherTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/ProtocolMessageTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/TableTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/TCPServerTests.cpp)

set(CUDA_TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/GPUFilterTests.cu
				${CMAKE_CURRENT_SOURCE_DIR}/GPUArtithmeticTests.cu
				${CMAKE_CURRENT_SOURCE_DIR}/GPUAggregationTests.cu)

if(WIN32)
	set_source_files_properties(${SOURCE_FILES} main.cpp PROPERTIES COMPILE_FLAGS "/wd4251 /wd4275")
	set_source_files_properties(${CUDA_FILES} PROPERTIES COMPILE_FLAGS "-Xcudafe=\"--diag_suppress=field_without_dll_interface --diag_suppress=base_class_has_different_dll_interface\"")
	set_source_files_properties(${TEST_FILES} PROPERTIES COMPILE_FLAGS "/wd4251 /wd4275")
	set_source_files_properties(${CUDA_TEST_FILES} PROPERTIES COMPILE_FLAGS "-Xcudafe=\"--diag_suppress=field_without_dll_interface --diag_suppress=base_class_has_different_dll_interface\"")
endif(WIN32)

add_executable(${PROJECT_NAME}_test ${SOURCE_FILES} ${CUDA_FILES} ${TEST_FILES} ${CUDA_TEST_FILES})

target_include_directories(${PROJECT_NAME}_test PUBLIC ${GTEST_INCLUDE_DIR}) 


target_link_libraries(${PROJECT_NAME}_test ${prefix}protobuf${suffix})
target_link_libraries(${PROJECT_NAME}_test ${YAML_CPP_LIBRARIES})
target_link_libraries(${PROJECT_NAME}_test ${Boost_LIBRARIES})
target_link_libraries(${PROJECT_NAME}_test antlr4_static)
target_link_libraries(${PROJECT_NAME}_test gtest)
target_link_libraries(${PROJECT_NAME}_test gtest_main)