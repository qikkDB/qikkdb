cmake_minimum_required(VERSION 3.15)
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)
SET_DIRECTORY_PROPERTIES(PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}/third_party)
set(PROJECT_NAME qikkDB)
project(${PROJECT_NAME} LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(GetGitRevisionDescription)

get_git_head_revision(GIT_BRANCH GIT_COMMIT_HASH)
git_describe(GIT_VERSION --tags)
if(GIT_VERSION MATCHES "^R[0-9]+\.[0-9]+$")
  set(GIT_BRANCH "")
else(GIT_VERSION MATCHES "^R[0-9]+\.[0-9]+$")
  string(REGEX REPLACE "^refs/heads/" " " GIT_BRANCH "${GIT_BRANCH}")
  if("${GIT_BRANCH}" STREQUAL "")
  set(GIT_BRANCH "detached")
  endif("${GIT_BRANCH}" STREQUAL "")
endif(GIT_VERSION MATCHES "^R[0-9]+\.[0-9]+$")
git_local_changes(GIT_IS_DIRTY)
if(GIT_IS_DIRTY STREQUAL "DIRTY")
  set(GIT_VERSION ${GIT_VERSION}-dirty)
endif(GIT_IS_DIRTY STREQUAL "DIRTY")

configure_file(
  ${CMAKE_SOURCE_DIR}/qikkDB/Version.h.in
  ${CMAKE_BINARY_DIR}/generated/Version.h
)

include_directories(${CMAKE_BINARY_DIR}/generated)

if(WIN32)
	add_definitions(-D_WIN32_WINNT=0x601 -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS -D_CRT_SECURE_NO_WARNINGS)
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

set(PROTOBUF_TAR_GZ https://github.com/protocolbuffers/protobuf/archive/v3.10.0.tar.gz)

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
	"-Dprotobuf_MSVC_STATIC_RUNTIME:BOOL=ON"
    "-DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}"
	"-DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}"
	"-DCMAKE_MAKE_PROGRAM:STRING=${CMAKE_MAKE_PROGRAM}"
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

###### YAML-CPP external project ########
set(YAML_CPP_TAR_GZ https://github.com/jbeder/yaml-cpp/archive/yaml-cpp-0.6.3.tar.gz)

ExternalProject_Add(
  yaml-cpp-external
  PREFIX yaml-cpp
  URL ${YAML_CPP_TAR_GZ}
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/yaml-cpp
  CMAKE_CACHE_ARGS
    "-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}"
    "-DYAML_MSVC_SHARED_RT:BOOL=OFF"
    "-DYAML_CPP_BUILD_TESTS:BOOL=OFF"
    # other project specific parameters
  BUILD_ALWAYS 1
  STEP_TARGETS build
  INSTALL_COMMAND ""
)

EXternalProject_Get_Property(yaml-cpp-external source_dir)
include_directories(${source_dir}/include)
link_directories(${CMAKE_CURRENT_BINARY_DIR}/yaml-cpp)

###### YAML-CPP external project end ########

###### JSONCPP external project ########
string(TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_upper )
set(JSONCPP_TAR_GZ https://github.com/open-source-parsers/jsoncpp/archive/1.9.3.tar.gz)

set(JSONCPP_CMAKE_CACHE_ARGS
    "-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}"
    "-DBUILD_STATIC_LIBS:BOOL=ON"
    "-DBUILD_SHARED_LIBS:BOOL=OFF" 
    "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY:STRING=${CMAKE_CURRENT_BINARY_DIR}/jsoncpp"
    "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY:STRING=${CMAKE_CURRENT_BINARY_DIR}/jsoncpp"
    "-DJSONCPP_WITH_TESTS:BOOL=OFF"
    "-DJSONCPP_WITH_POST_BUILD_UNITTEST:BOOL=OFF"
    "-DJSONCPP_WITH_PKGCONFIG_SUPPORT:BOOL=OFF"
    "-DJSONCPP_WITH_CMAKE_PACKAGE:BOOL=OFF")

set(JSONCPP_CMAKE_ARGS "")

IF(MSVC)
    set(JSONCPP_CMAKE_ARGS
    "-DCMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_upper}=${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_upper}} /MT$<$<CONFIG:Debug>:d>"
    "-DCMAKE_C_FLAGS_${CMAKE_BUILD_TYPE_upper}=${CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE_upper}} /MT$<$<CONFIG:Debug>:d>"
    "-DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=MultiThreaded$<$<CONFIG:Debug>:Debug>")
ENDIF(MSVC)

ExternalProject_Add(
  jsoncpp-external
  PREFIX jsoncpp
  URL ${JSONCPP_TAR_GZ}
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/jsoncpp
  CMAKE_ARGS
    ${JSONCPP_CMAKE_ARGS}
    # other project specific parameters
  CMAKE_CACHE_ARGS
    ${JSONCPP_CMAKE_CACHE_ARGS}
  BUILD_ALWAYS 1
  STEP_TARGETS build
  INSTALL_COMMAND ""
)

EXternalProject_Get_Property(jsoncpp-external source_dir)
include_directories(${source_dir}/include)
link_directories(${CMAKE_CURRENT_BINARY_DIR}/jsoncpp)

###### JSONCPP external project end ########

###### Antlr4 external project ########

add_definitions(-DANTLR4CPP_STATIC)
set(ANTLR4_WITH_STATIC_CRT ON)
# add external build for antlrcpp
include(ExternalAntlr4Cpp)
# add antrl4cpp artifacts to project environment
include_directories(${ANTLR4_INCLUDE_DIRS})

###### Antlr4 external project ########

###### Google Test external project ########
string( TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_lower )

IF(CMAKE_BUILD_TYPE_lower STREQUAL "debug")
	message("DEBUG MODE")
	set(GTEST_MAIN_BUILD_PATH third_party/src/googletest-build/googletest/${CMAKE_FIND_LIBRARY_PREFIXES}gtest_maind${suffix} )
    set(GTEST_BUILD_PATH third_party/src/googletest-build/googletest/${CMAKE_FIND_LIBRARY_PREFIXES}gtestd${suffix} )
ELSE(CMAKE_BUILD_TYPE_lower STREQUAL "debug")
	set(GTEST_MAIN_BUILD_PATH third_party/src/googletest-build/googletest/${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main${suffix} )
	set(GTEST_BUILD_PATH third_party/src/googletest-build/googletest/${CMAKE_FIND_LIBRARY_PREFIXES}gtest${suffix} )
ENDIF(CMAKE_BUILD_TYPE_lower STREQUAL "debug")

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
	  "-Dgtest_force_shared_crt:BOOL=OFF"
	  "-DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}"
	  "-DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}"
	  "-DCMAKE_MAKE_PROGRAM:STRING=${CMAKE_MAKE_PROGRAM}"
    INSTALL_COMMAND ""
    # Wrap download, configure and build steps in a script to log output
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
	LOG_BUILD ON
	BUILD_BYPRODUCTS ${GTEST_MAIN_BUILD_PATH} ${GTEST_BUILD_PATH}
)
ExternalProject_Get_Property(googletest source_dir)
set(GTEST_INCLUDE_DIR ${source_dir}/googletest/include)

# Library
ExternalProject_Get_Property(googletest binary_dir)


IF(CMAKE_BUILD_TYPE_lower STREQUAL "debug")
	message("DEBUG MODE")
	set(GTEST_MAIN_LIBRARY_PATH ${binary_dir}/googletest/${CMAKE_FIND_LIBRARY_PREFIXES}gtest_maind${suffix} )
    set(GTEST_MAIN_LIBRARY gtest_maind)
ELSE(CMAKE_BUILD_TYPE_lower STREQUAL "debug")
	set(GTEST_MAIN_LIBRARY_PATH ${binary_dir}/googletest/${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main${suffix} )
	set(GTEST_MAIN_LIBRARY gtest_main)
ENDIF(CMAKE_BUILD_TYPE_lower STREQUAL "debug")
add_library(${GTEST_MAIN_LIBRARY} UNKNOWN IMPORTED)
set_property(TARGET ${GTEST_MAIN_LIBRARY} PROPERTY IMPORTED_LOCATION
                ${GTEST_MAIN_LIBRARY_PATH} )
add_dependencies(${GTEST_MAIN_LIBRARY} googletest)

IF(CMAKE_BUILD_TYPE_lower STREQUAL "debug")
	set(GTEST_LIBRARY_PATH ${binary_dir}/googletest/${CMAKE_FIND_LIBRARY_PREFIXES}gtestd${suffix} )
    set(GTEST_LIBRARY gtestd)
ELSE(CMAKE_BUILD_TYPE_lower STREQUAL "debug")
	set(GTEST_LIBRARY_PATH ${binary_dir}/googletest/${CMAKE_FIND_LIBRARY_PREFIXES}gtest${suffix} )
	set(GTEST_LIBRARY gtest)
ENDIF(CMAKE_BUILD_TYPE_lower STREQUAL "debug")
add_library(${GTEST_LIBRARY} UNKNOWN IMPORTED)
set_property(TARGET ${GTEST_LIBRARY} PROPERTY IMPORTED_LOCATION
                ${GTEST_LIBRARY_PATH} )
add_dependencies(${GTEST_LIBRARY} googletest)

###### Google Test external project ########

set(Boost_USE_STATIC_LIBS ON)
IF(MSVC)
	set(Boost_USE_STATIC_RUNTIME ON)
ELSE(MSVC)
	set(Boost_USE_STATIC_RUNTIME OFF)
ENDIF(MSVC)
find_package(Boost 1.66.0 COMPONENTS log_setup log system filesystem iostreams REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

include_directories(.)
include_directories(messages)
include_directories(Types)
include_directories(GpuSqlParser)

add_subdirectory(qikkDB)
add_subdirectory(qikkDB_test)
add_custom_target(make-databases-dir ALL COMMAND ${CMAKE_COMMAND} -E make_directory  "${CMAKE_CURRENT_BINARY_DIR}/databases")
add_custom_target(make-test-databases-dir ALL COMMAND ${CMAKE_COMMAND} -E make_directory  "${CMAKE_CURRENT_BINARY_DIR}/test_databases")
add_custom_target(copy-default-config ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_SOURCE_DIR}/configuration
                ${CMAKE_CURRENT_BINARY_DIR}/configuration)

enable_testing()
add_test(NAME ${PROJECT_NAME}_test COMMAND ${PROJECT_NAME}_test)


