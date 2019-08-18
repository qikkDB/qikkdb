# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)
SET_DIRECTORY_PROPERTIES(PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}/third_party)
project ("TellStoryDB.NetworkClient")
set(CMAKE_CXX_STANDARD 17)

if(WIN32)
	add_definitions(-D_WIN32_WINNT=0x601 -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS -D_CRT_SECURE_NO_WARNINGS)
endif(WIN32)
add_definitions(-DBOOST_ALL_DYN_LINK)


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

set(PROTOBUF_TAR_GZ https://github.com/protocolbuffers/protobuf/archive/v3.8.0.tar.gz)

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
string( TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_lower )
find_package(Boost 1.66.0 COMPONENTS system iostreams REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
# Include sub-projects.
add_subdirectory ("TellStoryDB.NetworkClient")
