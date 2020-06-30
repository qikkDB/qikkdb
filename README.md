# qikkDB

GPU accelerated columnar database, delivering stellar performance for complex polygon operations and big data analytics. When you count your data in billions and want to see real-time results you need qikkDB. See also the project [website](https://qikk.ly/) and [documentation](https://docs.qikk.ly/).

# License
This project is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

# Instalation
We support Windows and Linux operating systems.

## Linux
You can use an [installation script](https://docs.qikk.ly/installation-getting-started#linux-installation-script) or [dockerfile](https://docs.qikk.ly/installation-getting-started#linux-deployment-via-docker).

## Windows
We provide [installation wizard](https://docs.qikk.ly/installation-getting-started#windows-installation-wizard) for Windows.

# Using
Learn how to [start a database](https://docs.qikk.ly/installation-getting-started#starting-database) and how to execute a [first query](https://docs.qikk.ly/installation-getting-started#first-query).

# Contributing
We welcome your contribution. Please create a pull request with your changes.

## Testing
We use [Google Tests](https://github.com/google/googletest) as testing framework. There are hundreds of unit tests and tens of integration tests in the project. Tests reside under the "qikkDB_test" directory.

# Building DB Core
This project works both on Windows and Linux operating systems.

## Windows
For development on Windows, Microsoft Visual Studio 2019 is recommended. The project itself has these dependencies:
- CUDA version 10.2 minimal
- CMake 3.15 or newer
- vcpkg
- boost

For installation of vcpkg and boost, follow these steps:
1. download vcpkg by cloning or downloading github repo https://github.com/Microsoft/vcpkg
2. open power shell and type following commands:


    .\bootstrap-vcpkg.bat
    .\vcpkg integrate install
    .\vcpkg install boost:x64-windows-static

To clone the repository, run this command:

    git clone https://github.com/qikkDB/qikkdb-community

For opening and building the project, run Visual Studio, click "Open a local folder" and choose created "qikkdb-community" folder, wait for CMake cache generating and if it finishes, click "Build" and "Build All".

## Linux
For development on Linux, the dependencies are following:
- CUDA version 10.2 minimal
- CMake 3.15 or newer
- boost

To install boost, following command could be used:

    mkdir -p ./boost/src
	cd ./boost/src
    curl -S -L https://dl.bintray.com/boostorg/release/1.69.0/source/boost_1_69_0.tar.gz | tar -xz
	cd boost_1_69_0
	./bootstrap.sh
	./b2 install -j32 --prefix=/opt/boost_1.69

To clone and build the project:

    git clone https://github.com/qikkDB/qikkdb-community
    cd qikkdb-community
    mkdir build
    cd build
    cmake -GNinja -DCMAKE_CXX_COMPILER=clang++-7 -DCMAKE_C_COMPILER=clang-7 -DBOOST_ROOT=/opt/boost_1.69 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_BUILD_TYPE=Release ..
    ninja
    

# Building Console Client
Run command:

dotnet build ./QikkDB.ConsoleClient/

then go to directory ./build/debug/client and run command:

dotnet ./QikkDB.ConsoleClient.dll