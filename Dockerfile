# Build
FROM nvidia/cuda:10.2-devel AS builder

WORKDIR /build/dropdbase_instarea

COPY . ./

WORKDIR /build

RUN mkdir /databases

# Install needed packages in non-interactive mode
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y build-essential \
	ninja-build \
	clang-7 \
	clang++-7 \
	curl \
	wget \
	git-all \
	uuid-dev \
	pkg-config \
	python2.7-dev
	
# Install CMake 3.17
RUN wget -qO- "https://cmake.org/files/v3.17/cmake-3.17.0-Linux-x86_64.tar.gz" | tar --strip-components=1 -xz -C /usr/local

WORKDIR /build	
	
# Install boost
RUN mkdir -p ./boost/src \
	&& cd ./boost/src \
    && curl -S -L https://dl.bintray.com/boostorg/release/1.69.0/source/boost_1_69_0.tar.gz | tar -xz \
	&& cd boost_1_69_0 \
	&& ./bootstrap.sh \
	&& ./b2 install -j32 --prefix=/opt/boost_1.69
	
WORKDIR /build

# Run CMake DropDBase
RUN	mkdir build_dropdbase \
	&& cd build_dropdbase \
	&& cmake -GNinja -DCMAKE_CXX_COMPILER=clang++-7 -DCMAKE_C_COMPILER=clang-7 -DBOOST_ROOT=/opt/boost_1.69 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_BUILD_TYPE=Release ../dropdbase_instarea

RUN cd build_dropdbase && ninja

# Build client console
FROM microsoft/dotnet:2.2-sdk AS console-build
WORKDIR /src
COPY ./ColmnarDB.ConsoleClient ColmnarDB.ConsoleClient/
COPY ./ColmnarDB.NetworkClient ColmnarDB.NetworkClient/
RUN dotnet restore ColmnarDB.ConsoleClient/ColmnarDB.ConsoleClient.csproj
WORKDIR /src/ColmnarDB.ConsoleClient
RUN dotnet publish -c Release -r linux-x64 --self-contained true ColmnarDB.ConsoleClient.csproj -o /app

# Application
FROM nvidia/cuda:10.2-runtime

WORKDIR /app

# Copy .exe file from build into app
COPY --from=builder /build/build_dropdbase/dropdbase/dropdbase_instarea .

# Copy Boost built libraries
COPY --from=builder /opt/boost_1.69 /opt/boost_1.69
RUN ldconfig /opt/boost_1.69/lib

# Copy configuration files into app
COPY configuration /configuration

RUN mkdir /databases

# Copy client console from console-build into app (without dotnet dependencies)
RUN mkdir /client
COPY --from=console-build /app ./client

ENTRYPOINT ["./dropdbase_instarea"]
