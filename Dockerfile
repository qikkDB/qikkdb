# Build
FROM nvidia/cuda:10.1-devel AS builder

WORKDIR /build/dropdbase_instarea

COPY . ./

WORKDIR /build

RUN mkdir /databases

# Install needed packages in non-interactive mode
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y build-essential \
	cmake \
	ninja-build \
	clang-7 \
	clang++-7 \
	curl \
	wget \
	git-all \
	uuid-dev \
	pkg-config \
	python2.7-dev

# Install yaml	
RUN mkdir -p ./yaml-cpp/src \
	&& mkdir -p ./yaml-cpp/build \
	&& cd ./yaml-cpp/src \
    && curl -SL https://github.com/jbeder/yaml-cpp/archive/yaml-cpp-0.6.2.tar.gz \
    | tar -xz \
	&& cd ../build \
	&& cmake -DCMAKE_BUILD_TYPE=Release ../src/yaml-cpp-yaml-cpp-0.6.2 \
	&& make -j \
	&& make -j install
	
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

# Application
FROM nvidia/cuda:10.1-runtime

WORKDIR /app

# Copy .exe file from build into app
COPY --from=builder /build/build_dropdbase/dropdbase/dropdbase_instarea .

# Copy configuration files into app
COPY configuration /configuration

RUN mkdir /databases

ENTRYPOINT ["./dropdbase_instarea"]