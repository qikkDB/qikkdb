FROM ubuntu:18.04

WORKDIR /build

COPY . ./

RUN apt-get update && apt-get install -y build-essential \
	cmake \
	ninja-build \
	clang-7 \
	clang++-7 \
	curl \
	python2.7-dev

# Install yaml	
RUN mkdir -p ./yaml-cpp/src \
	&& mkdir -p ./yaml-cpp/build \
	&& cd ./yaml-cpp/src \
    && curl -SL https://github.com/jbeder/yaml-cpp/archive/yaml-cpp-0.6.2.tar.gz \
    | tar -xz \
	&& cd ../build \
	&& cmake -DCMAKE_BUILD_TYPE=Release ../src/yaml-cpp-yaml-cpp-0.6.2 \
	&& make \
	&& make install
	
WORKDIR /build	
	
# Install boost
RUN mkdir -p ./boost/src \
	&& cd ./boost/src \
    && curl -S -L https://dl.bintray.com/boostorg/release/1.69.0/source/boost_1_69_0.tar.gz | tar -xz \
	&& cd boost_1_69_0 \
	&& ./bootstrap.sh \
	&& ./b2
	
WORKDIR /build
RUN mkdir build_dropdbase && cd build_dropdbase

RUN	cmake -GNinja -DCMAKE_CXX_COMPILER=clang++-7 -DCMAKE_C_COMPILER=clang-7 -DBOOST_ROOT=/opt/boost_1.69 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-10.1/bin/nvcc -DCMAKE_BUILD_TYPE=Release ../dropdbase_instarea

RUN ninja