FROM ubuntu:18.04 AS builder

WORKDIR /build/dropdbase_instarea

COPY . ./

WORKDIR /build

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

# Install NVIDIA repo metadata
RUN mkdir -p ./cuda/src && cd ./cuda/src
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.168-1_amd64.deb
RUN dpkg --install cuda-repo-ubuntu1804_10.1.168-1_amd64.deb

# Install CUDA GPG key
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

# RUN apt-get install gnupg-curl # This command returns code 100 because it is unable to locate this package
RUN apt-get update
RUN apt-get -y install cuda

RUN systemctl enable nvidia-persistenced

WORKDIR /build

# Install Ninja
RUN	mkdir build_dropdbase \
	&& cd build_dropdbase \
	&& cmake -GNinja -DCMAKE_CXX_COMPILER=clang++-7 -DCMAKE_C_COMPILER=clang-7 -DBOOST_ROOT=/opt/boost_1.69 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-10.1/bin/nvcc -DCMAKE_BUILD_TYPE=Release ../dropdbase_instarea

RUN cd build_dropdbase && ninja








FROM ubuntu:18.04

WORKDIR /build/dropdbase_instarea

COPY . ./

WORKDIR /build

# Install needed packages in non-interactive mode
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y build-essential \
	wget
	
WORKDIR /build

# Install NVIDIA repo metadata
RUN mkdir -p ./cuda/src && cd ./cuda/src
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.168-1_amd64.deb
RUN dpkg --install cuda-repo-ubuntu1804_10.1.168-1_amd64.deb

# Install CUDA GPG key
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

# RUN apt-get install gnupg-curl # This command returns code 100 because it is unable to locate this package
RUN apt-get update
RUN apt-get -y install cuda

RUN systemctl enable nvidia-persistenced

WORKDIR /build

COPY --from=builder /build/dropdbase/dropdbase_instarea .

ENTRYPOINT ["dropdbase_instarea"]