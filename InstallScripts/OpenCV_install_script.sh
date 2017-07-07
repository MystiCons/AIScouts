#!/bin/bash

sudo apt-get update

sudo apt-get upgrade -y

sudo apt-get install -y build-essential cmake

sudo apt-get install python-pip python3-pip python-dev python3-dev -y

sudo apt-get install build-essential cmake pkg-config -y 

sudo apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev -y

sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev -y

sudo apt-get install libxvidcore-dev libx264-dev -y

sudo apt-get install libgtk-3-dev -y

sudo apt-get install libatlas-base-dev gfortran -y 

sudo apt-get install python2.7-dev python3.5-dev -y 

sudo apt-get install libopencv-dev python-opencv -y

sudo apt-get install -y libtbb-dev libeigen3-dev

cd ~
mkdir opencv
cd opencv
wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.2.0.zip
unzip opencv.zip

wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.2.0.zip
unzip opencv_contrib.zip

#if numpy wont install add sudo
pip install numpy
pip3 install numpy

wget -O opencv-3.2.0.zip https://sourceforge.net/projects/opencvlibrary/files/opencv-unix/3.2.0/opencv-3.2.0.zip/download
unzip opencv-3.2.0.zip

cd opencv-3.2.0
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D BUILD_NEW_PYTHON_SUPPORT=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.2.0/modules \ -D BUILD_EXAMPLES=ON ..

make -j4

sudo make install
sudo ldconfig


