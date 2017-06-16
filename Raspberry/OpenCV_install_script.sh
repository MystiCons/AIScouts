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
wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.2.0.zip
unzip opencv.zip

wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.2.0.zip
unzip opencv_contrib.zip


#if numpy wont install add sudo
pip install numpy
pip3 install numpy



