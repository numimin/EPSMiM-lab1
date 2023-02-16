#!/bin/bash

cmake .
make
mv lab1 cmake-build-debug
cd cmake-build-debug
./lab1 500 500 500 1 1
gnuplot script.txt
feh main.dat.png