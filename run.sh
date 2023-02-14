#!/bin/bash

cmake .
make
mv lab1 cmake-build-debug
cd cmake-build-debug
./lab1 8000 8000 100 200 200
gnuplot script.txt
feh main.dat.png