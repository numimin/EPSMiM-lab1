#!/bin/bash

g++ -O3 -march=native -o lab1 main.cpp
./lab1 500 500 501 20 20 $1
gnuplot script.txt
feh main.dat.png
