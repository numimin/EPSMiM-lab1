#!/bin/bash

g++ -O3 -march=native -o lab4 --std=c++20 "atomic.cpp"
./lab4 500 500 500 1 1 $1
gnuplot script.txt
feh main.dat.png
