#!/bin/bash

g++ -O3 -march=native -o lab1 --std=c++20 "parallel_memory_main.cpp"
./lab1 500 500 500 1 1 $1
gnuplot script.txt
feh main.dat.png
