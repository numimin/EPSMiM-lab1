#!/bin/bash

g++ -O3 -march=native -o lab1 "aligned_main.cpp"
./lab1 500 500 500 1 1 
gnuplot script.txt
feh main.dat.png
