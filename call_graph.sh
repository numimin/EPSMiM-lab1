#!/bin/bash

g++ -O3 -g -pg -fno-inline -fno-omit-frame-pointer -march=native -o lab1 --std=c++20 "atomic.cpp"
./lab1 8000 8000 100 1 1 6
gprof ./lab1 > lab1.profile.txt
python ~/Documents/github/gprof2dot/gprof2dot.py lab1.profile.txt > profile.dot
xdot profile.dot
