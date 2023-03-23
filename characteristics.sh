#!/bin/bash

g++ -O3 -march=native -g -o lab1 "main1.cpp"
perf stat -e cycles -e instructions -e branches -e branch-misses ./lab1 8000 8000 100 1 1 

perf stat -e L1-dcache-loads -e L1-dcache-load-misses ./lab1 8000 8000 100 1 1 
    
perf stat -e LLC-loads -e LLC-load-misses -e LLC-stores -e LLC-store-misses ./lab1 8000 8000 100 1 1 
