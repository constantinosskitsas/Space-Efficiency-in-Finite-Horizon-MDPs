#!/bin/bash
for i in 1 2 3 4 5	6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
    for j in 100 200 400 800 1600 3200 6400 12800 25600 51200 102400 204800 409600 819200 
    do
        ./run_model.sh naiveMTMR $j $i 1 >> SINGLE_VARREW_NEW_NAIVE.out &&
        #./run_model.sh rootPrun $j $i 1 >> SINGLE_VARREW_NEW_NAIVE.out &&
        #./run_model.sh treePrun $j $i 1 >> SINGLE_VARREW_NEW_NAIVE.out &&
        ./run_model.sh infiniteMR $j $i 0.1 >> VARREW_NEW_INF_1.out &&
        ./run_model.sh infiniteMR $j $i 0.3 >> VARREW_NEW_INF_3.out &&
        ./run_model.sh infiniteMR $j $i 0.99 >> SINGLE_VARREW_NEW_INF_9.out &&
        #./run_model.sh infinite $j $i 0.1 >> SINGLE_VARREW_NEW_INF_1.out &&
        #./run_model.sh infinite $j $i 0.3 >> SINGLE_VARREW_NEW_INF_3.out
        ./run_model.sh infiniteMTMR $j $i 0.99 >> SINGLE_VARREW_NEW_INFM.out
    done
done
