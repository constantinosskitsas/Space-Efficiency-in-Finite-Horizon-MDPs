# Space-Efficiency-in-Finite-Horizon-MDPs
## **Introduction**

## Required Libraries
-pthread Basic library for threads in C++
[nlohmann](https://github.com/nlohmann/json) jason Library. We recomend to read [nlohmann](https://github.com/nlohmann/json) documentation to run the library properly.
## Algorithms 
      infinite : value iteration gith a given discount rate and stable Rewards (Stationary Policy)
      infiniteMR : value iteration gith a given discount rate and Modified Rewards (Stationary Policy)
      infiniteM : value iteration gith discount rate=1 and stable Rewards (Stationary Policy)
      infiniteMT : Multi-Threaded value iteration gith discount rate=1 and stable Rewards (Stationary Policy)
      infiniteMTMR: Multi-Threaded value iteration gith discount rate=1 and Modified Rewards (Stationary Policy)
      naive: value iteration gith discount rate=1 and stable Rewards (Non-Stationary Policy)
      naivePr: Pruning value iteration gith discount rate=1 and stable Rewards (Non-Stationary Policy)
      naiveMT :Multi-Threaded value iteration gith discount rate=1 and stable Rewards (Non-Stationary Policy)
      naiveMTMR:Multi-Threaded value iteration gith discount rate=1 and Modified Rewards (Non-Stationary Policy)
      root : Root value iteration gith discount rate=1 and stable Rewards (Non-Stationary Policy)
      rootPrun: Root Prunning value iteration gith discount rate=1 and stable Rewards (Non-Stationary Policy)
      tree: Tree value iteration gith discount rate=1 and stable Rewards (Non-Stationary Policy)
      treePrun: Pruning Tree value iteration gith discount rate=1 and stable Rewards (Non-Stationary Policy)
      treeRec : Recalculation Tree value iteration gith discount rate=1 and stable Rewards (Non-Stationary Policy)
      inplace : Inplace value iteration gith discount rate=1 and stable Rewards (Non-Stationary Policy)

### How to run experiments :
We created scripts that represent the experiments we have in the paper
```shell
1)Mem_Run.sh Memory and Runtime Experiment. Figure xx
2)Rewards_MR.sh Modified Rewards Experiment. Figure xx
3)Rewards_exp.sh Rewards Experiment. Figure xx

Run :1)g++ -pthread /run_model.cpp -o run_model.sh 
     2)Run the scripts.
```
## Additional Information : 

  The scripts pass 4 parameters the algorithm,randomness seed and Horizon size and gamma(discount rate) (run_model.cpp).Gamma has no effect for the Finite Horizon algorithms
  You can run the code also using the run_model.cpp and passing the abovementioned parameters.
  compare_all_models.cpp is also for experiments. However it accepts no parameters and changes have to be made from code.
  To Change the model (number of states) you have to change manually the parameter CONF_FILE in run_models.cpp or compare_all_models.cpp respectivelly.

## Reference

Please cite our work in your publications if it helps your research:

```
The paper is under submission. 
```  
