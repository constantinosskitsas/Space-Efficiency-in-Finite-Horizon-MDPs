# Space-Efficiency-in-Finite-Horizon-MDPs
## **Introduction**
Can we get finite-horizon Markov decision processes (FHMDPs)
to be solved with low memory requirements? Such models find
application in many cases where an decision-making agent needs
to act in a probabilistic environment from resource management
to medicine to service provisioning. However, computing optimal
policies such an agent should follow by dynamic programming
value iteration raises either prohibitive space complexity, or, in re-
verse, non-scalable time complexity requirements. This scalability
question has been largely neglected. In this paper, we propose SIFT
(Space Efficient Finite Horizon MDPs) a suite of algorithms that
achieve a golden middle between space and time requirements. Our
former algorithm raises space complexity growing with the square
root of the horizon length without a time-complexity overhead,
while the latter’s space requirements depend only logarithmically
in horizon length with a corresponding logarithmic time complexity
overhead. A thorough experimental study confirms that SIFT algo-
rithms achieve the predicted gains, while approximation techniques
do not achieve the same combination of time efficiency, space effi-
ciency, and result quality.
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
1)Mem_Run.sh Memory and Runtime Experiment. 
2)Rewards_MR.sh Modified Rewards Experiment. 
3)Rewards_exp.sh Rewards Experiment. 

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
@article{skitsas2022sifter,
  title={SIFTER: space-efficient value iteration for finite-horizon MDPs},
  author={Skitsas, Konstantinos and Papageorgiou, Ioannis G and Talebi, Mohammad Sadegh and Kantere, Verena and Katehakis, Michael N and Karras, Panagiotis},
  journal={Proceedings of the VLDB Endowment},
  volume={16},
  number={1},
  pages={90--98},
  year={2022},
  publisher={VLDB Endowment}
}
```
The project was implemented in collaboration with Giannis Papageorgiou @el16104.
