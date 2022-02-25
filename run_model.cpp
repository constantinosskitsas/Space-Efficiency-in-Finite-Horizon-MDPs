#include <iostream>
#include "FiniteMDPModel.h"
#include "ModelConf.h"
#include <vector>
#include "Complex.h"
#include <chrono>
#include <sstream>


#include "stdlib.h"
#include "stdio.h"
#include <string>

/*

This script is used to run only ONE model and create measurements. 

To compile in Windows, type in a terminal:
    g++ -o output_script.exe run_model.cpp -lpsapi
and execute by typing:
    .\output_script.exe <algorithm_type> <horizon_size> <seed>

To compile in Linux, type in a terminal:
    g++ -o output_script.sh run_model.cpp
and execute by typing:
    ./output_script.exe <algorithm_type> <horizon_size> <seed>

where <algorithm_type> can be: infinite, naive, root, tree, inplace
<horizon_size> can be any positive integer
and <seed> can be any positive integer

*/

using namespace std::chrono;

using namespace std;

float avg(vector<float> v)
{
    float return_value = 0.0;
    int n = v.size();

    for (int i = 0; i < n; i++)
    {
        return_value += v[i];
    }
    return (return_value / (float)n);
}


pair<string, int> randomchoice(vector<pair<string, int>> v, FiniteMDPModel &model)
{
    float n = (float)v.size();
    float x = 1.0 / n;
    //float r = myRand(0, 1);
    float r = model.unif(model.eng);
    for (int i = 1; i < n + 1; i++)
    {
        if (r < x * i)
            return v[i - 1];
    }
    return v[0];
}

int main(int argc, char *argv[])
{
    int horizon = 100;
    string algorithm_type = "none";
    int seed = 21;
    model_type algo;
    if (argc >1) {
        algorithm_type = argv[1];
        if (algorithm_type == "infinite") algo = infinite;
        else if (algorithm_type == "infiniteM") algo = infiniteM;
        else if (algorithm_type == "naive") algo = naive;
        else if (algorithm_type == "root") algo = root;
        else if (algorithm_type == "tree") algo = tree;
        else if (algorithm_type == "inplace") algo = inplace;
        else if (algorithm_type == "infiniteMR") algo = infiniteMR;
        else if (algorithm_type == "infiniteMT") algo = infiniteMT;
        else if (algorithm_type == "infiniteMTMR") algo = infiniteMTMR;
        else if (algorithm_type == "naivePr") algo = naivePr;
        else if (algorithm_type == "naiveMT") algo = naiveMT;
        else if (algorithm_type == "naiveMTMR") algo = naiveMTMR;
        else if (algorithm_type == "rootPrun") algo = rootPrun;
        else if (algorithm_type == "treePrun") algo = treePrun;
        else if (algorithm_type == "treeRec") algo = treeRec;
    }
    float gama = 0.5;
    /*if (argc == 5){
        std::size_t pos;
        horizon = std::stoi(argv[2], &pos);
        seed = std::stoi(argv[3], &pos);
        gama = std::stof(argv[4], &pos);
    }*/
    std::size_t pos;
    horizon = std::stoi(argv[2], &pos);
    seed = std::stoi(argv[3], &pos);
    gama = std::stof(argv[4], &pos);


    int training_steps = 10000;
    int max_memory_used = 0;
    int load_period = 250;
    int MIN_VMS = 1;
    int MAX_VMS = 20;
    float epsilon = 0.7;
    //string CONF_FILE = "./model_parameters/mdp_small_1.json";
    string CONF_FILE = "EffectiveMDP/new_MDP/model_parameters/mdp_small_3.json"; 
    //string CONF_FILE = "EffectiveMDP/new_MDP/model_parameters/mdp_huge.json"; 
    ModelConf conf(CONF_FILE);

    ComplexScenario scenario(5000, load_period, 10, MIN_VMS, MAX_VMS);
    FiniteMDPModel model(conf.get_model_conf(), seed);
    model.set_state(scenario.get_current_measurements());
    float total_reward = 0.0;
    pair<string, int> action;
    //TRAIN THE MODEL

    for (int time = 0; time < training_steps; time++){

        float x = model.unif(model.eng);
        if (x < epsilon)
        {
            action = randomchoice(model.get_legal_actions(), model);
        }
        else
        {
            action = model.suggest_action();
        }
        float reward = scenario.execute_action(action);
        json meas = scenario.get_current_measurements();
        model.update(action, meas, reward);
        if (time % 500 == 1){
            model.value_iteration(0.1);
        }
    }
    model.initial_state_num = model.current_state_num;
    int count0=0;
    int count1=0;
    for (int i=0;i< model.states.size();i++){
        for (int j=0;j< model.states[i].qstates.size();j++){
            for (int k=0;k< model.states[i].qstates[j].transitions.size();k++){
                if (model.states[i].qstates[j].get_transition(k)>0){
                    model.states[i].qstates[j].trans.push_back(model.states[i].qstates[j].get_transition(k));
                    model.states[i].qstates[j].transtate.push_back(k);
                    count1++;
                }
                else {
                    count0++;
                }
            }
        }
    }
    model.discount = gama;
    cout << "model discount " << model.discount << endl; 
    /*for (int i=0;i< model.states.size();i++){
        for (int j=0;j< model.states[i].qstates.size();j++){
            for (int k=0; k < model.states[i].qstates[j].transtate.size(); k++){
                cout << "State " << i << " Qstate " << j << " probability = " << model.states[i].qstates[j].trans[k] << " to " << model.states[i].qstates[j].transtate[k] << endl;
            }
        }
    }*/

    model.runAlgorithm(algo, horizon);
}