#include <iostream>
#include "FiniteMDPModel.h"
#include "ModelConf.h"
#include <vector>
#include "Complex.h"
#include <chrono>
#include <fstream>
#include "stdlib.h"
#include "stdio.h"
#include "string.h"


/*

This script is used to run all models and create measurements. 

To compile in Windows, type in a terminal:
    g++ -o output_script.exe compare_all_models.cpp -lpsapi
and execute by typing:
    .\output_script.exe <model_parameters.json>

To compile in Linux, type in a terminal:
    g++ -o output_script.sh compare_all_models.cpp
and execute by typing:
    ./output_script.exe <model_parameters.json>

where <model_parameters.json> is the path of the input json file containing the data (in JSON format) to configure the MDP Model.

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
    int num_tests = 1;
    int training_steps = 5000;
    //vector<int> horizon {100,200,400,800,1600,3200,6400,12800,25600,51200,102400,204800,409600,819200};
    vector<int> horizon {100,200,400};
    //vector<int> horizon {12800};
    int seed = 18;
    int max_memory_used = 0;    
    int load_period = 250;
    int MIN_VMS = 1;
    int MAX_VMS = 20;
    float epsilon = 0.7;
   //string CONF_FILE = "EffectiveMDP/new_MDP/model_parameters/mdp_huge.json"; //sorry for that but keep it in comments
     string CONF_FILE = "EffectiveMDP/new_MDP/model_parameters/mdp_small_3.json"; 
    //string CONF_FILE = "model_parameters/mdp_huge.json"; //sorry for that but keep it in comments
    //string CONF_FILE = argv[1];
    ModelConf conf(CONF_FILE);
    float total_rewards_results[5][13];
    for (int i=0; i<5; i++){
        for (int j=0; j<13; j++){
            total_rewards_results[i][j] = 0.0;
        }
    }

    ComplexScenario scenario(5000, load_period, 10, MIN_VMS, MAX_VMS);

    pair<string, int> action;
    for (int number_of_tests =0; number_of_tests < num_tests; number_of_tests++){
    FiniteMDPModel model(conf.get_model_conf(), seed);
    model.set_state(scenario.get_current_measurements());
    float total_reward = 0.0;
    
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
    int count0=0;
    int count1=0;
    //max_memory_used = getValue();
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
    model.initial_state_num = model.current_state_num;
    model.discount=0.9; //for the infiniteM to test the discount=1
        for (int j = 0; j < 1; j++){//just to test the same model results
        for (int i = 0; i < horizon.size(); i++){
            model.resetModel();
            /*
            model.runAlgorithm(infinite, horizon[i]);
            total_rewards_results[0][i] += model.total_reward;
            model.resetModel();
            /*
            
            model.runAlgorithm(naive, horizon[i]);
            total_rewards_results[1][i] += model.total_reward;
            model.resetModel();
            /*
            cout << "Horizon size: " << horizon[i] << endl;
            cout << "Total Reward Expected: " << model.expected_reward << endl;
            cout << "Total Reward Collected: " << model.total_reward << endl;
            cout << "Peak memory used (MB): " << model.max_memory_used / 1000000.0 << endl;
            
            
            
            model.runAlgorithm(root, horizon[i]);
            total_rewards_results[2][i] += model.total_reward;
            model.resetModel();
            */
            cout << "Hi: " << endl;
            model.runAlgorithm(naiveMT, horizon[i]);
            total_rewards_results[3][i] += model.total_reward;
            model.resetModel();
            /*
            model.runAlgorithm(infiniteM, horizon[i]);
            total_rewards_results[4][i] += model.total_reward;
            model.resetModel();
            
            model.runAlgorithm(inplace, horizon[i]);
            total_rewards_results[5][i] += model.total_reward;
            model.resetModel();
            */
        }
        }//just to test the same model results

    for (int i=0; i < 5; i++){
        for (int j =0; j < horizon.size(); j++){
            cout << "(" << horizon[j] << "," << total_rewards_results[i][j]  << ")";
         
        }
        cout << endl;
    }
    
    }
}
