#include <iostream>
#include <vector>
#include <map>
#include <stdexcept>
#include <string>
#include <stack>
#include <random>
#include <math.h>
#include <array>
#include <chrono>
#include <sstream>
#include<thread>
#include "MDPModel.h"
#include "Complex.h"

#include "stdlib.h"
#include "stdio.h"


#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <nlohmann\json.hpp>
#endif

#ifdef linux
#include <nlohmann/json.hpp>
#define SIZE_T int
#endif

using namespace std::chrono;
enum model_type {infinite, naive, root,rootPrun, tree, inplace, infiniteM, naiveMT, naivePr, treeRec, treePrun, naiveMTMR, infiniteMTMR, infiniteMT, infiniteMR};

SIZE_T getValue(){
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return(pmc.WorkingSetSize); //Value in Bytes!
    else
        return 0;
#endif
#ifdef linux
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];

    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, "VmSize:", 7) == 0){
            result = parseLine(line);
            break;
        }
    }
    fclose(file);
    return result;
#endif

}

using json = nlohmann::json;

//using namespace std;

class FiniteMDPModel: public MDPModel{
    public:
        stack<int> index_stack;
        stack<vector<pair<int , float>>> finite_stack;
        stack<vector<int>> action_stack; //STACK TO CONTAIN VECTOR OF BEST QSTATE FOR EACH INDEX
        float total_reward = 0.0;
        int max_memory_used = 0;
        int init_memory_used=0;
        int max_stack_memory = 0;
        int steps_made = 0;
        float expected_reward = 0.0;
        default_random_engine eng;
        uniform_real_distribution<float> unif;
        int stack_memory = 0;

    FiniteMDPModel(json conf = json({}), int seed = 21){
        if (conf.contains("discount"))
        discount = conf["discount"];

        if (conf.contains("parameters"))
        parameters = _get_params(conf["parameters"]);

        for (auto& element : parameters.items()) {
            index_params.push_back(element.key());
            _update_states(element.key(), element.value());
        }


        int num_states = states.size();

        for (auto & element : states){
            element.set_num_states(num_states);
        }
        if (conf.contains("actions")){
        _set_maxima_minima(parameters, conf["actions"]);

        if (conf.contains("initial_qvalues"))
            _add_qstates(conf["actions"], conf["initial_qvalues"]);
        }
        update_algorithm  = true;
        
        eng = default_random_engine(seed);
        unif = uniform_real_distribution<float>(0,1);
            
    };

    void setInitialState(State &s){
        current_state_num = s.state_num;
    }

    /*
    New function, checks if the current memory used by the process is greater than
    the current maximum and updates it accordingly, using the getValue() function.
    No input.
    No output.
    */
    void checkMemoryUsage(){
        SIZE_T x = getValue();
        if (getValue() > max_memory_used){
            max_memory_used = getValue();
        }
    }

    /*
    New function, checks if the current number of elements in the stack is greater than
    the current maximum and updates it accordingly.
    No input.
    No output.
    */
    void checkStackSize(){
        if (stack_memory > max_stack_memory){
            max_stack_memory = stack_memory;
        }
    }


    void _q_update_finite(QState &qstate, vector<pair<int,float>> &V,int b){
        float new_qvalue = 0.0;
        float r;
        float t;
        for (int i=0; i < V.size(); i++){
            t = qstate.get_transition(i);
            r = qstate.get_reward(i);
            new_qvalue += t * (r +V[i].second);
        }
        qstate.set_qvalue(new_qvalue);
    }


    void calculateValuestest(int k, int starting_index, vector<pair<int, float>> &V, bool tree = false){

        for (int i = starting_index+1 ; i < k+1; i++){
            for (int j = 0 ; j < states.size(); j++ ){
                
                for (int m = 0; m < states[j].get_qstates().size(); m++){
                    _q_update_finite(states[j].qstates[m], V, i);
                }
                states[j].update_value();
            }
            V = getStateValuestest(states);
            if (!tree){
                index_stack.push(i);
                finite_stack.push(V);
                //stack_memory++;
               // checkStackSize();
                }
            }

        checkMemoryUsage();

    }


    void calculateValuestestRec(int k, int starting_index, vector<pair<int, float>> &V, bool tree = false){

        for (int i = starting_index+1 ; i < k+1; i++){
            for (int j = 0 ; j < states.size(); j++ ){
                
                for (int m = 0; m < states[j].get_qstates().size(); m++){
                    if (states[j].qstates[m].lastused<i){
                        states[j].qstates[m].set_qvalue(-100000);
                        continue;
                        }
                    _q_update_finite(states[j].qstates[m], V, i);
                }
                states[j].update_value();
            }
            V = getStateValuestest(states);
            if (!tree){
                index_stack.push(i);
                finite_stack.push(V);
                //stack_memory++;
               // checkStackSize();
                }
            }

        checkMemoryUsage();

    }

    void calculateValuesPrunningRAU(int k, int starting_index, vector<pair<int, float>> &V, bool tree = false){
            float num0rew=-1;
            int statenum=-1;
            for (int i = starting_index+1 ; i < k+1; i++){
                num0rew=calcrewa(V);
            for (int j = 0 ; j < states.size(); j++ ){
                
            for (int n = 0; n < states[j].get_qstates().size(); n++){
                if (states[j].qstates[n].lastused<i){
                    states[j].qstates[n].set_qvalue(-100000);
                    continue;
                    }
                    float new_qvalue = 0.0;
                    float r;
                    float t;
                    if (states[j].num_visited==0)
                        states[j].qstates[n].set_qvalue(num0rew);
                    else{
                    for (int m=0; m < states[j].qstates[n].trans.size(); m++){ //FOR EVERY ACCESIBLE STATE FROM CURRENT QSTATE
                        t = states[j].qstates[n].trans[m];
                        statenum=states[j].qstates[n].transtate[m];
                        //r = states[j].qstates[n].get_reward(statenum, i);
                        r = states[j].qstates[n].get_reward(statenum);
                        new_qvalue += t * (r +V[statenum].second);
                    }
                    states[j].qstates[n].set_qvalue(new_qvalue);
                }
                }
                states[j].update_value();
            }
            V = getStateValuestest(states);
            //V = getStateValuestestUN(states,i);
            if (!tree){
                //index_stack.push(i);
                finite_stack.push(V);
                //stack_memory++;
               // checkStackSize();
                }
            }

        checkMemoryUsage();

    }
    void calculateValuesPrunningR(int k, int starting_index, vector<pair<int, float>> &V, bool tree = false){
            float num0rew=-1;
            int statenum=-1;
            for (int i = starting_index+1 ; i < k+1; i++){
                num0rew=calcrewa(V);
            for (int j = 0 ; j < states.size(); j++ ){
                
            for (int n = 0; n < states[j].get_qstates().size(); n++){
                    float new_qvalue = 0.0;
                    float r;
                    float t;
                    if (states[j].num_visited==0)
                        states[j].qstates[n].set_qvalue(num0rew);
                    else{
                    for (int m=0; m < states[j].qstates[n].trans.size(); m++){ //FOR EVERY ACCESIBLE STATE FROM CURRENT QSTATE
                        t = states[j].qstates[n].trans[m];
                        statenum=states[j].qstates[n].transtate[m];
                        //r = states[j].qstates[n].get_reward(statenum, i);
                        r = states[j].qstates[n].get_reward(statenum);
                        new_qvalue += t * (r +V[statenum].second);
                    }
                    states[j].qstates[n].set_qvalue(new_qvalue);
                }
                }
                states[j].update_value();
            }
            V = getStateValuestest(states);
            //V = getStateValuestestUN(states,i);
            if (!tree){
                //index_stack.push(i);
                finite_stack.push(V);
                //stack_memory++;
               // checkStackSize();
                }
            }

        checkMemoryUsage();

    }
    void calculateValuesPrunning(int k, int starting_index, vector<pair<int, float>> &V, bool tree = false){
            float num0rew=-1;
            int statenum=-1;
            for (int i = starting_index+1 ; i < k+1; i++){
                num0rew=calcrewa(V);
            for (int j = 0 ; j < states.size(); j++ ){
                
            for (int n = 0; n < states[j].get_qstates().size(); n++){
                    float new_qvalue = 0.0;
                    float r;
                    float t;
                    if (states[j].num_visited==0)
                        states[j].qstates[n].set_qvalue(num0rew);
                    else{
                    for (int m=0; m < states[j].qstates[n].trans.size(); m++){ //FOR EVERY ACCESIBLE STATE FROM CURRENT QSTATE
                        t = states[j].qstates[n].trans[m];
                        statenum=states[j].qstates[n].transtate[m];
                       //r = states[j].qstates[n].get_reward(statenum, i);
                        r = states[j].qstates[n].get_reward(statenum);
                        new_qvalue += t * (r +V[statenum].second);
                    }
                    states[j].qstates[n].set_qvalue(new_qvalue);
                }
                }
                states[j].update_value();
            }
            V = getStateValuestest(states);
            //V = getStateValuestestUN(states,i);
            if (!tree){
                index_stack.push(i);
                finite_stack.push(V);
                //stack_memory++;
               // checkStackSize();
                }
            }

        checkMemoryUsage();

    }

        void calculateValuesPrunningA(int k, int starting_index, vector<pair<int, float>> &V, bool tree = false){
            float num0rew=-1;
            int statenum=-1;
            for (int i = starting_index+1 ; i < k+1; i++){
                num0rew=calcrewa(V);
            for (int j = 0 ; j < states.size(); j++ ){
                
            for (int n = 0; n < states[j].get_qstates().size(); n++){
                    float new_qvalue = 0.0;
                    float r;
                    float t;
                    if (states[j].num_visited==0)
                        states[j].qstates[n].set_qvalue(num0rew);
                    else{
                    for (int m=0; m < states[j].qstates[n].trans.size(); m++){ //FOR EVERY ACCESIBLE STATE FROM CURRENT QSTATE
                        t = states[j].qstates[n].trans[m];
                        statenum=states[j].qstates[n].transtate[m];
                       //r = states[j].qstates[n].get_reward(statenum, i);
                        r = states[j].qstates[n].get_reward(statenum);
                        new_qvalue += t * (r +V[statenum].second);
                    }
                    states[j].qstates[n].set_qvalue(new_qvalue);
                }
                }
                states[j].update_value();
            }
            //V = getStateValuestest(states);
            V = getStateValuestestUN(states,i);
            if (!tree){
                index_stack.push(i);
                finite_stack.push(V);
                //stack_memory++;
               // checkStackSize();
                }
            }

        checkMemoryUsage();

    }

    void calculateValuesPrunningRecal(int k, int starting_index, vector<pair<int, float>> &V, bool tree = false){
            float num0rew=-1;
            int statenum=-1;
            for (int i = starting_index+1 ; i < k+1; i++){
                num0rew=calcrewa(V);
            for (int j = 0 ; j < states.size(); j++ ){
                
            for (int n = 0; n < states[j].get_qstates().size(); n++){
                    float new_qvalue = 0.0;
                    float r;
                    float t;
                    if (states[j].num_visited==0)
                        states[j].qstates[n].set_qvalue(num0rew);
                    else{
                    for (int m=0; m < states[j].qstates[n].trans.size(); m++){ //FOR EVERY ACCESIBLE STATE FROM CURRENT QSTATE
                        t = states[j].qstates[n].trans[m];
                        statenum=states[j].qstates[n].transtate[m];
                        r = states[j].qstates[n].get_reward(statenum, i);
                        //r = states[j].qstates[n].get_reward(statenum);
                        new_qvalue += t * (r +V[statenum].second);
                    }
                    states[j].qstates[n].set_qvalue(new_qvalue);
                }
                }
                states[j].update_value();
            }
            //V = getStateValuestest(states);
            V = getStateValuestest(states);
            if (!tree){
                index_stack.push(i);
                finite_stack.push(V);
                //stack_memory++;
               // checkStackSize();
                }
            }

        checkMemoryUsage();

    }


       pair<std::string,int> finite_suggest_action(){
        return states[current_state_num].get_optimal_action();
    }

    void takeAction(bool isInfinite, int time_step, bool p=false){
        pair<std::string,int> action;
        if (isInfinite) {action = suggest_action();}
        else {action = finite_suggest_action();}
        float reward;
        //float reward = scenario.execute_action(action);
        //json meas = scenario.get_current_measurements();

        int prev_state_num = current_state_num;
        //current_state_num = _get_state(meas)->get_state_num();
        float x = unif(eng);
        float acc = 0.0;
        for (int i=0; i< states[prev_state_num].qstates.size();i++){
            if (states[prev_state_num].qstates[i].action.first == action.first){
                for (int j=0; j<states[prev_state_num].qstates[i].transitions.size();j++){
                    if (states[prev_state_num].qstates[i].get_transition(j) > 0){
                        acc += states[prev_state_num].qstates[i].get_transition(j);
                        if (x < acc){
                            current_state_num = j;
                            reward = states[prev_state_num].qstates[i].get_reward(current_state_num, time_step);
                            //stable rewards
                            //reward = states[prev_state_num].qstates[i].get_reward(current_state_num);
                            break;
                        }
                    }
                }
            }
        }

        total_reward += reward;
        if (p){
            cout << "Current state: " << prev_state_num<< endl;
            for (int i=0; i < states[prev_state_num].qstates.size(); i++){
                cout << "   Qstate: " << states[prev_state_num].qstates[i].action.first << " ,QValue:" << states[prev_state_num].qstates[i].qvalue << endl;
            }
            cout << action.first << " ,Reward: " << reward << "   Next state: " << endl << states[current_state_num] << endl;
        }
    }
    void takeAction(bool isInfinite, bool p=false){
        pair<std::string,int> action;
        if (isInfinite) {action = suggest_action();}
        else {action = finite_suggest_action();}
        float reward;
        //float reward = scenario.execute_action(action);
        //json meas = scenario.get_current_measurements();

        int prev_state_num = current_state_num;
        //current_state_num = _get_state(meas)->get_state_num();
        float x = unif(eng);
        float acc = 0.0;
        for (int i=0; i< states[prev_state_num].qstates.size();i++){
            if (states[prev_state_num].qstates[i].action.first == action.first){
                for (int j=0; j<states[prev_state_num].qstates[i].transitions.size();j++){
                    if (states[prev_state_num].qstates[i].get_transition(j) > 0){
                        acc += states[prev_state_num].qstates[i].get_transition(j);
                        if (x < acc){
                            current_state_num = j;
                            //stable rewards
                            reward = states[prev_state_num].qstates[i].get_reward(current_state_num);
                            break;
                        }
                    }
                }
            }
        }

        total_reward += reward;
        if (p){
            cout << "Current state: " << prev_state_num<< endl;
            for (int i=0; i < states[prev_state_num].qstates.size(); i++){
                cout << "   Qstate: " << states[prev_state_num].qstates[i].action.first << " ,QValue:" << states[prev_state_num].qstates[i].qvalue << endl;
            }
            cout << action.first << " ,Reward: " << reward << "   Next state: " << endl << states[current_state_num] << endl;
        }
    }
    void takeActionPrunning(int corraction, int time_step){
        float reward=0;
        int prev_state_num = current_state_num;
        float x = unif(eng);
        float acc = 0.0;
            for (int j=0; j<states[prev_state_num].qstates[corraction].trans.size();j++){
                    acc += states[prev_state_num].qstates[corraction].trans[j];
                    if (x < acc){
                        current_state_num = states[prev_state_num].qstates[corraction].transtate[j];
                        reward = states[prev_state_num].qstates[corraction].get_reward(current_state_num, time_step);
                        //stable reward
                        //reward = states[prev_state_num].qstates[corraction].get_reward(current_state_num);
                        break;
                    }
                }
        total_reward += reward;
    }

       void takeActionPrunning(int corraction){
        float reward=0;
        int prev_state_num = current_state_num;
        float x = unif(eng);
        float acc = 0.0;
            for (int j=0; j<states[prev_state_num].qstates[corraction].trans.size();j++){
                    acc += states[prev_state_num].qstates[corraction].trans[j];
                    if (x < acc){
                        current_state_num = states[prev_state_num].qstates[corraction].transtate[j];
                        reward = states[prev_state_num].qstates[corraction].get_reward(current_state_num);
                        break;
                    }
                }
        total_reward += reward;
    }



    /*
    Auxiliary function that returns the Value Function for every state of the model.
    No input.
    Returns a vector containing the Value Function value for every state of the model.
    */
    vector<float> getStateValueFunction(){
        vector<float> values;
        //values.reserve(states.size());    
        for (int i=0; i < states.size(); i++){
            values.push_back(states[i].get_value());
        }
        
        return values;
    }

    /*
    Auxiliary function that returns the best QState for every state of the model.
    No input.
    Returns a vector containing the index of the best QState for every state of the model.
    */
    vector<int> getStateActions(){
        vector<int> values;
        //values.reserve(states.size());    

        for (int i=0; i < states.size(); i++){
            values.push_back(states[i].get_best_qstate());
        }
        
        return values;
    }

    /*
    New function, does the same as calculateValues(), but stores in memory (stack) only the bestQState.
    Takes as argument the horizon of the Finite-Horizon MDP.
    Returns the total reward the agent is EXPECTED to collect.
    */
 float calculatePolicyPrunning(int k){
        vector<float> V_tmp;
        //V_tmp.reserve(states.size());      
        V_tmp = getStateValueFunction();
        int statenum;
        float num0rew=0.0;
        float test=1.0 /(float)states.size();
        for (int i = 1 ; i < k+1; i++){ //FOR EVERY INDEX UP TO THE HORIZON
                 num0rew=calcrewa(V_tmp);  
            for (int j = 0 ; j < states.size(); j++ ){ //FOR EVERY STATE   
                     
                for (int n = 0; n < states[j].get_qstates().size(); n++){ //FOR EVERY QSTATE OF EACH STATE
                    float new_qvalue = 0.0;
                    float r;
                    float t;
                    if (states[j].qstates[n].num_taken==0){
                        states[j].qstates[n].set_qvalue(num0rew);
                }   
                    
                    else{
                    for (int m=0; m < states[j].qstates[n].trans.size(); m++){ //FOR EVERY ACCESIBLE STATE FROM CURRENT QSTATE
                        t = states[j].qstates[n].trans[m];
                        statenum=states[j].qstates[n].transtate[m];

                        //r = states[j].qstates[n].get_reward(statenum, i);
                        r = states[j].qstates[n].get_reward(statenum);
                        new_qvalue += t * (r +V_tmp[statenum]);

                    }
                 states[j].qstates[n].set_qvalue(new_qvalue);
                }
                
                }
                states[j].update_value();
            }
            V_tmp = getStateValueFunction();

            action_stack.push(getStateActions());
            stack_memory++;
            checkStackSize();
            checkMemoryUsage();
        }
        return V_tmp[initial_state_num];
    }


float calculatePolicyPrunningMTTurnpike(int k,bool Rew,int Tnum=5){
        vector<float> V_tmp;
        vector<thread> mythreads;
        int sz=states.size();
        int sz1=(int) sz/Tnum;
        std::thread th[Tnum];
        //V_tmp.reserve(states.size());      
        V_tmp = getStateValueFunction();
        //vector<float> V_tmp1=getStateValueFunction();
        long double num0rew=0;
            auto f = [](int i,vector<float>& V_tmp,long double num0rew,int s,int sz2,std::vector<State> &states,bool Rew) {
                        int statenum;
        
        for (int j = s ; j < sz2; j++ ){ //FOR EVERY STATE   
                     
            for (int n = 0; n < states[j].get_qstates().size(); n++){ //FOR EVERY QSTATE OF EACH STATE
                long double new_qvalue = 0.0;
                float r;
                float t;
                if (states[j].qstates[n].num_taken==0)
                    states[j].qstates[n].set_qvalue(num0rew);
                else{
                    for (int m=0; m < states[j].qstates[n].trans.size(); m++){ //FOR EVERY ACCESIBLE STATE FROM CURRENT QSTATE
                        t = states[j].qstates[n].trans[m];
                        statenum=states[j].qstates[n].transtate[m];
                        if (Rew)
                            r = states[j].qstates[n].get_reward(statenum);
                        else
                            r = states[j].qstates[n].get_reward(statenum, i);
                        new_qvalue += t * (r +V_tmp[statenum]);
                    }
                    states[j].qstates[n].set_qvalue(new_qvalue);
                }
            }
            states[j].update_value();
        }
            };


        for (int i = 1 ; i < k+1; i++){ //FOR EVERY INDEX UP TO THE HORIZON
            pair <float,float> minmax;
            num0rew=calcrewa(V_tmp);  
            //cout << "Horizon Average Value " << num0rew << endl;           
            //std::thread th1(MTpC,k,V_tmp,num0rew,0,sz1);
            //std::thread th2(MTpC,k,V_tmp,num0rew,sz1+1,sz);
            
            for (int d=0;d<(Tnum-1);d++)
                th[d]=std::thread (f,i,std::ref(V_tmp),num0rew,sz1*d,sz1*(d+1),std::ref(states),Rew);
             th[Tnum-1]=std::thread (f,i,std::ref(V_tmp),num0rew,sz1*(Tnum-1),sz,std::ref(states),Rew);
            //std::thread th1(f,i,std::ref(V_tmp),num0rew,0,sz1,std::ref(states),Rew);
            //std::thread th2(f,i,std::ref(V_tmp),num0rew,sz1,sz1*2,std::ref(states),Rew);
            //std::thread th3(f,i,std::ref(V_tmp),num0rew,sz1*2,sz1*3,std::ref(states),Rew);
            //std::thread th4(f,i,std::ref(V_tmp),num0rew,sz1*3,sz1*4,std::ref(states),Rew);
            //std::thread th5(f,i,std::ref(V_tmp),num0rew,sz1*4,sz,std::ref(states),Rew);
            //std::thread th2(f,k,std::ref(V_tmp),num0rew,sz1+1,sz,std::ref(states));
            //MTpC(i,V_tmp,num0rew,sz1*4,sz);
            for (int d=0;d<Tnum;d++)
                th[d].join();
            //th1.join();
            //th2.join();
            //th3.join();
            //th4.join();
            //th5.join();
            V_tmp = getStateValueFunction();  
            //stack_memory++;
            //checkStackSize();
            checkMemoryUsage();
            }
            action_stack.push(getStateActions());
            //action_stack.push(getStateActions());
        return V_tmp[initial_state_num];
    }


float calculatePolicyPrunningMT(int k,int Tnum=4){
        vector<float> V_tmp;
        int sz=states.size();
        int sz1=(int) sz/(Tnum+1);
        //V_tmp.reserve(states.size());      
        V_tmp = getStateValueFunction();
        std::thread th[Tnum];
        //vector<float> V_tmp1=getStateValueFunction();
        long double num0rew=0;
            auto f = [](int i,vector<float>& V_tmp,long double num0rew,int s,int sz2,std::vector<State> &states) {
                        int statenum;
        
        for (int j = s ; j < sz2; j++ ){ //FOR EVERY STATE   
                     
            for (int n = 0; n < states[j].get_qstates().size(); n++){ //FOR EVERY QSTATE OF EACH STATE
                long double new_qvalue = 0.0;
                float r;
                float t;
                if (states[j].qstates[n].num_taken==0)
                    states[j].qstates[n].set_qvalue(num0rew);
                else{
                    for (int m=0; m < states[j].qstates[n].trans.size(); m++){ //FOR EVERY ACCESIBLE STATE FROM CURRENT QSTATE
                        t = states[j].qstates[n].trans[m];
                        statenum=states[j].qstates[n].transtate[m];
                       //r = states[j].qstates[n].get_reward(statenum, i);
                        r = states[j].qstates[n].get_reward(statenum);
                        new_qvalue += t * (r +V_tmp[statenum]);
                    }
                    states[j].qstates[n].set_qvalue(new_qvalue);
                }
            }
            states[j].update_value();
        }
            };


        for (int i = 1 ; i < k+1; i++){ //FOR EVERY INDEX UP TO THE HORIZON
            pair <float,float> minmax;
            num0rew=calcrewa(V_tmp);  
            //cout << "Horizon Average Value " << num0rew << endl;           
            //std::thread th1(MTpC,k,V_tmp,num0rew,0,sz1);
            //std::thread th2(MTpC,k,V_tmp,num0rew,sz1+1,sz);
            for (int d=0;d<Tnum;d++)
                th[d]=std::thread (f,i,std::ref(V_tmp),num0rew,sz1*d,sz1*(d+1),std::ref(states));
             //th[Tnum-1]=std::thread th1(f,i,std::ref(V_tmp),num0rew,sz1*(Tnum-1),sz,std::ref(states),Rew);
            //std::thread th1(f,i,std::ref(V_tmp),num0rew,0,sz1,std::ref(states));
            //std::thread th2(f,i,std::ref(V_tmp),num0rew,sz1,sz1*2,std::ref(states));
            //std::thread th3(f,i,std::ref(V_tmp),num0rew,sz1*2,sz1*3,std::ref(states));
            //std::thread th4(f,i,std::ref(V_tmp),num0rew,sz1*3,sz1*4,std::ref(states));

            //std::thread th2(f,k,std::ref(V_tmp),num0rew,sz1+1,sz,std::ref(states));
            MTpC(i,V_tmp,num0rew,sz1*Tnum,sz);
            for (int d=0;d<Tnum;d++)
                th[d].join();
            //th1.join();
            //th2.join();
            //th3.join();
            //th4.join();
            V_tmp = getStateValueFunction();
            //minmax=calcMinMax(1000000000,-100000,minmax);
            //cout << "Horizon Min and Max Value " << minmax.first<<" , "<< minmax.second << endl;
            action_stack.push(getStateActions());
            stack_memory++;
            checkStackSize();
            checkMemoryUsage();
            }
            //action_stack.push(getStateActions());
        return V_tmp[initial_state_num];
    }

  float calculatePolicyPrunningMTMR(int k){
        vector<float> V_tmp;
        int sz=states.size();
        int sz1=(int) sz/5;
        //V_tmp.reserve(states.size());      
        V_tmp = getStateValueFunction();
        //vector<float> V_tmp1=getStateValueFunction();
        long double num0rew=0;
            auto f = [](int i,vector<float>& V_tmp,long double num0rew,int s,int sz2,std::vector<State> &states) {
                        int statenum;
        
        for (int j = s ; j < sz2; j++ ){ //FOR EVERY STATE   
                     
            for (int n = 0; n < states[j].get_qstates().size(); n++){ //FOR EVERY QSTATE OF EACH STATE
                long double new_qvalue = 0.0;
                float r;
                float t;
                if (states[j].qstates[n].num_taken==0)
                    states[j].qstates[n].set_qvalue(num0rew);
                else{
                    for (int m=0; m < states[j].qstates[n].trans.size(); m++){ //FOR EVERY ACCESIBLE STATE FROM CURRENT QSTATE
                        t = states[j].qstates[n].trans[m];
                        statenum=states[j].qstates[n].transtate[m];
                       r = states[j].qstates[n].get_reward(statenum, i);
                      
                        new_qvalue += t * (r +V_tmp[statenum]);
                    }
                    states[j].qstates[n].set_qvalue(new_qvalue);
                }
            }
            states[j].update_value();
        }
            };


        for (int i = 1 ; i < k+1; i++){ //FOR EVERY INDEX UP TO THE HORIZON
            pair <float,float> minmax;
            num0rew=calcrewa(V_tmp);  
            //cout << "Horizon Average Value " << num0rew << endl;           
            //std::thread th1(MTpC,k,V_tmp,num0rew,0,sz1);
            //std::thread th2(MTpC,k,V_tmp,num0rew,sz1+1,sz);

            std::thread th1(f,i,std::ref(V_tmp),num0rew,0,sz1,std::ref(states));
            std::thread th2(f,i,std::ref(V_tmp),num0rew,sz1,sz1*2,std::ref(states));
            std::thread th3(f,i,std::ref(V_tmp),num0rew,sz1*2,sz1*3,std::ref(states));
            std::thread th4(f,i,std::ref(V_tmp),num0rew,sz1*3,sz1*4,std::ref(states));

            //std::thread th2(f,k,std::ref(V_tmp),num0rew,sz1+1,sz,std::ref(states));
            MTpCMR(i,V_tmp,num0rew,sz1*4,sz);
            th1.join();
            th2.join();
            th3.join();
            th4.join();
            V_tmp = getStateValueFunction();
            //minmax=calcMinMax(1000000000,-100000,minmax);
            //cout << "Horizon Min and Max Value " << minmax.first<<" , "<< minmax.second << endl;
            action_stack.push(getStateActions());
            stack_memory++;
            checkStackSize();
            checkMemoryUsage();
            }
            //action_stack.push(getStateActions());
        return V_tmp[initial_state_num];
    }  
  

pair<float,float>  calcMinMax(float minvalue,float maxvalue,pair<float,float> minmax)
{
    for (int i=0;i<states.size();i++)
        for (int j=0;j<states[i].qstates.size();j++){
            if (states[i].qstates[j].qvalue<minvalue)
                    minvalue=states[i].qstates[j].qvalue;
            if (states[i].qstates[j].qvalue>maxvalue)
                    maxvalue=states[i].qstates[j].qvalue;
        }
        minmax.first=minvalue;
        minmax.second=maxvalue;
    return minmax;

}


    void MTpC(int i,vector<float>& V_tmp,long double num0rew,int s,int sz2){
        int statenum;
        for (int j = s ; j < sz2; j++ ){ //FOR EVERY STATE   
                     
            for (int n = 0; n < states[j].get_qstates().size(); n++){ //FOR EVERY QSTATE OF EACH STATE
                long double new_qvalue = 0.0;
                float r;
                float t;
                if (states[j].qstates[n].num_taken==0)
                    states[j].qstates[n].set_qvalue(num0rew);
                else{
                    for (int m=0; m < states[j].qstates[n].trans.size(); m++){ //FOR EVERY ACCESIBLE STATE FROM CURRENT QSTATE
                        t = states[j].qstates[n].trans[m];
                        statenum=states[j].qstates[n].transtate[m];
                        //r = states[j].qstates[n].get_reward(statenum, i);
                        r = states[j].qstates[n].get_reward(statenum);
                        new_qvalue += t * (r +V_tmp[statenum]);
                    }
                    states[j].qstates[n].set_qvalue(new_qvalue);
                }
            }
            states[j].update_value();
        }                 
        
}
void MTpCMR(int i,vector<float>& V_tmp,long double num0rew,int s,int sz2){
        int statenum;
        for (int j = s ; j < sz2; j++ ){ //FOR EVERY STATE   
                     
            for (int n = 0; n < states[j].get_qstates().size(); n++){ //FOR EVERY QSTATE OF EACH STATE
                long double new_qvalue = 0.0;
                float r;
                float t;
                if (states[j].qstates[n].num_taken==0)
                    states[j].qstates[n].set_qvalue(num0rew);
                else{
                    for (int m=0; m < states[j].qstates[n].trans.size(); m++){ //FOR EVERY ACCESIBLE STATE FROM CURRENT QSTATE
                        t = states[j].qstates[n].trans[m];
                        statenum=states[j].qstates[n].transtate[m];
                        r = states[j].qstates[n].get_reward(statenum, i);
                        
                        new_qvalue += t * (r +V_tmp[statenum]);
                    }
                    states[j].qstates[n].set_qvalue(new_qvalue);
                }
            }
            states[j].update_value();
        }
 }

    float calculatePolicy(int k){
        vector<float> V_tmp;
        //V_tmp.reserve(states.size());      
        V_tmp = getStateValueFunction();
        for (int i = 1 ; i < k+1; i++){ //FOR EVERY INDEX UP TO THE HORIZON
            for (int j = 0 ; j < states.size(); j++ ){ //FOR EVERY STATE
                for (int n = 0; n < states[j].get_qstates().size(); n++){ //FOR EVERY QSTATE OF EACH STATE
                    long double new_qvalue = 0.0;
                    long double r;
                    long double t;
                    for (int m=0; m < V_tmp.size(); m++){ //FOR EVERY ACCESIBLE STATE FROM CURRENT QSTATE
                        t = states[j].qstates[n].get_transition(m);
                        //r = states[j].qstates[n].get_reward(m, i);
                        r = states[j].qstates[n].get_reward(m);
                        new_qvalue += t * (r +V_tmp[m]);
                    }

                    states[j].qstates[n].set_qvalue(new_qvalue);
                }
                states[j].update_value();
            }
            V_tmp = getStateValueFunction();

            action_stack.push(getStateActions());
            stack_memory++;
            checkStackSize();
            checkMemoryUsage();
        }
        return V_tmp[initial_state_num];
    }
    long double calcrewa(vector<float> V_tmp){
        long double new_qvalue=0;
        long double x=1.0/(long double)states.size();
        for (int m=0; m < V_tmp.size(); m++){ //FOR EVERY ACCESIBLE STATE FROM CURRENT QSTATE
                        //new_qvalue += (V_tmp[m]);
                        new_qvalue += x*(V_tmp[m]);
                    }
                    //x=(float)(new_qvalue/(float)states.size());
                    return new_qvalue;
        //return x;
    }
    long double calcrewa1(vector<float> V_tmp){
        float new_qvalue=0;
        float x=1.0/(long double)states.size();
        for (int m=0; m < V_tmp.size(); m++){ //FOR EVERY ACCESIBLE STATE FROM CURRENT QSTATE
                        //new_qvalue += (V_tmp[m]);
                        new_qvalue += x*(V_tmp[m]);
                    }
                    //x=(float)(new_qvalue/(float)states.size());
                    return new_qvalue;
        //return x;
    }


    float calcrewa(vector<pair<int,float>> V_tmp){
        float new_qvalue=0;
        float x=0.0;
        for (int m=0; m < V_tmp.size(); m++){ //FOR EVERY ACCESIBLE STATE FROM CURRENT QSTATE
                        new_qvalue += (V_tmp[m].second);
                    }
        x=(float)(new_qvalue/(float)states.size());
        return x;
    }
   
    /*
    Runs the Naive Finite-Horizon MDP method using calculatePolicy.
    Takes as argument the Finite-Horizon MDP's horizon;
    */

    void naiveEvaluation(int horizon){
        resetValueFunction();
        expected_reward=calculatePolicy(horizon);
        int actiont=-1;
        vector<int> V;
        steps_made = 0;
        //changesEval(action_stack);
        while (!action_stack.empty()){
            loadBestQStates(action_stack.top());
            //takeAction(false,horizon - steps_made);
            takeAction(false);
            action_stack.pop();
            steps_made++;
        }
    }

    void naiveEvaluationPr(int horizon){
        resetValueFunction();
        expected_reward = calculatePolicyPrunning(horizon);
        int actiont=-1;
        vector<int> V;
        steps_made = 0;
        while (!action_stack.empty()){
            V=action_stack.top();
            actiont=V[current_state_num];
            takeActionPrunning(actiont);
            action_stack.pop();
            steps_made++;
        }
    }


    void naiveEvaluationMT(int horizon){
        resetValueFunction();
        expected_reward = calculatePolicyPrunningMT(horizon);
        int actiont=-1;
        vector<int> V;
        steps_made = 0;
        while (!action_stack.empty()){
            V=action_stack.top();
            actiont=V[current_state_num];
            takeActionPrunning(actiont);
            action_stack.pop();
            steps_made++;
        }
    }
        void naiveEvaluationMTMR(int horizon){
        resetValueFunction();
        expected_reward = calculatePolicyPrunningMTMR(horizon);
        int actiont=-1;
        vector<int> V;
        steps_made = 0;
        while (!action_stack.empty()){
            V=action_stack.top();
            actiont=V[current_state_num];
            takeActionPrunning(actiont, horizon - steps_made);
            action_stack.pop();
            steps_made++;
        }
    }
void changesEval(stack<vector<int>>  policy){
    vector<int> newP;
    vector<int> old=policy.top();
    policy.pop();
    int lvlcounter=0;
    while (!policy.empty()){
        newP=policy.top();
        policy.pop();
        for (int i=0;i<states.size();i++)
            if (newP[i]!=old[i])
                lvlcounter++;
        old=newP;        
    	}
        cout <<lvlcounter << endl;
     }

    void inPlaceEvaluation(int horizon){
        vector<pair<int,float>> V;
        //V_temp.reserve(states.size()); 
        //V.reserve(states.size()); 
        int steps_remaining = horizon;
        resetValueFunction();
        V =getStateValuestest(states);
        calculateValuesPrunningR(steps_remaining, 0, V,true);
        //calculateValuestest(steps_remaining, 0, V,true);
        expected_reward = V[initial_state_num].second;
        loadValueFunctiontest(V);
        //takeAction(false, horizon);
        takeActionPrunning(V[current_state_num].first);
        steps_made++;
        steps_remaining--;
        if (getValue() > max_memory_used){
            max_memory_used = getValue();
        }
        while(steps_remaining > 0){
            resetValueFunction();
            V=getStateValuestest(states);
            calculateValuesPrunningR(steps_remaining, 0, V,true);
            loadValueFunctiontest(V);
            //takeAction(false, steps_remaining);
            takeActionPrunning(V[current_state_num].first);
            steps_made++;
            steps_remaining--;
        }
    }   

void rootEvaluation(int horizon){

        vector<pair<int,float>> V;
        int steps_remaining = horizon;
        int floor_of_square_root = floor(sqrt(horizon));
        int i=0;
        V=getStateValuestest(states);
        for ( ;i+floor_of_square_root <= horizon; i=i+floor_of_square_root){
            calculateValuestest(i+floor_of_square_root,i,  V, true);
            finite_stack.push(V);
        }
        if (i!=horizon){
            V=getStateValuestest(states);
            calculateValuestest(horizon, i, V);
        }
        expected_reward = V[initial_state_num].second;
        checkMemoryUsage();
        for ( ;i < horizon; i=i+1){ 
           
            loadValueFunction(V);
            takeAction(false);
            finite_stack.pop();
            steps_remaining--;
            V = finite_stack.top();  
        }
        while(steps_remaining > 0){
            if (finite_stack.empty()){
                resetValueFunction();
                V=getStateValuestest(states);
                calculateValuestest(steps_remaining, 0, V);
            }
            else{
                if( (steps_remaining+1)%floor_of_square_root==0){
                    V=finite_stack.top();
                    calculateValuestest(steps_remaining, steps_remaining-floor_of_square_root+1,V );
                    checkMemoryUsage();
                }
                else
                    V = finite_stack.top();            
            }
            loadValueFunction(V);
            takeAction(false);
            checkMemoryUsage();
            finite_stack.pop();
            steps_remaining--;
        }
    }

void rootEvaluationPrun(int horizon){

        vector<pair<int,float>> V;
        int steps_remaining = horizon;
        int actiont=-1;
        resetValueFunction();
        int floor_of_square_root = floor(sqrt(horizon));
        int i=0;
        V=getStateValuestest(states);
        for ( ;i+floor_of_square_root <= horizon; i=i+floor_of_square_root){
            calculateValuesPrunningR(i+floor_of_square_root,i,  V, true);
            finite_stack.push(V);
        }
        if (i!=horizon){
            V=getStateValuestest(states);
            calculateValuesPrunningR(horizon, i, V);
        }
        expected_reward = V[initial_state_num].second;
        checkMemoryUsage();
        for ( ;i < horizon; i=i+1){ 
           
            loadValueFunction(V);
            takeActionPrunning(V[current_state_num].first);
            //takeActionPrunning(V[current_state_num].first, steps_remaining);
            //takeAction(false, horizon-i);
            finite_stack.pop();
            steps_remaining--;
            V = finite_stack.top();  
        }
        while(steps_remaining > 0){
            if (finite_stack.empty()){
                resetValueFunction();
                V=getStateValuestest(states);
                calculateValuesPrunningR(steps_remaining, 0, V);
                //calculateValuesPrunningRAU(steps_remaining, 0, V);
            }
            else{

                if( (steps_remaining+1)%floor_of_square_root==0){
                    V=finite_stack.top();
                    calculateValuesPrunningR(steps_remaining, steps_remaining-floor_of_square_root+1,V );
                    //calculateValuesPrunningRAU(steps_remaining, steps_remaining-floor_of_square_root+1,V );
                    checkMemoryUsage();
                }
                else
                    V = finite_stack.top();            
            }
            loadValueFunctiontest(V);
            takeActionPrunning(V[current_state_num].first);
            checkMemoryUsage();
            finite_stack.pop();
            steps_remaining--;
        }
    }

    void printValueFunction(){
        if(!finite_stack.empty()){
            for (int i = 0; i < finite_stack.top().size(); i++){
                cout << "State " << i << ": " << finite_stack.top()[i].second << endl;
            }
        }
        else{
            for (int i = 0; i < states.size(); i++){
                cout << "State " << states[i].get_state_num()<< ": " << states[i].value << endl;
            }  
        }
    }

    void resetValueFunction(){
        for (int i=0; i<states.size(); i++){
            for (int j=0; j < states[i].qstates.size(); j++){
                states[i].qstates[j].qvalue = 0.0;
            }
            states[i].value = 0.0;
        }
    }


    void resetModel(){
        resetValueFunction();
        current_state_num = initial_state_num;
        total_reward = 0.0;
        expected_reward = 0.0;
        max_memory_used = 0.0;
        steps_made = 0;
        stack_memory = 0;
    }


    //calculates Value Function for target index while saving every intermediate index needed
    void treeTraversal(int target, int horizon, vector<pair<int,float>> &V){
        int l = 0;
        int r = horizon;

        //V.reserve(states.size());
       // V_tmp.reserve(states.size());
        int k = (l + r)/2;
        if (!index_stack.empty()){
            if (index_stack.top() == target){
                V = finite_stack.top();
                finite_stack.pop();
                index_stack.pop();
                //stack_memory--;
                return ;
            }
            else{
                k = index_stack.top();
            }
        }


        while ( l <= r){
            if (k == target){
                if (finite_stack.empty()){
                    resetValueFunction();//if no vector is saved in memory, calculate objective from the beginning
                    V=getStateValuestest(states);
                    calculateValuestest(k, 0, V, true);

                }
                else{
                    V=finite_stack.top();
                    calculateValuestest(k, index_stack.top(),V , true);//use last saved vector in memory to calculate objective
                    
                }
                break;
            }
            else if ( k < target){
                if (finite_stack.empty()){
                    resetValueFunction();//if no vector is saved in memory, calculate objective from the beginning
                    V=getStateValuestest(states);
                    calculateValuestest(k, 0, V, true);
                    finite_stack.push(V);
                    index_stack.push(k);
                    //stack_memory++;
                    //checkStackSize();
                }
                else{
                    if (index_stack.top() != k){
                        V=finite_stack.top();
                        calculateValuestest(k, index_stack.top(), V, true);
                        finite_stack.push(V);//use last saved vector in memory to calculate objective
                        index_stack.push(k);
                        //stack_memory++;
                        //checkStackSize();
                    }
                }
                l = k + 1;
                k = (l + r)/2;
            }
            else if ( k > target){
                r = k - 1;
                k = (l + r)/2;
            }
        }
        checkMemoryUsage();
        
    }
    void treeTraversalPrunnA(int target, int horizon, vector<pair<int,float>> &V){
        int l = 0;
        int r = horizon;

        //V.reserve(states.size());
       // V_tmp.reserve(states.size());
        int k = (l + r)/2;
        if (!index_stack.empty()){
            if (index_stack.top() == target){
                V = finite_stack.top();
                finite_stack.pop();
                index_stack.pop();
                //stack_memory--;
                return ;
            }
            else{
                k = index_stack.top();
            }
        }


        while ( l <= r){
            if (k == target){
                if (finite_stack.empty()){
                    resetValueFunction();//if no vector is saved in memory, calculate objective from the beginning
                    V=getStateValuestest(states);
                    calculateValuesPrunningA(k, 0, V, true);

                }
                else{
                    V=finite_stack.top();
                    calculateValuesPrunningA(k, index_stack.top(),V , true);//use last saved vector in memory to calculate objective
                    
                }
                break;
            }
            else if ( k < target){
                if (finite_stack.empty()){
                    resetValueFunction();//if no vector is saved in memory, calculate objective from the beginning
                    V=getStateValuestest(states);
                    calculateValuesPrunningA(k, 0, V, true);
                    finite_stack.push(V);
                    index_stack.push(k);
                    //stack_memory++;
                    //checkStackSize();
                }
                else{
                    if (index_stack.top() != k){
                        V=finite_stack.top();
                        calculateValuesPrunningA(k, index_stack.top(), V, true);
                        finite_stack.push(V);//use last saved vector in memory to calculate objective
                        index_stack.push(k);
                        //stack_memory++;
                        //checkStackSize();
                    }
                }
                l = k + 1;
                k = (l + r)/2;
            }
            else if ( k > target){
                r = k - 1;
                k = (l + r)/2;
            }
        }
        checkMemoryUsage();
        
    }
    void treeTraversalPrunn(int target, int horizon, vector<pair<int,float>> &V){
        int l = 0;
        int r = horizon;

        //V.reserve(states.size());
       // V_tmp.reserve(states.size());
        int k = (l + r)/2;
        if (!index_stack.empty()){
            if (index_stack.top() == target){
                V = finite_stack.top();
                finite_stack.pop();
                index_stack.pop();
                //stack_memory--;
                return ;
            }
            else{
                k = index_stack.top();
            }
        }


        while ( l <= r){
            if (k == target){
                if (finite_stack.empty()){
                    resetValueFunction();//if no vector is saved in memory, calculate objective from the beginning
                    V=getStateValuestest(states);
                    calculateValuesPrunning(k, 0, V, true);

                }
                else{
                    V=finite_stack.top();
                    calculateValuesPrunning(k, index_stack.top(),V , true);//use last saved vector in memory to calculate objective
                    
                }
                break;
            }
            else if ( k < target){
                if (finite_stack.empty()){
                    resetValueFunction();//if no vector is saved in memory, calculate objective from the beginning
                    V=getStateValuestest(states);
                    calculateValuesPrunning(k, 0, V, true);
                    finite_stack.push(V);
                    index_stack.push(k);
                    //stack_memory++;
                    //checkStackSize();
                }
                else{
                    if (index_stack.top() != k){
                        V=finite_stack.top();
                        calculateValuesPrunning(k, index_stack.top(), V, true);
                        finite_stack.push(V);//use last saved vector in memory to calculate objective
                        index_stack.push(k);
                        //stack_memory++;
                        //checkStackSize();
                    }
                }
                l = k + 1;
                k = (l + r)/2;
            }
            else if ( k > target){
                r = k - 1;
                k = (l + r)/2;
            }
        }
        checkMemoryUsage();
        
    }

    void treeTraversalPrunnAu(int target, int horizon, vector<pair<int,float>> &V){
        int l = 0;
        int r = horizon;

        //V.reserve(states.size());
       // V_tmp.reserve(states.size());
        int k = (l + r)/2;
        if (!index_stack.empty()){
            if (index_stack.top() == target){
                V = finite_stack.top();
                finite_stack.pop();
                index_stack.pop();
                //stack_memory--;
                return ;
            }
            else{
                k = index_stack.top();
            }
        }


        while ( l <= r){
            if (k == target){
                if (finite_stack.empty()){
                    resetValueFunction();//if no vector is saved in memory, calculate objective from the beginning
                    V=getStateValuestest(states);
                    //calculateValuesPrunning(k, 0, V, true);
                    calculateValuestestRec(k, 0, V, true);
                    //calculateValuestest(k, 0, V, true);

                }
                else{
                    V=finite_stack.top();
                    //calculateValuesPrunningRecal(k, index_stack.top(),V , true);//use last saved vector in memory to calculate objective
                    calculateValuestestRec(k, index_stack.top(),V , true);
                    //calculateValuestest(k, index_stack.top(),V , true);
                }
                break;
            }
            else if ( k < target){
                if (finite_stack.empty()){
                    resetValueFunction();//if no vector is saved in memory, calculate objective from the beginning
                    V=getStateValuestest(states);
                    //calculateValuesPrunning(k, 0, V, true);
                    calculateValuestestRec(k, 0, V, true);
                    //calculateValuestest(k, 0, V, true);
                    finite_stack.push(V);
                    index_stack.push(k);
                    //stack_memory++;
                    //checkStackSize();
                }
                else{
                    if (index_stack.top() != k){
                        V=finite_stack.top();
                        //calculateValuesPrunningRecal(k, index_stack.top(), V, true);
                        calculateValuestestRec(k, index_stack.top(), V, true);
                        //calculateValuestest(k, index_stack.top(), V, true);
                        finite_stack.push(V);//use last saved vector in memory to calculate objective
                        index_stack.push(k);
                        //stack_memory++;
                        //checkStackSize();
                    }
                }
                l = k + 1;
                k = (l + r)/2;
            }
            else if ( k > target){
                r = k - 1;
                k = (l + r)/2;
            }
        }
        checkMemoryUsage();
        
    }

    void treeEvaluation(int horizon){
        int steps_remaining = horizon;
        vector<pair<int,float>> V;
        //V.reserve(states.size());
        while(steps_remaining > 0){
            treeTraversal(steps_remaining, horizon,V);
            if (steps_remaining == horizon)
                expected_reward = V[initial_state_num].second;
            loadValueFunctiontest(V);
            //takeAction(false, steps_remaining);
            takeAction(false);
            checkMemoryUsage();
            steps_remaining--;
        }
    }
    void treeEvaluationPrunn(int horizon){
        int steps_remaining = horizon;
        int actiont=-1;
        vector<pair<int,float>> V;
        //V.reserve(states.size());
        treeTraversalPrunn(steps_remaining, horizon,V);
        expected_reward = V[initial_state_num].second;
        loadValueFunctiontest(V);
        takeActionPrunning(V[current_state_num].first);
        checkMemoryUsage();
            steps_remaining--;
        while(steps_remaining > 0){

            treeTraversalPrunn(steps_remaining, horizon,V);
            loadValueFunctiontest(V);
            takeActionPrunning(V[current_state_num].first);
            checkMemoryUsage();
            steps_remaining--;
        }
    }

    void treeEvaluationRec(int horizon){
        int steps_remaining = horizon;
        int actiont=-1;
        vector<pair<int,float>> V;
        //V.reserve(states.size());
        treeTraversalPrunnA(steps_remaining, horizon,V);
        expected_reward = V[initial_state_num].second;
        loadValueFunctiontest(V);
        takeActionPrunning(V[current_state_num].first);
        checkMemoryUsage();
            steps_remaining--;
        while(steps_remaining > 0){

            treeTraversalPrunnAu(steps_remaining, horizon,V);
            loadValueFunctiontest(V);
            takeActionPrunning(V[current_state_num].first);
            checkMemoryUsage();
            steps_remaining--;
        }
    }
    
    void infiniteEvaluation(int horizon){
        resetValueFunction();
        max_memory_used=value_iteration(0.1, false);
        max_memory_used = getValue();
        expected_reward = states[initial_state_num].value;
        for (int time = horizon; time > 0; time--){
            takeAction(true);
            }
    }

        void infiniteEvaluationMR(int horizon){
        resetValueFunction();
        max_memory_used=value_iteration(0.1, false);
        max_memory_used = getValue();
        expected_reward = states[initial_state_num].value;
        for (int time = horizon; time > 0; time--){
            takeAction(true, time);
            
            }
    }
    
    void turnpikeInfinite(int horizon){
        vector<pair<int,float>> V;
        vector<int> K;
        int actiont=-1;
        int steps_remaining = horizon;
        resetValueFunction();
        V =getStateValuestest(states);
        calculateValuesPrunningR(steps_remaining, 0, V,true);
        expected_reward = V[initial_state_num].second;
        loadValueFunctiontest(V);
        takeActionPrunning(V[current_state_num].first);
        steps_made++;
        steps_remaining--;
        if (getValue() > max_memory_used){
            max_memory_used = getValue();
        }
        while(steps_remaining > 0){
            
            takeActionPrunning(V[current_state_num].first);
            //takeActionPrunning(actiont, horizon - steps_made);
            steps_made++;
            steps_remaining--;
        }
    }
    void turnpikeInfiniteC(int horizon,bool Rew){
        int steps_remaining = horizon;
        resetValueFunction();
        resetModel();
        expected_reward = calculatePolicyPrunningMTTurnpike(horizon,Rew);
        cout << action_stack.size() << endl;
        //expected_reward = calculatePolicyPrunning(horizon);
        int actiont=-1;
        vector<int> V;
        V=action_stack.top();
        steps_made = 0;
        //changesEval(action_stack);
        while(steps_remaining > 0){
            actiont=V[current_state_num];
            if (Rew)
                takeActionPrunning(actiont);
            else
                takeActionPrunning(actiont,steps_remaining);
            steps_remaining--;
            steps_made++;
        }
    }
    void turnpikeInfiniteCMR(int horizon){
        int steps_remaining = horizon;
        resetValueFunction();
        resetModel();
        while(!action_stack.empty())
            action_stack.pop();
        expected_reward = calculatePolicyPrunningMTMR(horizon);
        cout << action_stack.size() << endl;
        //expected_reward = calculatePolicyPrunning(horizon);
        int actiont=-1;
        vector<int> V;
        V=action_stack.top();
        steps_made = 0;
        //changesEval(action_stack);
        while(steps_remaining > 0){
            actiont=V[current_state_num];
            takeActionPrunning(actiont,steps_remaining);
            action_stack.pop();
            steps_remaining--;
            steps_made++;
        }
    }


    void runAlgorithm(model_type alg, int horizon=100){

        auto start = high_resolution_clock::now();
        switch(alg) {   
            case infinite:
                cout << "INFINITE MDP MODEL: " << endl;
                init_memory_used = getValue();
                infiniteEvaluation(horizon);
                break;
            case infiniteMR:
                cout << "INFINITE MDP MODEL: " << endl;
                init_memory_used = getValue();
                infiniteEvaluationMR(horizon);
                break;

            case infiniteM:
                cout << "INFINITEM MDP MODEL: " << endl;
                init_memory_used = getValue();
                //infiniteEvaluationM(horizon);
                turnpikeInfinite(horizon);
                break;   

            case infiniteMT:
                cout << "INFINITEM MDP MODEL: " << endl;
                init_memory_used = getValue();
                //infiniteEvaluationM(horizon);
                turnpikeInfiniteC(horizon,true);
                break; 

            case infiniteMTMR:
                cout << "INFINITEM MDP MODEL: " << endl;
                init_memory_used = getValue();
                turnpikeInfiniteC(horizon,false);
                break; 

            case naive:
                cout << "NAIVE FINITE MDP MODEL: " << endl;
                init_memory_used = getValue();
                naiveEvaluation(horizon);
                break;

            case naivePr:
                cout << "NAIVE FINITE MDP MODEL: " << endl;
                init_memory_used = getValue();
                naiveEvaluationPr(horizon);
                break;

            case naiveMT:
                cout << "NAIVE FINITE MDP MODEL: " << endl;
                init_memory_used = getValue();
                naiveEvaluationMT(horizon);
                break;

            case naiveMTMR:
                cout << "NAIVE FINITE MDP MODEL: " << endl;
                init_memory_used = getValue();
                naiveEvaluationMTMR(horizon);
                break;

            case root:
                cout << "ROOT FINITE MDP MODEL: " << endl;
                init_memory_used = getValue();
                rootEvaluation(horizon);
                break;

            case rootPrun:
                cout << "ROOT FINITE MDP MODEL: " << endl;
                init_memory_used = getValue();
                rootEvaluationPrun(horizon);
                break;

            case tree:
                cout << "TREE FINITE MDP MODEL: " << endl;
                init_memory_used = getValue();
                treeEvaluation(horizon);
                break;

            case treePrun:
                cout << "TREE FINITE MDP MODEL: " << endl;
                init_memory_used = getValue();
                treeEvaluationPrunn(horizon);
                break;

            case treeRec:
                cout << "TREE FINITE MDP MODEL: " << endl;
                init_memory_used = getValue();
                treeEvaluationRec(horizon);
                break;

            case inplace:
                cout << "IN-PLACE FINITE MDP MODEL: " << endl;
                init_memory_used = getValue();
                inPlaceEvaluation(horizon);
                break;

            default:
                cout << "Invalid Model Type. Valid model types are: infinite, naive, root, tree, inplace" << endl;
                return;
        }

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        cout << "Horizon size: " << horizon << endl;
        cout << "Total Reward Expected: " << expected_reward << endl;
        cout << "Total Reward Collected: " << total_reward << endl;
        cout << "Execution time (sec): " << duration.count() * 0.000001 << endl;
        cout << "Peak memory used (MB): " << (max_memory_used) / 1000000.0 << endl;
        cout << "Peak memory used (MB): " << (init_memory_used) / 1000000.0 << endl;
        cout << endl;
    }



};
