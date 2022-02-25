#include <iostream>
#include <vector>
#include <map>
#include <stdexcept>
#include <string>
#include <stack>
#include <sstream>
#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include <random>
#include <math.h>
#include <array>
#include <chrono>
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <nlohmann\json.hpp>
#endif

#ifdef linux
#include <nlohmann/json.hpp>
#define SIZE_T int
#endif

#ifdef _WIN32
#include <nlohmann\json.hpp>
#endif

#ifdef linux
#include <nlohmann/json.hpp>
#endif

// for convenience
using json = nlohmann::json;

using namespace std;

int parseLine(char* line){
    // This assumes that a digit will be found and the line ends in " Kb".
    int i = strlen(line);
    const char* p = line;
    while (*p <'0' || *p > '9') p++;
    line[i-3] = '\0';
    i = atoi(p);
    return i;
}

string printableParameters(map<string,pair<float,float>> params){
    string s = "[";
    for (auto& element:params){
        s = s + "(";
        s = s + element.first + "," + to_string(element.second.first) + "," + to_string(element.second.second) + ") ";
    }
    s = s + "]";
    return s;
}
SIZE_T getValue1(){
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
        if (strncmp(line, "VmRSS:", 6) == 0){
            result = parseLine(line);
            break;
        }
    }
    fclose(file);
    return result;
#endif

}

//class State;

class QState{
public:
    pair<string,int> action;
    int num_taken;
    float qvalue;
    vector<int> transitions={};
    vector<float> rewards = {};
    int num_states;
    vector<int> transtate={};
    vector<float> trans={};
    int lastused=-1;
    float upper_bound;//Maximum value of QStates for a given state
    float lower_bound;//Minimum value of QStates for a given state

    QState(pair<string,int> actionn, int numstates, float qvaluee){
        action = actionn;
        num_taken = 0;
        qvalue = qvaluee;
        num_states = numstates;
        if (num_states > 0){
            for (int i=0; i< num_states;i++){
            transitions.push_back(0);
            rewards.push_back(0.0);
            }
        }

    }

    QState(){
        std::string s0 ("NOT SET");
        action.first = s0;
        action.second = -1;
        num_taken = 0;
        qvalue = 0.0;
        num_states = -1;
        if (num_states > 0){
            for (int i=0; i< num_states;i++){
            transitions.push_back(0);
            rewards.push_back(0.0);
            }
        }

    }

    ~QState(){
        transitions.clear();
        transitions.shrink_to_fit();
        rewards.clear();
        rewards.shrink_to_fit();
    }

    void update(int state_num, float reward){
        num_taken++;
        transitions[state_num] += 1;
        rewards[state_num] += reward;
    }


    pair<string, int> get_action(){
        return action;
    }

    float get_qvalue(){
        return qvalue;
    }

    bool has_transition(int state_num){
        return (transitions[state_num] > 0);
    }


    float get_transition(int state_num){
        if (num_taken == 0){
            return (1.0 /(float)num_states);
        }
        else{
            return ((float)transitions[state_num]*1.0 /(float)num_taken*1.0);
        }
    }

    int get_num_transitions(int state_num){
        return transitions[state_num];
    }


    float get_reward(int state_num){
        if (transitions[state_num] == 0)
            return 0.0;
        else
            return (rewards[state_num] / (float)transitions[state_num]);
    }

    float get_reward(int state_num, int time_step, float reward_factor = 0.8){

        if (transtate.size() < 2){
            return get_reward(state_num);
        }
        

        float static_reward;
        if (transitions[state_num] == 0)
            static_reward = 0.0;
        else
            static_reward = (rewards[state_num] / (float)transitions[state_num]);
        
        //if (state_num == transtate[0]){
            //return reward_factor * static_reward;
        //}
        if (state_num == transtate[time_step % transtate.size()]){
           //return static_reward + ((1 - reward_factor)* get_reward(transtate[(time_step-1) % transtate.size()]) * trans[0]) / (get_transition(state_num));
           //return static_reward + ((1 - reward_factor) * get_reward(transtate[0]) * trans[0]) / (get_transition(state_num));
            //static_reward + ((1 - reward_factor) * get_reward(transtate[0]) * trans[0]) / (get_transition(state_num));
            return 0;
        }
        else{
            return static_reward;
        }

        /*if (state_num == transtate[time_step % transtate.size()]){
            return reward_factor * static_reward;
        }
        else{
            return static_reward + ((1 - reward_factor)* get_reward(transtate[time_step % transtate.size()]) * trans[time_step % transtate.size()]) / (((transtate.size() - 1)*1.0) * get_transition(state_num));
        }*/
    }

    void set_qvalue(float qvaluee){
        qvalue = qvaluee;
    }

    int get_num_taken(){
        return num_taken;
    }


    vector<int> get_transitions(){
        return transitions;
    }

    vector<float> get_rewards(){
        return rewards;
    }

    friend ostream &operator<<( ostream &output, const QState& q);

};

ostream &operator<<( ostream &output, const QState& q){ 
         output << "Action: "<< q.action.first<<"\tQ-value: "<< q.qvalue << "\tTaken: "<< q.num_taken << "\tUpper Bound: "<< q.upper_bound << endl;
         return output;            
}

class  State{
public: 
     vector<QState> qstates ={};
    int state_num;
    int num_states;
    float value;
    //QState best_qstate;
    int best_qstate;
    bool isBestQStateSet = false;
    int num_visited;
    map<string,pair<float,float>> parameters;
    //float max_lower_bound = -INFINITY;
    float max_lower_bound;

    State(map<string,pair<float,float>> parameterss = {} , int statenum = 0, float initialvalue = 0.0, int numstates = 0){
        value = 0.0;
        num_visited = 0;
        state_num   = statenum;
        num_states  = numstates;
        value = initialvalue;
        parameters = parameterss;
        isBestQStateSet  = false;
        vector<QState> qstates = {};
        //max_lower_bound = -INFINITY;
    }

    ~State(){
        qstates.clear();
        //cout << "Deleting State " << state_num << endl;
        //std::vector<QState>().swap(qstates);
        qstates.shrink_to_fit();
        parameters.clear();
    }

    void visit(){
        num_visited++;
    }

    int get_state_num(){
        return state_num;
    }

    void set_num_states(int numstates){
        num_states = numstates;
    }

    float get_value(){
        return value;
    }

    /*QState get_best_qstate(){
        return best_qstate;
    }*/

    int get_best_qstate(){
        return best_qstate;
    }

    pair<string,int> get_optimal_action(){
        return qstates[best_qstate].get_action();
    }

    int best_action_num_taken(){
        return qstates[best_qstate].get_num_taken();
    }

    void update_value(){

        //best_qstate = qstates[0];
        best_qstate = 0;
        value = qstates[0].get_qvalue();

        for (int i=0; i< qstates.size(); i++) {
            if (qstates[i].get_qvalue() > value){
                //best_qstate = element;
                best_qstate = i;
                value = qstates[i].get_qvalue();
            }
        }

        isBestQStateSet = true;
    }

    map<string,pair<float,float>> get_parameters(){
        return parameters;
    } 

    void add_new_parameter(string name, pair<float,float> values){
        parameters[name] = values;
    }

    pair<bool,pair<float,float>> get_parameter(string param){
        for (auto& x:parameters){
            if (x.first == param)
                return make_pair(true, x.second);
        }
    }



    void add_qstate(QState &q){
        qstates.push_back(q);
        if (!isBestQStateSet){
            best_qstate = qstates.size() - 1;
        }
    }

    vector<QState> get_qstates(){
        return qstates;
    }


    QState* get_qstate(pair<string, int> &action){

        for (int i=0; i < qstates.size(); i++){
            if (qstates[i].get_action() == action)
                return &qstates[i];
        }
    }

    vector<pair<string, int>> get_legal_actions(){

        vector<pair<string, int>> actions;
        for (auto & element : qstates){
            actions.push_back(element.get_action());
        }
        return actions;
    }

    friend ostream &operator<<(ostream &output, State& s);

    void print_detailed(){
        cout << state_num << ": " << printableParameters(parameters) << ", visited: "<< num_visited << "\tMax Lower Bounds: "<< max_lower_bound <<endl;
        for (auto& qs:this->get_qstates())
            cout << qs << endl;
    }


};

ostream &operator<<(ostream &output, State& s){
        output << s.state_num << ": " << printableParameters(s.parameters) <<endl;
        return output;
}



class MDPModel{
    public:
        float discount;
        vector<State> states = {State()};
        vector<string> index_params = {};
        int current_state_num;
        int initial_state_num;
        json parameters = {};
        float update_error = 0.1;
        bool update_algorithm;
        int max_VMs;
        int min_VMs;
        
    MDPModel(json conf = json({}), bool upd_alg = true){
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
        update_algorithm  = upd_alg;
            
    };


    json _get_params(json par){
        json new_pars;
        for (auto& element : par.items()){
            new_pars[element.key()] = {};
            if (element.value().contains("values")){
                vector<pair<float,float>> values;
                for (auto& x:element.value()["values"]){
                    values.push_back(make_pair(x, x));
                }
                new_pars[element.key()]["values"] = values;
            }

            else if (element.value().contains("limits")){
                vector<pair<float,float>> values;
                vector<float> aux;
                for (auto& x:element.value()["limits"]) aux.push_back(x);
                for (int i = 1; i< aux.size(); i++ )
                    values.push_back(make_pair(aux[i-1] , aux[i]));
                new_pars[element.key()]["values"] = values;
            }
        }
        return new_pars;
    }

    void set_state(json measurements){
        current_state_num = _get_state(measurements);
    }

    void _update_states(string name, json new_parameter){
        int statenum = 0;
        vector<State> new_states ={};
        for (auto& x:new_parameter["values"]){
            for (auto& s:states){
                map<string,pair<float,float>> last;
                last = s.get_parameters();
                State new_state(last, statenum);
                new_state.add_new_parameter(name, x);
                new_states.push_back(new_state);
                statenum++;
            }
        }
        states = new_states;
    }

    int _get_state(json measurements){
        for (auto& s:states){
            bool matches = true;
            for (auto& par:s.get_parameters()){
                float min_v = par.second.first;
                float max_v = par.second.second;
                if (measurements[par.first] < min_v || measurements[par.first] > max_v){
                    matches = false;
                    break;
                }
            }
            if (matches) return s.state_num;
        }
    }

    void _set_maxima_minima(json parameters, json acts){
        if (acts.contains("add_VMs") || acts.contains("remove_VMs")){
            vector<int> vms;
            for (auto& x:parameters["number_of_VMs"]["values"]){
                vms.push_back(x[0]);
            }
            max_VMs = *max_element(vms.begin(), vms.end());
            min_VMs = *min_element(vms.begin(), vms.end());

        }
    }

    void _add_qstates(json acts, float initq){
        int num_states = states.size();
        for (auto& action:acts.items()){
            for (auto& val:action.value()){
                pair<string,int> act = make_pair(action.key(), val);
                for (auto& s:states){
                        if (_is_permissible(s, act)){
                            QState q(act, num_states, initq);
                            s.add_qstate(q);
                        }

                }
            }
        }

        for (auto& s:states){
            s.update_value();
        }
    }

    bool _is_permissible(State &s, pair<string,int> a){
        string action_type = a.first;
        int action_value = a.second;
        pair<bool, pair<float,float>> param_values;
        if (action_type == "add_VMs"){
             param_values = s.get_parameter("number_of_VMs");
             if (param_values.first)
                return (max(param_values.second.first , param_values.second.second) + action_value <= max_VMs);
        }
        else if (action_type == "remove_VMs"){
             param_values = s.get_parameter("number_of_VMs");
             if (param_values.first)
                return (min(param_values.second.first , param_values.second.second) - action_value >= min_VMs);
        }
        else return true;
    }

    pair<string,int> suggest_action(){
        return states[current_state_num].get_optimal_action();
    }

    vector<pair<string, int>> get_legal_actions(){
        return states[current_state_num].get_legal_actions();
    }
    
    void update(pair<string,int> &action, json measurements, float reward){
        states[current_state_num].visit(); //increase number of times visited by 1 for the current state

        QState* qstate = states[current_state_num].get_qstate(action); //find qstate corresponding to the chosen action
        if (qstate->num_states == -1) return;

        int new_state = _get_state(measurements); //find next state corresponding to current measurements
        
        qstate->update(new_state, reward); //increase number of times taken, create transition between action and next state and assign reward
        
        if (update_algorithm){
            _q_update(*qstate, states); //set qvalue of chosen action to new qvalue
            states[current_state_num].update_value(); //update the value of the current state, choosing the maximum of its qvalues
        }

        else{
            this->value_iteration(0.1); //execute Value Iteration
        }

        current_state_num = new_state; //move the agent to the next state
    }


    //UPDATE_FINITE: moves the agent to the new state after choosing an action without updating the model
    void update_finite(pair<string,int> action, json measurements, float reward){
        int new_state = _get_state(measurements);
        current_state_num = new_state; 
    }

    void _q_update(QState &qstate, vector<State> &V){
        float new_qvalue = 0.0;
        float r;
        float t;
        for (int i=0; i < V.size(); i++){
            t = qstate.get_transition(i);
            r = qstate.get_reward(i);
            new_qvalue += t * (r + discount * V[i].get_value());
        }
        qstate.set_qvalue(new_qvalue);
    }

    void _q_updatesss(QState &qstate, vector<State> &V, int time_step){
        float new_qvalue = 0.0;
        float r;
        float t;
        for (int i=0; i < V.size(); i++){
            t = qstate.get_transition(i);
            r = qstate.get_reward(i, time_step);
            new_qvalue += t * (r + discount * V[i].get_value());
        }
        qstate.set_qvalue(new_qvalue);
    }

 void _q_update2(QState &qstate, vector<float> &V){
        float new_qvalue = 0.0;
        float r;
        float t;
        for (int i=0; i < V.size(); i++){
            t = qstate.get_transition(i);
            r = qstate.get_reward(i);
            new_qvalue += t * (r + discount * V[i]);
        }
        qstate.set_qvalue(new_qvalue);
    }

void _q_update2xx(QState &qstate, vector<float> &V){
        float new_qvalue = 0.0;
        float r;
        float t;
        for (int i=0; i < V.size(); i++){
            t = qstate.get_transition(i);
            r = qstate.get_reward(i);
            new_qvalue += t * (r + V[i]);
        }
        qstate.set_qvalue(new_qvalue);
    }
    
void _q_update2a(QState &qstate, vector<float> &V, int time_step){
        float new_qvalue = 0.0;
        float r;
        float t;
        for (int i=0; i < V.size(); i++){
            t = qstate.get_transition(i);
            r = qstate.get_reward(i, time_step);
            new_qvalue += t * (r + V[i]);
        }
        qstate.set_qvalue(new_qvalue);
    }


    int value_iteration(float error = -1.0, bool verbose = false, bool useBounds = false ){
        if (error < 0){
            error = update_error;
        }
        bool repeat = true;
        float old_value;
        float new_value;
        int max=0;
        vector<float> V_tmp;
        //V_tmp.reserve(states.size());
        vector<float> V;
        //V.reserve(states.size());
        if (useBounds)
            update_bounds();
        while(repeat){
            repeat = false;

            for (int i=0; i < states.size(); i++){
                V_tmp.push_back(states[i].get_value());
            }

            if (verbose) {
                printDetails(); //Just to print details for every state
            }

            for (int j = 0 ; j < states.size(); j++ ){
                for (int m = 0; m < states[j].get_qstates().size(); m++){
                   _q_update2(states[j].qstates[m], V_tmp);
                }
                old_value = states[j].get_value();
                states[j].update_value();
                new_value = states[j].get_value();
                if (abs(old_value - new_value) > error)
                    repeat = true;
            }
            max=getValue1();
            V_tmp.clear();
        }
        return max;
    }
    void value_iterationM(int horizon){
        //vector<State> V_tmp;
        vector<float> V_tmp;
        for (int i = 1 ; i < horizon+1; i++ ){
            //V_tmp = states;
            //V_tmp = getStateValues1(states, V_tmp);
            getStateOnlyValues(V_tmp);
            for (int j = 0 ; j < states.size(); j++ ){
                for (int m = 0; m < states[j].get_qstates().size(); m++){
                    _q_update2(states[j].qstates[m], V_tmp);
                }
                states[j].update_value();
            }
            V_tmp.clear();
        }
    }

    void getStateOnlyValues(vector<float> &V){
        for (int i=0; i < states.size(); i++){
            V.push_back(states[i].get_value());
        }
    }

    vector<string> get_parameters(){
        return index_params;
    }
    
    float get_percent_not_taken(){
        float total = 0.0;
        float not_taken = 0.0;
        for (auto& s:states){
            for (auto& qs:s.get_qstates()){
                total = total + 1.0;
                if (qs.get_num_taken() == 0)
                    not_taken = not_taken + 1.0;
            }
        }
        return not_taken / total;
    }

    void print_model(bool detailed=false){
        for (auto& s:states){
            if (detailed){
                s.print_detailed();
                cout << endl;
            }
            else
                cout << s << endl;
        }
    }

    void update_bounds(){
        float t = 0.0;
        float curr_max = -INFINITY;
        float curr_min = INFINITY;
        bool f = false;
        for (auto& s:states){
            s.max_lower_bound = -INFINITY;
            for (auto& qs:s.qstates){//for every QState calculate upper and lower bounds of Q(s,a)
                for (int i=0; i < states.size(); i++){
                    t = qs.get_transition(i);
                    if (t != 0.0){//calculate max/min of V(s') accessible from current state s
                        curr_max = max(curr_max, states[i].get_value());
                        curr_min = min(curr_min, states[i].get_value());
                        f = true;
                    }
                }
                if (f){//if the state is not terminal, calculate max/min of rewards and add to the values calculated above to get bounds
                qs.upper_bound = *max_element(qs.rewards.begin(), qs.rewards.end()) + discount * curr_max;
                qs.lower_bound = *min_element(qs.rewards.begin(), qs.rewards.end()) + discount * curr_min;
                }
                else{//if the state is terminal, set upper bound to highest posssible and lower bound to lowest possible
                    qs.upper_bound = INFINITY;
                    qs.lower_bound = -INFINITY;
                }

                curr_max = -INFINITY;
                curr_min = INFINITY;
                f=false;
                if (qs.lower_bound > s.max_lower_bound) s.max_lower_bound = qs.lower_bound;//calculate maximum of upper bounds of a state, useful for value iteration condition
        }
    }
    }

    void printDetails(){
        for (int i=0; i<states.size(); i++){
                cout << "State " << states[i].get_state_num()<< ": " << endl;
                for (int j=0; j<states[i].get_qstates().size(); j++){
                    cout << "\tQstate " << states[i].qstates[j].action.first <<": " << endl;
                    for (int k=0; k< states[i].qstates[j].transitions.size(); k++){
                        if(states[i].qstates[j].get_transition(k) > 0){
                        cout << "\tTransition to " << k << " :" << states[i].qstates[j].get_transition(k) << ", Reward: " << states[i].qstates[j].get_reward(k) << endl;
                        }

                    }
                }
        }
    }

        vector<pair<int,float>> getStateValues1(vector<State> V,vector<pair<int,float>> values){

        for (int i=0; i < V.size(); i++){
            values.push_back(make_pair( V[i].get_best_qstate(), V[i].get_value()));
        }
        
        return values;
    }

    vector<pair<int,float>> getStateValuestest(vector<State>& V){
        vector<pair<int,float>> values;
        //values.reserve(V.size());    
        for (int i=0; i < V.size(); i++){
            values.push_back(make_pair( V[i].get_best_qstate(), V[i].get_value()));
        }
        
        return values;
    }
        vector<pair<int,float>> getStateValuestestUN(vector<State>& V,int step){
        vector<pair<int,float>> values;
        //values.reserve(V.size());    
        for (int i=0; i < V.size(); i++){
            values.push_back(make_pair( V[i].get_best_qstate(), V[i].get_value()));
            states[i].qstates[V[i].get_best_qstate()].lastused=step;
        }
        
        return values;
    }

    void loadValueFunction(vector<pair<int, float>> V){
        for (int i=0; i < V.size(); i++){
            states[i].best_qstate = V[i].first;
            states[i].value = V[i].second;
        }
    }
    void loadValueFunctiontest(vector<pair<int, float>>& V){
        for (int i=0; i < V.size(); i++){
            states[i].best_qstate = V[i].first;
            states[i].value = V[i].second;
        }
    }
    /*
    Auxiliary function, loads best QState indexes from external vector to model states.
    Takes as input a vector of indices of best QStates for every state.
    No output.
    */
    void loadBestQStates(vector<int> &V){
        for (int i=0; i < V.size(); i++){
            states[i].best_qstate = V[i];
        }
    }
        
};
