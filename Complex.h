#ifndef COMPLEX_H
#define COMPLEX_H
#include <iostream>
#include <math.h>

#ifdef _WIN32
#include <nlohmann\json.hpp>
#endif

#ifdef linux
#include <nlohmann/json.hpp>
#endif

#define PI 3.14159265

using json = nlohmann::json;

using namespace std;

extern bool flag = true;

double myRand(double LO, double HI){
    /*if (flag){
        srand(static_cast <unsigned> (time(0)));
        flag = false;
    }

    double r3 = LO + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(HI-LO)));
    return r3;*/
    return 0.5;
}

class ComplexScenario{
public:
    int time = 0;
    int training_steps;
    int load_period;
    int MIN_VMS;
    int MAX_VMS;
    json measurements;

    ComplexScenario(int trainingsteps=5000, int loadperiod=250, int initvms=10, int minvms=1, int maxvms=2){
        training_steps = trainingsteps;
        load_period = loadperiod;
        MIN_VMS = minvms;
        MAX_VMS = maxvms;
        measurements = this->_get_measurements(initvms);
    }

    json get_current_measurements(){
        double load = measurements["total_load"];
        double capacity = this->get_current_capacity();
        double served_load = min(capacity, (double)load);

        json curr_meas = measurements;
        curr_meas += json::object_t::value_type("total_load", served_load);
        return curr_meas;
    }

    double execute_action(pair<string,int> &action){
        time++;
        int num_vms = measurements["number_of_VMs"];
        string action_type = action.first;
        int action_value = action.second;

        if (action_type == "add_VMs")
            num_vms += action_value;
        if (action_type == "remove_VMs")
            num_vms -= action_value;

        if (num_vms < MIN_VMS)
            num_vms = MIN_VMS;
        if (num_vms > MAX_VMS)
            num_vms = MAX_VMS;

        measurements = _get_measurements(num_vms);
        double reward = _get_reward(action);
        return reward;

    }

    double  _get_reward(pair<string,int> action){
        int vms = measurements["number_of_VMs"];
        double load = measurements["total_load"];
        double capacity = get_current_capacity();
        double served_load = min(capacity, (double)load);

        double reward = served_load - 2.0*(double)vms;
        return reward;
    }

    double get_current_capacity(){
        int vms = measurements["number_of_VMs"];
        double read_load = measurements["%_read_load"];
        double io_per_sec = measurements["io_per_sec"];
        int ram_size = measurements["RAM_size"];
        double io_penalty;
        double ram_penalty;

        if (io_per_sec < 0.7)
            io_penalty = 0.0;
        else if (io_per_sec < 0.9)
            io_penalty = 10.0 * (io_per_sec - 0.7);
        else
            io_penalty = 2.0;

        if (ram_size == 1024)
            ram_penalty = 0.3;
        else
            ram_penalty = 0.0;
            
        double capacity = (read_load * 10.0 - io_penalty - ram_penalty) * (double)vms;

        return capacity;
    }

    json _get_measurements(int num_vms){

        json measurements = {
            {"number_of_VMs", num_vms},
            {"RAM_size", this->_get_ram_size()},
            {"number_of_CPUs", this->_get_num_cpus()},
            {"storage_capacity", this->_get_storage_capacity()},
            {"perc_free_RAM", this->_get_free_ram()},
            {"perc_CPU_usage", this->_get_cpu_usage()},
            {"io_per_sec", this->_get_io_per_sec()},
            {"total_load", this->_get_load()},
            {"%_read_load", this->_get_read_load()},
            {"total_latency", this->_get_latency()}
        };

        return measurements;

        }

    double get_incoming_load(){
        return measurements["total_load"];
    }

    double _get_load(){
        if (time <= training_steps)
            return 50.0 + 50.0 * sin(2.0 * PI * time / load_period);
        else
            return 50.0 + 50.0 * sin(2.0 * PI * time * 2 / load_period);
    }

    double _get_read_load(){
        return 0.75 + 0.25 * sin(2.0 * PI * time / 340);
    }

    double _get_io_per_sec(){
        return 0.6 + 0.4 * sin(2.0 * PI * time / 195);
    }

    int  _get_ram_size(){
        if(time % 440 < 220)
            return 1024;
        else
            return 2048;
    }

    double _get_latency(){
        return 0.5 + 0.5 * myRand(0, 1);
    }

    double _get_free_ram(){
        return 0.4 + 0.4 * myRand(0, 1);
    }

    double _get_cpu_usage(){
        return 0.6 + 0.3 * myRand(0, 1);
    }

    int _get_storage_capacity(){
        double x = myRand(0,1);
        if (x < 0.33333)
            return 10;
        else if (x < 0.66666)
            return 20;
        else
            return 40;
    }

    int _get_num_cpus(){
        double x = myRand(0,1);
        if (x < 0.5)
            return 2;
        else
            return 4;
    }

};
#endif 