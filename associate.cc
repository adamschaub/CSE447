#include <string>
#include <cstdlib>
#include <armadillo>
#include <fstream>
#include <iostream>
#include <map>
#include <stdlib.h>
#include <vector>
#include <sstream>

using namespace std;
using namespace arma;

const double FLOOR_MOD = 5;

void split(string input, char delim, vector<string>& output);
void reduceFloor(vector<double>& vals);
void fillStates(vector<double>& vals, vector<bool>& states);


int main(int argc, char* argv[]) {
    //string circuit = argv[1];
    string input;
    vector<double>* vals = new vector<double> [9];
    vector<bool>* states = new vector<bool> [9];

    for(int i = 3; i <= 11; ++i) {
        cout<<"Channel "<<i<<endl;
        string inPath = "low_freq_csv_2/channel_"+to_string(i)+".csv";
        ifstream in(inPath.c_str());
        while(getline(in, input, '\n')) {
            vector<string> tokens;
            split(input, ',', tokens);
            vals[i-3].push_back(atof(tokens[1].c_str()));
        }
        reduceFloor(vals[i-3]);
        fillStates(vals[i-3], states[i-3]);
        in.close();
    }
    string outPath = "low_freq_csv_2/device_states.csv";
    ofstream out;
    out.open(outPath.c_str(), fstream::out);

    out<<"time,kitchen_outlets1,lighting,stove,microwave,washer_dryer,kitchen_outlets2,refrigerator,dishwasher,disposal"<<endl;
    for(int i = 1, len = states[0].size(); i < len; ++i) {
        out<<i-1<<",";
        for(int j = 0; j < 9; ++j) {
            out<<states[j][i];
            if(j != 8) out<<",";
        }
        out<<endl;
    }

    out.close();
    delete[] vals;
    delete[] states;

    return 0;
}

void split(string input, char delim, vector<string>& output) {
    stringstream sstr(input);
    string token;
    while(getline(sstr, token, delim)) {
        output.push_back(token);
    }

}

void reduceFloor(vector<double>& vals) {
    double min = 9999;
    for(auto b = vals.begin(), e = vals.end(); b != e; ++b) {
        if(*b < min && *b != 0) {
            min = *b;
        }
    }

    if(min < 10) min = 10;

    for(auto b = vals.begin(), e = vals.end(); b != e; ++b) {
        if(*b <= min) {
            *b = 0;
        }
    }
}

void fillStates(vector<double>& vals, vector<bool>& states) {
    //determine yes/no states
    for(auto b = vals.begin(), e = vals.end(); b != e; ++b) {
        if(*b > 0) {
            states.push_back(true);
        }
        else {
            states.push_back(false);
        }
    }
}