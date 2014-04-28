//a quick function to parse the REDD data into csv format
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
void split(string input, char delim, vector<string>& output);

int main(int argc, char* argv[]) {
    string input;

    for(int i = 1; i <= 11; ++i) {
        cout<<"Channel "<<i<<endl;
        string inPath = "low_freq/house_2/channel_"+to_string(i)+".dat";
        ifstream in(inPath.c_str());
        string outPath = "low_freq_csv_2/channel_"+to_string(i)+".csv";
        ofstream out;
        out.open(outPath.c_str(), fstream::out | fstream::app);
        out<<"timestamp,power"<<endl;
        long long t = 0;
        while(getline(in, input, '\n')) {
            vector<string> tokens;
            split(input, ' ', tokens);
            out<<t++<<","<<tokens[1]<<endl;
        }
        out.close();
        in.close();
    }
}

void split(string input, char delim, vector<string>& output) {
    stringstream sstr(input);
    string token;
    while(getline(sstr, token, delim)) {
        output.push_back(token);
    }

}