#include <string>
#include <cstdlib>
#include <armadillo>
#include <fstream>
#include <iostream>
#include <map>
#include <stdlib.h>
#include <vector>
#include <sstream>
#include <queue>

using namespace std;
using namespace arma;


const int NUM_CHANNELS = 9;
const double FLOOR_MOD = 5;
const double MIN_OFFSET_MOD = 1.6;
const int MIN_T = 5;
const int TEST_TIME = 48*60*60;
const int KNN = 9;
const double SLOPE_THRESH = 50;
const double EVT_THRESH = 1;
const double SUB_P_THRESH = 0.15;
const double SUB_P_THRESH_AGG = 0.35;
const double AGG_EVT_THRESH = 40;
const double PERCEPTRON_ALPHA = 0.9;

struct subEvent {
    int i;
    int j;
    double pDiff;
    int start;
    int end;
};

struct event {
    int start;
    int end;
    double avgP;
    vector<pair<int, double>> vals;
    vector<pair<int, double>> fallingPOI;
    vector<pair<int, double>> risingPOI;
};

//simple average power/period signature
struct signature {
    double period1;
    double avgP1;
    double period2;
    double avgP2;
    double period3;
    double avgP3;
    double frequency;
    int start;
    int end;
    int deviceID;
};

struct min_record {
    double min;
    int i;
};


//compare by length of POI. Consider also scoring on power difference betweeen POIs
class min_comp {
    bool reverse;
public:
    min_comp(const bool& rev = true) {
        reverse = rev;
    }
    bool operator() (const subEvent& lhs, const subEvent& rhs) const {
        double t_metric = (double)(lhs.end-lhs.start)/(double)(rhs.end-rhs.start);
        double p_metric = (lhs.pDiff+1)/(rhs.pDiff+1);
        if(reverse) return t_metric-p_metric < 0;
        return t_metric-p_metric > 0;
    }
    bool operator() (const min_record& lhs, const min_record& rhs) const {
        if(reverse) return lhs.min < rhs.min;
        return lhs.min > rhs.min;
    }
};


void split(string input, char delim, vector<string>& output);
void addEventData(vector<double>& vals, event& evt);
void findPOI(event& evt);
double reduceFloor(vector<double>& vals);
void fillStates(vector<double>& vals, vector<bool>& states);
void getSignature(vector<event>& event_list, vector<signature>& sig, double minP, int id);
double reducePOILists(event& evt, const subEvent& sub);
void reducePeriod(vector<pair<int,double>>& vals, int start, int end, double p, double mod);

//Classifying methods
int getKNN(vector<rowvec>* tests, rowvec& comp, int n, int k);
void testPerceptron(vector<rowvec>* test, rowvec** w, string title);
void trainPerceptronSet(vector<rowvec>* a, rowvec** w);
void trainPerceptron(vector<rowvec>& a, vector<rowvec>& b, rowvec& w);


int main(int argc, char* argv[]) {
    string input;
    vector<double> vals[NUM_CHANNELS];
    double min[NUM_CHANNELS+1];

    map<int, string> label;
    map<string, int> reverseLabel;
    string labelPath = "low_freq_csv_2/labels.dat";
    ifstream in(labelPath.c_str());
    while(getline(in, input, '\n')) {
        vector<string> tokens;
        split(input, ' ', tokens);
        label[atoi(tokens[0].c_str())] = tokens[1];
        reverseLabel[tokens[1]] = atoi(tokens[0].c_str());
    }

    for(int i = 3; i <= NUM_CHANNELS+2; ++i) {
        cout<<"Channel "<<i<<endl;
        string inPath = "low_freq_csv_2/channel_"+to_string(i)+".csv";
        ifstream in(inPath.c_str());
        while(getline(in, input, '\n')) {
            vector<string> tokens;
            split(input, ',', tokens);
            vals[i-3].push_back(atof(tokens[1].c_str()));
        }
        min[i-3] = reduceFloor(vals[i-3]);
        in.close();
    }

    vector<pair<int,double>> agg_vals;
    cout<<"Aggregate"<<endl;
    string inPath = "low_freq_csv_2/channel_agg.csv";
    ifstream aggIn(inPath.c_str());
    while(getline(aggIn, input, '\n')) {
        vector<string> tokens;
        split(input, ',', tokens);
        agg_vals.push_back(make_pair(atoi(tokens[0].c_str()), atof(tokens[1].c_str())));
    }
    min[NUM_CHANNELS] = reduceFloor(vals[NUM_CHANNELS]);
    aggIn.close();

    //--------EVENT DETECTION--------//
    vector<event> event_list[NUM_CHANNELS+1];
    vector<signature> sigs[NUM_CHANNELS+1];
    
    for(int i = 0; i < NUM_CHANNELS; ++i) {
        long long total = 0;
        int evts = 0;
        bool evt = false;
        int start = 0;
        int t = 10000000;
        event* n_evt = NULL;
        for(auto b = vals[i].begin(), e = vals[i].end(); b != e; b++) {
            if(*b > min[i]*MIN_OFFSET_MOD && !evt) {
                evt = true;
                start = total;
                n_evt = new event;
            }
            if(*b < min[i]*MIN_OFFSET_MOD && evt ) {
                n_evt->start = start-1;
                n_evt->end = total-1;
                if(n_evt->end-n_evt->start > MIN_T) {
                    addEventData(vals[i], *n_evt);
                    findPOI(*n_evt);
                    event_list[i].push_back(*n_evt);
                }
                delete n_evt;

                evts++;
                evt = false;
                if(total - start < t && total - start > MIN_T) {
                    t = total - start;
                }
            }
            total++;
        }
        cout<<"Channel "<<i+3<<endl;
        cout<<"Total events: "<<event_list[i].size()<<endl;
        if(event_list[i].size() > 0) {
            getSignature(event_list[i], sigs[i], min[i], i+3);
        }
    }

    //separeate aggregate events
    vector<signature> started_events;
    vector<signature> agg_events;
    auto curr_event = agg_vals.begin();
    for(auto b = agg_vals.begin(), e = agg_vals.end(); b != e; b++) {
        //do not include final point in event average
        auto next = b + 1;
        if(next != e) {
            double s = 100*(next->second-b->second)/b->second;
            if(s > SLOPE_THRESH && abs(next->second - b->second) > AGG_EVT_THRESH) {
                signature evt;
                evt.avgP1 = next->second-b->second;
                evt.period1 = 0;
                evt.avgP2 = 0;
                evt.period2 = 0;
                evt.avgP3 = 0;
                evt.period3 = 0;
                evt.frequency = 0;
                evt.start = next->first;

                curr_event = b;
                started_events.push_back(evt);
            }
            else if(s < -SLOPE_THRESH && abs(next->second - b->second) > AGG_EVT_THRESH) {
                for(int i = 0, len = started_events.size(); i < len; ++i) {
                    if(abs((started_events[i].avgP1-b->second)/started_events[i].avgP1) < SUB_P_THRESH_AGG) {
                        started_events[i].avgP1 += b->second-next->second;
                        started_events[i].avgP1 /= 2;
                        started_events[i].end = next->first;
                        started_events[i].period1 = started_events[i].end - started_events[i].start;

                        agg_events.push_back(started_events[i]);
                        started_events.erase(started_events.begin() + i);
                        reducePeriod(agg_vals, started_events[i].start, started_events[i].end, started_events[i].avgP1, 0.9);
                        b = curr_event;
                        break;
                    }
                }
            }
        }
    }
    cout<<"Agg events: "<<agg_events.size()<<endl;

    //write events to file. Need average power (up to 3 subevents pser), period,
    //and frequency if some choosen threshold is acheived to warrant.
    vector<rowvec> train[NUM_CHANNELS];
    string outPath = "low_freq_csv_2/event_classes.csv";
    ofstream eout(outPath.c_str());
    eout<<"averagePower1,period1,averagePower2,period2,averagePower3,period3,frequency,start,end,device"<<endl;
    for(int i = 0; i < NUM_CHANNELS; ++i) {
        for(int j = 0, len = sigs[i].size(); j < len; ++j) {
            eout<<sigs[i][j].avgP1<<","<<sigs[i][j].period1<<","<<sigs[i][j].avgP2<<","<<sigs[i][j].period2<<","<<sigs[i][j].avgP3<<","<<sigs[i][j].period3<<","<<sigs[i][j].frequency<<","<<sigs[i][j].start<<","<<sigs[i][j].end<<","<<label[sigs[i][j].deviceID]<<endl;
            rowvec n_row(10);
            n_row<<1<<sigs[i][j].avgP1<<sigs[i][j].period1<<sigs[i][j].avgP2<<sigs[i][j].period2<<sigs[i][j].avgP3<<sigs[i][j].period3<<sigs[i][j].frequency<<0<<0<<endr;
            train[i].push_back(n_row);
        }
    }
    eout.close();

    vector<rowvec> test;
    for(int j = 0, len = agg_events.size(); j < len; ++j) {
        rowvec n_row(10);
        n_row<<1<<agg_events[j].avgP1<<agg_events[j].period1<<agg_events[j].avgP2<<agg_events[j].period2<<agg_events[j].avgP3<<agg_events[j].period3<<agg_events[j].frequency<<agg_events[j].start<<agg_events[j].end<<endr;
        test.push_back(n_row);
    }
    
    rowvec** pWeights = new rowvec*[NUM_CHANNELS-1];
    for(int i = 0; i < NUM_CHANNELS-1; ++i) {
        pWeights[i] = new rowvec[NUM_CHANNELS];
    }

    for(int i = 0; i < NUM_CHANNELS-1; ++i) {
        for(int j = i+1; j < NUM_CHANNELS; ++j) {
            pWeights[i][j].resize(train[0][0].n_elem);
        }
    }

    outPath = "low_freq_csv_2/agg_events.csv";
    ofstream aggOut(outPath.c_str());
    aggOut<<"averagePower1,period1,averagePower2,period2,averagePower3,period3,frequency,start,end,device"<<endl;
    for(int i = 0, len = test.size(); i < len; ++i) {
        int n = getKNN(train, test[i], KNN, 2);
        agg_events[i].deviceID = n;
        aggOut<<test[i][1]<<","<<test[i][2]<<","<<test[i][3]<<","<<test[i][4]<<","<<test[i][5]<<","<<test[i][6]<<","<<test[i][7]<<","<<test[i][8]<<","<<test[i][9]<<","<<label[n+3]<<endl;
    }
    aggOut.close();

    //Does not work, times out
    /*cout<<"Training perceptron"<<endl;
    trainPerceptronSet(train, pWeights);
    cout<<"Done!"<<endl;*/

    bool on[TEST_TIME][NUM_CHANNELS];
    bool trueOn[TEST_TIME][NUM_CHANNELS];
    string devicesInPath = "low_freq_csv_2/device_states.csv";
    ifstream devicesIn(devicesInPath.c_str());
    //get rid of header row
    getline(devicesIn, input, '\n');

    while(getline(devicesIn, input, '\n')) {
        vector<string> tokens;
        split(input, ',', tokens);
        if(atoi(tokens[0].c_str()) >= TEST_TIME) break;
        for(int i = 1, len = tokens.size(); i < len; ++i) {
            if(tokens[i] == "0") {
                trueOn[atoi(tokens[0].c_str())][i-1] = false;
            }
            else {
                trueOn[atoi(tokens[0].c_str())][i-1] = true;
            }
        }
    }

    for(int i = 0; i < TEST_TIME; ++i) {
        for(int j = 0; j < NUM_CHANNELS; ++j) {
            on[i][j] = false;
        }
    }

    for(auto b = agg_events.begin(), e = agg_events.end(); b != e; ++b) {
        for(int i = b->start; i < b->end; ++i) {
            if(i < TEST_TIME) on[i][b->deviceID] = true;
        }
    }

    string devicePath = "low_freq_csv_2/agg_devices.csv";
    ofstream deviceOut(devicePath.c_str());
    deviceOut<<"time,kitchen_outlets1,lighting,stove,microwave,washer_dryer,kitchen_outlets2,refrigerator,dishwasher,disposal"<<endl;
    for(int i = 0; i < TEST_TIME; ++i) {
        deviceOut<<i<<",";
        for(int j = 0; j < NUM_CHANNELS; ++j) {
            deviceOut<<on[i][j];
            if(j != NUM_CHANNELS-1) deviceOut<<",";
        }
        deviceOut<<endl;
    }
    deviceOut.close();

    int totalEvts = 0;
    int totalDevices[NUM_CHANNELS];
    int missedEvts = 0;
    int missedDevices[NUM_CHANNELS];
    int falsePositive = 0;

    for(int i = 0; i < NUM_CHANNELS; ++i) {
        missedDevices[i] = 0;
        totalDevices[i] = 0;
    }

    for(int i = 0; i < TEST_TIME; ++i) {
        for(int j = 0; j < NUM_CHANNELS; ++j) {
            if(trueOn[i][j]) {
                totalEvts++;
                totalDevices[j]++;
                if(!on[i][j]) {
                    missedEvts++;
                    missedDevices[j]++;
                }
            }
            else if(on[i][j]) {
                falsePositive++;
            }
        }
    }
    cout<<"Classifying: "<<missedEvts<<"/"<<totalEvts<<" -> "<<1-(double)missedEvts/(double)totalEvts<<endl;
    cout<<"False Positive: "<<falsePositive<<"/"<<totalEvts<<" -> "<<1-(double)falsePositive/(double)totalEvts<<endl;
    for(int i = 0; i < NUM_CHANNELS; ++i) {
        cout<<"\t"<<label[i+3]<<": "<<missedDevices[i]<<"/"<<totalDevices[i]<<" -> "<<1-(double)missedDevices[i]/(double)totalDevices[i]<<endl;
    }

    return 0;
}

void split(string input, char delim, vector<string>& output) {
    stringstream sstr(input);
    string token;
    while(getline(sstr, token, delim)) {
        output.push_back(token);
    }
}

double reduceFloor(vector<double>& vals) {
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

    return min;
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

void addEventData(vector<double>& vals, event& evt){
    int s = 0;
    if(evt.start > 0) s = evt.start-1;
    int e = evt.end+1;
    for(int i = s; i < e; ++i) {
        evt.vals.push_back(make_pair(i-1, vals[i]));
    }
}

//determine subevents
void findPOI(event& evt) {
    double avg = 0;
    int evtLen = evt.vals.size();
    for(auto b = evt.vals.begin(), e = evt.vals.end(); b != e; b++) {
        //do not include final point in event average
        if(b != (e-1)) {
            avg += b->second/evtLen;
        }
        auto next = b + 1;
        if(next != e) {
            double s = 100*(next->second-b->second)/b->second;
            if((s > SLOPE_THRESH || s < -SLOPE_THRESH) && abs(next->second - b->second) > EVT_THRESH) {
                if(s > 0) evt.risingPOI.push_back(make_pair(next->first, next->second));
                else evt.fallingPOI.push_back(make_pair(next->first, next->second));
            }
        }
        else {
            evt.fallingPOI.push_back(make_pair(b->first, b->second));
        }
    }
    evt.avgP = avg;
}


void getSignature(vector<event>& event_list, vector<signature>& sigs, double minP, int id) {
    double mean = 0;
    double mean_period = 0;
    int num_evt = event_list.size();
    for(auto b = event_list.begin(), e = event_list.end(); b != e; ++b) {
        if(b->avgP > 0) mean += b->avgP;
        mean_period += (double)(b->end-b->start);
    }
    mean /= num_evt;
    mean_period /= num_evt;

    double var = 0;
    double var_period = 0;
    for(auto b = event_list.begin(), e = event_list.end(); b != e; ++b) {
        if(b->avgP > 0) var += (b->avgP-mean)*(b->avgP-mean);
        var_period += (b->end-b->start-mean_period)*(b->end-b->start-mean_period);   
    }
    var /= num_evt;
    var_period /= num_evt;

    //parse subevents
    for(int k = 0, len = event_list.size(); k < len; ++k) {
        signature sig;
        sig.period1 = 0;
        sig.avgP1 = 0;
        sig.period2 = 0;
        sig.avgP2 = 0;
        sig.period3 = 0;
        sig.avgP3 = 0;
        sig.frequency = 0;
        sig.deviceID = id;
        //separate three most prevalent subevents
        for(int n = 0; n < 3; ++n) {
            priority_queue<subEvent, vector<subEvent>, min_comp> minTest;
            for(int i = 0, len = event_list[k].risingPOI.size(); i < len; ++i) {
                for(int j = 0, len = event_list[k].fallingPOI.size(); j < len; ++j) {
                    if(event_list[k].risingPOI[i].first < event_list[k].fallingPOI[j].first) {
                        //possible rising/falling pair
                        //check if power ratings are similar enough
                        if(abs((event_list[k].risingPOI[i].second-event_list[k].fallingPOI[j].second))/event_list[k].risingPOI[i].second < SUB_P_THRESH || (n==0 && (event_list[k].risingPOI.size()==1 && event_list[k].risingPOI.size()==1))) {
                            subEvent p;
                            p.start = event_list[k].risingPOI[i].first;
                            p.end = event_list[k].fallingPOI[j].first;
                            p.pDiff = abs(event_list[k].risingPOI[i].second-event_list[k].fallingPOI[j].second)/event_list[k].risingPOI[i].second;
                            p.i = i;
                            p.j = j;
                            minTest.push(p);
                        }
                    }
                }
            }
            //selet top as major event, and reduce the remaining values and iterate (max 3 times).
            double avgP = 0;
            if(minTest.size() > 0) {
                sig.start = event_list[k].risingPOI[minTest.top().i].first;
                sig.end = event_list[k].fallingPOI[minTest.top().j].first;
                avgP = reducePOILists(event_list[k], minTest.top());
            }
            else {
                break;
            }
            
            if(n == 0) {
                sig.avgP1 = avgP;
                sig.period1 = minTest.top().end - minTest.top().start;    
            }
            else if(n == 1) {
                sig.avgP2 = avgP;
                sig.period2 = minTest.top().end - minTest.top().start;
            }
            else if(n == 2) {
                sig.avgP3 = avgP;
                sig.period3 = minTest.top().end - minTest.top().start;
            }

            if(event_list[k].risingPOI.size() == 1 || event_list[k].fallingPOI.size() == 1) {
                break; //no more matches to be found.
            }
        }
        if(sig.avgP1 > 0) sigs.push_back(sig);
    }
    cout<<"Total Signatures "<<sigs.size()<<endl;
    cout<<sigs[0].avgP1<<endl;
}

double reducePOILists(event& evt, const subEvent& sub) {
    double powerOffset= 0;
    if(evt.risingPOI[sub.i].second > evt.fallingPOI[sub.j].second) {
        powerOffset = evt.risingPOI[sub.i].second;
    }
    else {
        powerOffset = evt.fallingPOI[sub.j].second;
    }

    //remove used POIs from list
    evt.risingPOI.erase(evt.risingPOI.begin() + sub.i);
    evt.fallingPOI.erase(evt.fallingPOI.begin() + sub.j);

    //reduce remaining POIs
    for(int i = evt.risingPOI.size()-1; i >= 0; --i) {
        evt.risingPOI[i].second -= powerOffset;

        //consider thresholding this instead of a flat 0 value
        if(evt.risingPOI[i].second <= 0) {
            evt.risingPOI.erase(evt.risingPOI.begin() + i);
        }
    }
    
    for(int i = evt.fallingPOI.size()-1; i >= 0; --i) {
        evt.fallingPOI[i].second -= powerOffset;

        //consider thresholding this instead of a flat 0 value
        if(evt.fallingPOI[i].second <= 0) {
            evt.fallingPOI.erase(evt.fallingPOI.begin() + i);
        }
    }

    return powerOffset;
}

void reducePeriod(vector<pair<int,double>>& vals, int start, int end, double p, double mod) {
    for(int i = 0, len = vals.size(); i < len; ++i) {
        if(vals[i].first >= start && vals[i].first <= end) {
            vals[i].second -= p*mod;
            if(vals[i].second < 0) vals[i].second = 0;
        }
    }
}

int getKNN(vector<rowvec>* tests, rowvec& comp, int n, int k) {
    priority_queue<min_record, vector<min_record>, min_comp> minTest;

    int minList[NUM_CHANNELS];
    for(int i = 0; i < NUM_CHANNELS; ++i) {
        minList[i] = 0;
    }

    for(int i = 0; i < NUM_CHANNELS; ++i) {
        for(int j = 0, len = tests[i].size(); j < len; ++j) {
            double min = norm(tests[i][j] - comp, k);
            min_record nMin;
            nMin.min = min;
            nMin.i = i;
            if(minTest.size() < (unsigned)n) {
                minTest.push(nMin);
                minList[i]++;
            }
            else if(minTest.top().min > min){
                minList[minTest.top().i]--;
                minTest.pop();
                minTest.push(nMin);
                minList[i]++;
            }
        }
    }
    int max = minList[0];
    int max_i = 0;
    for(int i = 0; i < NUM_CHANNELS; ++i) {
        if(minList[i] > max) {
            max = minList[i];
            max_i = i;
        }
    }
    return max_i;
}

void testPerceptron(vector<rowvec>* test, rowvec** w, string title) {
    //for each instance, vote across all combinations of perceptron
    for(int i = 0; i < NUM_CHANNELS; ++i) {
        for(int j = 0, len = test[i].size(); j < len; ++j) {
            //vote for perceptron
            int votes[NUM_CHANNELS];
            for(int c = 0; c < NUM_CHANNELS; ++c) {
                votes[c] = 0;
            }
            int max = 0;
            for(int k = 0; k < NUM_CHANNELS-1; ++k) {
                for(int v = k+1; v < NUM_CHANNELS; ++v) {
                    double t = dot(test[i][j], w[k][v]);
                    if(t >= 0) votes[k]++;
                    else votes[v]++;

                    if(votes[k] > votes[max]) {
                        max = k;
                    }
                    else if(votes[v] > votes[max]) {
                        max = v;
                    }
                }
            }
            //i is the true class, max is the best guess
        }
    }
}

void trainPerceptronSet(vector<rowvec>* a, rowvec** w) {
    for(int i = 0; i < NUM_CHANNELS-1; ++i) {
        for(int j = i+1; j < NUM_CHANNELS; ++j) {
            trainPerceptron(a[i], a[j], w[i][j]);
        }
    }
}

void trainPerceptron(vector<rowvec>& a, vector<rowvec>& b, rowvec& w) {
    w.fill(0); //start off with zero vector

    //classify a >= 0, b < 0
    bool good = false;
    long long maxIter = a.size()*b.size()*10000;
    long long count = 0;
    while(!good && count < maxIter) {
        good = true;
        count++;
        for(auto i = a.begin(), e = a.end(); i != e; i++) {
            double r = dot(*i, w);
            if(r < 0) {
                w += *i * PERCEPTRON_ALPHA;
                good = false;
            }
        }
        for(auto i = b.begin(), e = b.end(); i != e; i++) {
            double r = dot(*i, w);
            if(r >= 0) {
                w -= *i * PERCEPTRON_ALPHA;
                good = false;
            }
        }
    }
    if(count >= maxIter) {
        w.fill(0);
        cout<<"Zeroes set"<<endl;
    }
    else {
        cout<<w<<endl;
    }
}
