#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

using std::ifstream, std::vector, std::string, std::max, std::printf, std::cout;

void sep() {
    printf("-------------------------------------------\n");
}

int main() {
    ifstream config("tests.txt");
    int groupNum, dataNum;
    config >> groupNum >> dataNum;
    string data[dataNum];
    for (int i=0; i<dataNum; ++i){
        config >> data[i];
    }
    string groupNames[groupNum];
    int testNums[groupNum];
    int maxTestSize=0;
    
    for (int i=0; i<groupNum; ++i) {
        config >> groupNames[i] >> testNums[i];
        maxTestSize = max(maxTestSize,testNums[i]);
    }

    string testNames[groupNum][maxTestSize];
    getline(config,testNames[0][0]);
    for (int i=0; i<groupNum; ++i){
        for (int j=0; j<testNums[i]; ++j) {
            getline(config,testNames[i][j]);
        }
    }

    bool tests[groupNum][maxTestSize][dataNum];

    ifstream pyData("results/pyResults.txt");
    ifstream cppData("results/cppResults.txt");
    string pyLine;
    string cppLine;

    for(int d=0; d<dataNum; ++d){
        for(int i=0; i<groupNum; ++i) {
            for(int j=0; j<testNums[i]; ++j){
                getline(pyData,pyLine);
                getline(cppData,cppLine);
                if(pyLine.compare(cppLine) == 0)
                    tests[i][j][d] = true;
                else
                    tests[i][j][d] = false;
            }
        }
    }

    for (int i=0; i<groupNum; ++i){
        cout << "\n\033[01m" << groupNames[i] << "\033[00m\n";
        sep();
        for(int j=0; j<testNums[i]; ++j){
            int cnt=0;
            string fails = "\033[31m";
            for(int d=0; d<dataNum; ++d){
                if(tests[i][j][d])
                    ++cnt;
                else
                    fails += " " + data[d];
            }
            if(cnt == dataNum)
                cout << "\033[92m" << testNames[i][j] << " [" << cnt << "/" << dataNum << "]\033[00m\n";
            else
                cout << "\033[93m" << testNames[i][j] << " [" << cnt << "/" << dataNum << "]\033[31m" << fails << "\033[00m\n";
        }
    }
}