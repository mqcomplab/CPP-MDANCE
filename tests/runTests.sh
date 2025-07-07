g++ -O3 -o test parseData.cpp ../src/tools/bts.cpp ../src/tools/esim.cpp -I /usr/local/include
python3 parseData.py > results/pyResults.txt 2> results/pyTime.txt
./test > results/cppResults.txt 2> results/cppTime.txt
g++ -o parseTests parseTests.cpp
./parseTests
rm test
rm parseTests
