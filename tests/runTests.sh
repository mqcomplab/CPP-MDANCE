g++ -o test parseData.cpp ../tools/bts.cpp -I /usr/local/include
python3 parseData.py > pyResults.txt
./test > cppResults.txt
g++ -o parseTests parseTests.cpp
./parseTests
