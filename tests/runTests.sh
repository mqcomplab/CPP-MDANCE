echo "Coppying test executable from build directory..."
cp ../build/tests/mdance_tests .
echo "Calculating Python results..."
python3 parseData.py > results/pyResults.txt 2> results/pyTime.txt
echo "Calculating C++ results..."
./mdance_tests > results/cppResults.txt 2> results/cppTime.txt
cp ../build/tests/parser .
echo "Comparing results..."
./parser
rm mdance_tests
rm parser
