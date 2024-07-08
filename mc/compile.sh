python setup.py build_ext --inplace

g++ -O3 -Wall -shared -mavx -mfma -std=c++11 -fPIC $(python3 -m pybind11 --includes) solver.cpp -o solver$(python3-config --extension-suffix)