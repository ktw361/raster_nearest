CC=g++
all:
	${CC} rastertriangle_so.cpp -std=c++11 -o rastertriangle_so.so -shared -fPIC -Wall  -I/usr/include

test:
	${CC} rastertriangle_test.cpp -std=c++11 -g -o rastertriangle_test 
