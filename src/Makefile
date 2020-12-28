CC=g++
all:
	${CC} raster_nearest.cpp -std=c++11 -o raster_nearest.so -shared -fPIC -Wall  -I/usr/include

debug:
	${CC} raster_nearest.cpp -std=c++11 -g3 -ggdb -o raster_nearest.so -shared -fPIC -Wall  -I/usr/include

clean:
	rm raster_nearest.so
