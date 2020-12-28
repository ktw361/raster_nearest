#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_resize.h"
#include "stb_image_write.h"
#include "rastertriangle.h"
#include "raster_nearest.cpp"

using namespace std;
typedef unsigned char uchar;

using std::vector;
using std::pair;

using std::swap;
using std::min;
using std::max;


//n * 3 * 2 floats in xys
void rasterTrianglesRGB(int h, int w,
                        unsigned char *img,
                        int n, float * xys, unsigned char * colors)
{
	for (int i=0;i<n;i++){
		RasterTriangle::PixelEnumerator<float> pixels(
			xys[i*6+0],
			xys[i*6+1],
			xys[i*6+2],
			xys[i*6+3],
			xys[i*6+4],
			xys[i*6+5],
		0,h-1,0,w-1);
		//h/6,h-h/6-1,w/6,w-w/6-1);
		int x,y;
		float u,v;
		while (pixels.getNext(&x,&y,&u,&v)){
			img[(x*w+y)*3+0]=colors[i*9+0*3+0]*(1-u-v)+colors[i*9+1*3+0]*u+colors[i*9+2*3+0]*v;
			img[(x*w+y)*3+1]=colors[i*9+0*3+1]*(1-u-v)+colors[i*9+1*3+1]*u+colors[i*9+2*3+1]*v;
			img[(x*w+y)*3+2]=colors[i*9+0*3+2]*(1-u-v)+colors[i*9+1*3+2]*u+colors[i*9+2*3+2]*v;
		}
	}
}



int main() {
    auto ret = RasterTriangle::allPixels<double>(
            -5, 0, 
            0, 5,
            5, 0,
            -8, 8,
            -1, 8);

    for (size_t i = 0; i != ret.size(); ++i) {
        auto p = ret[i];
        double x, y, u, v;
        x = p.first.first;
        y = p.first.second;
        u = p.second.first;
        v = p.second.second;
        
        x = (x + (5*u) + (10*v));
        y = (y + (5*u) + 0*v);

        printf("(%.2ff, %.2ff), %d %d %.2f %.2f\n", x, y,
                p.first.first, p.first.second, p.second.first, 
                p.second.second);
    }
    string fname = "/Users/zhifanzhu/Desktop/test.jpg";
    /* string fname = "/Users/zhifanzhu/Desktop/test.png"; */
    int w = 300, h = 200;
    int comp = 3;
    /* int stride = h*sizeof(uchar); */
    uchar *data = new uchar[w*h*3];
    for (int i = 0; i != w*h*3; ++i) {
        data[i] = 0;
    }

	int * triangle = (int *) malloc(sizeof(int)*h*w);	
	int * xys = (int *) malloc(sizeof(int)*h*w*2);
    float *zbuf =new float[h*w];
    for (int i = 0; i != h*w; ++i) 
        zbuf[i]=1e8;
    int *rbuf = (int*)malloc(sizeof(int)*h*w);
    int *gbuf = (int*)malloc(sizeof(int)*h*w);
    int *bbuf = (int*)malloc(sizeof(int)*h*w);

    float p2d[] = {30, 20,
                   70,180,
                   120, 30};
    float pz[] = {1, 1, 1};
    float pr[] = {80, 160, 220};
    float pg[] = {80, 160, 220};
    float pb[] = {255, 255, 255};
    int m1 = 0, m2 = 1, m3 = 2;
    cover_rgbd(p2d, pz, pr, pr, pr, m1, m2,
            m3, 0, triangle, zbuf, rbuf, gbuf, bbuf,
            h, w, xys, 1);
    for (int i = 0; i != h; ++i) {
        for (int j = 0; j != w; ++j) {
            data[i*w*3+j*3+0] = rbuf[i*w+j];
            data[i*w*3+j*3+1] = gbuf[i*w+j];
            /* data[i*w*3+j*3+1] = gbuf[i*w+j]; */
            /* data[i*w*3+j*3+2] = bbuf[i*w+j]; */
        }
    }

    stbi_write_jpg(fname.c_str(), w, h, comp, data, 100);
    delete []data;


    return 0;
}
