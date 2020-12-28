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
/* #include "rastertriangle_so.cpp" */
#include "rastertriangle_nearest.cpp"

using namespace std;
typedef unsigned char uchar;

using std::vector;
using std::pair;

using std::swap;
using std::min;
using std::max;


/* //n * 3 * 2 floats in xys */
/* void rasterTrianglesRGB(int h,int w,unsigned char *img,int n,float * xys,unsigned char * colors){ */
/* 	for (int i=0;i<n;i++){ */
/* 		RasterTriangle::PixelEnumerator<float> pixels( */
/* 			xys[i*6+0], */
/* 			xys[i*6+1], */
/* 			xys[i*6+2], */
/* 			xys[i*6+3], */
/* 			xys[i*6+4], */
/* 			xys[i*6+5], */
/* 		0,h-1,0,w-1); */
/* 		//h/6,h-h/6-1,w/6,w-w/6-1); */
/* 		int x,y; */
/* 		float u,v; */
/* 		while (pixels.getNext(&x,&y,&u,&v)){ */
/* 			img[(x*w+y)*3+0]=colors[i*9+0*3+0]*(1-u-v)+colors[i*9+1*3+0]*u+colors[i*9+2*3+0]*v; */
/* 			img[(x*w+y)*3+1]=colors[i*9+0*3+1]*(1-u-v)+colors[i*9+1*3+1]*u+colors[i*9+2*3+1]*v; */
/* 			img[(x*w+y)*3+2]=colors[i*9+0*3+2]*(1-u-v)+colors[i*9+1*3+2]*u+colors[i*9+2*3+2]*v; */
/* 		} */
/* 	} */
/* } */
/* void getAllPixels_phase1(int lx,int hx,int ly,int hy,float * xy3,void ** handle,int *n){ */
/* 	vector<pair<pair<int,int>,pair<float,float> > > *all_pixels=new vector<pair<pair<int,int>,pair<float,float> > >(RasterTriangle::allPixels( */
/* 		xy3[0],xy3[1],xy3[2],xy3[3],xy3[4],xy3[5],lx,hx,ly,hy)); */
/* 	handle[0]=all_pixels; */
/* 	n[0]=all_pixels->size(); */
/* } */
/* void getAllPixels_phase2(void ** handle,int * xys,float * uvs){ */
/* 	vector<pair<pair<int,int>,pair<float,float> > > *all_pixels=(vector<pair<pair<int,int>,pair<float,float> > >*)handle[0]; */
/* 	int n=all_pixels->size(); */
/* 	for (int i=0;i<n;i++){ */
/* 		xys[i*2+0]=(*all_pixels)[i].first.first; */
/* 		xys[i*2+1]=(*all_pixels)[i].first.second; */
/* 		uvs[i*2+0]=(*all_pixels)[i].second.first; */
/* 		uvs[i*2+1]=(*all_pixels)[i].second.second; */
/* 	} */
/* 	delete all_pixels; */
/* } */


/* void getAllPixels_phase2_nouv(void ** handle,int * xys){ */
/* 	vector<pair<pair<int,int>,pair<float,float> > > *all_pixels=(vector<pair<pair<int,int>,pair<float,float> > >*)handle[0]; */
/* 	int n=all_pixels->size(); */
/* 	for (int i=0;i<n;i++){ */
/* 		xys[i*2+0]=(*all_pixels)[i].first.first; */
/* 		xys[i*2+1]=(*all_pixels)[i].first.second; */
/* 	} */
/* 	delete all_pixels; */
/* } */


/* int pixelsInTriangle(int h,int w,float * xy3,int *xys){ */
/* 	void*  handle[1]; */
/* 	int nret = 0; */
/*  	getAllPixels_phase1(0,w-1,0,h-1,xy3,handle,&nret); */	
/* 	getAllPixels_phase2_nouv(handle,xys); */
/* 	return nret; */
/* } */


/* bool equal(float* points, int i, int j){ */
/*   	if ((abs(points[i*2] - points[j*2])<1e-5) && (abs(points[i*2+1] - points[j*2+1])<1e-5)) */
/*     		return 1; */
/*   	return 0; */

/* } */
/* float dis(float x1,float y1,float x2,float y2){ */
/*   	return sqrt( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)  ); */
/* } */

/* void swap(float&x,float&y){ */
/*   	float temp = x;  x =y;   y =temp; */
/* } */

/* float inter2(float * points_2d,float * cof,int ii,int jj,int i,int j){ */
/*   	float x1 = points_2d[i*2]; */
/*   	float y1 = points_2d[i*2+1]; */
/*   	float x2 = points_2d[j*2]; */
/*   	float y2 = points_2d[j*2+1]; */

/*   	float s1 = dis(ii,jj,x1,y1); */
/*   	float s2 = dis(ii,jj,x2,y2); */
/*   	return s2/(s1+s2)*cof[i] + s1/(s1+s2)*cof[j]; */
/* } */

/* float inter(float* points_2d, */
/*             float* cof, */
/*             int ii, int jj, */
/*             int i, int j, int k) */
/* { */
/* 	if (equal(points_2d,i,j) &&  equal(points_2d,j,k) && equal(points_2d,i,k)) */
/* 		return cof[i]; */
/* 	if (equal(points_2d,i,j)) */
/* 		return inter2(points_2d,cof,ii,jj,j,k); */
/* 	if (equal(points_2d,i,k)) */
/* 		return inter2(points_2d,cof,ii,jj,j,k); */
/* 	if (equal(points_2d,j,k)) */
/* 		return inter2(points_2d,cof,ii,jj,i,j); */

/* 	float x1 = points_2d[i*2]; */
/* 	float y1 = points_2d[i*2+1]; */
/* 	float x2 = points_2d[j*2]; */
/* 	float y2 = points_2d[j*2+1]; */
/* 	float x3 = points_2d[k*2]; */
/* 	float y3 = points_2d[k*2+1]; */

/* 	// in x_line or y_line */
/* 	if ((abs(y1-y2)<1e-5) && (abs(y3-y2)<1e-5)) */
/* 		return inter2(points_2d,cof,ii,jj,i,j); */
/* 	if ((abs(x1-x2)<1e-5) && (abs(x3-x2)<1e-5)) */
/* 		return inter2(points_2d,cof,ii,jj,i,j); */

/* 	// sort */
/* 	if (points_2d[i*2]>points_2d[j*2]) */
/* 		swap(i,j); */
/* 	if (points_2d[j*2]>points_2d[k*2]) */
/* 		swap(j,k); */
/* 	if (points_2d[i*2]>points_2d[j*2]) */
/* 		swap(i,j); */

/* 	if ((ii>points_2d[j*2]) && (ii<points_2d[k*2])) */
/* 		swap(i,k); */

/* 	x1 = points_2d[i*2];y1 = points_2d[i*2+1]; */
/* 	x2 = points_2d[j*2];y2 = points_2d[j*2+1]; */
/* 	x3 = points_2d[k*2];y3 = points_2d[k*2+1]; */


/* 	float ya =  y1 + (ii-x1)/(x2-x1)*(y2-y1); */
/* 	float yb =  y1 + (ii-x1)/(x3-x1)*(y3-y1); */
/* 	float cofa = (cof[i] * (ii - x2) + cof[j] * (x1 - ii)  ) / (x1-x2); */
/* 	float cofb = (cof[i] * (ii - x3) + cof[k] * (x1 - ii) ) / (x1-x3); */
/* 	float cof_ret = (cofa*(yb-jj) + cofb*(jj-ya) ) / (yb-ya); */

/* 	if ( cof_ret <min(cof[i],min(cof[j],cof[k])) ) */
/* 		cof_ret = min(cof[i],min(cof[j],cof[k])); */
/* 	if ( cof_ret >max(cof[i],max(cof[j],cof[k])) ) */
/* 		cof_ret = max(cof[i],max(cof[j],cof[k])); */

/* 	return cof_ret; */

/* } */


/* bool isline(float* xy3){ */
/* 	float ax = xy3[0]-xy3[4]; */
/* 	float ay = xy3[1]-xy3[5]; */
/* 	float bx = xy3[2]-xy3[4]; */
/* 	float by = xy3[3]-xy3[5]; */
/* 	float cross = ax*by-bx*ay; */
/* 	if (abs(cross)<1e-5) */
/* 		return 1; */
/* 	return 0; */

/* } */


/* void cover_rgbd(float* p,  // P_2d */
/*                 float* pz, float* pr, float* pg, float* pb, */ 
/*                 int m1,int m2,int m3, int idx, int* triangle , */
/*                 float* zbuf, int* rbuf, int* gbuf, int* bbuf, */ 
/*                 int h,int w, int * xys, int debug) */
/* { */

/* 	float xy3[6]; */
/* 	xy3[0] = p[m1*2+0]; */
/* 	xy3[1] = p[m1*2+1]; */
/* 	xy3[2] = p[m2*2+0]; */
/* 	xy3[3] = p[m2*2+1]; */
/* 	xy3[4] = p[m3*2+0]; */
/* 	xy3[5] = p[m3*2+1]; */

/* 	if (debug == 1){ */
/* 		printf("%d, %d, %d\n", m1, m2, m3); */
/* 		/1* printf("%f ", xy3[0]); *1/ */
/* 		/1* printf("%f ", xy3[1]); *1/ */
/* 		/1* printf("%f ", xy3[2]); *1/ */
/* 		/1* printf("%f ", xy3[3]); *1/ */
/* 		/1* printf("%f ", xy3[4]); *1/ */
/* 		/1* printf("%f ", xy3[5]); *1/ */
/* 	} */

/* 	if (isline(xy3)) */
/* 		return; */

/* 	int num = pixelsInTriangle(h,w,xy3,xys); */
/* 	for (int i =0; i<num; i++){ */
/* 		int x = xys[i*2]; */
/* 		int y = xys[i*2+1]; */
/* 		float zz = inter(p,pz,x,y,m1,m2,m3); */
/* 		float r_f = inter(p,pr,x,y,m1,m2,m3); */
/* 		/1* float g_f = inter(p,pg,x,y,m1,m2,m3); *1/ */
/* 		/1* float b_f = inter(p,pb,x,y,m1,m2,m3); *1/ */
/* 		if (debug==1) */
/* 			/1* printf("zz = %f %f\n",zz,zbuf[y*w+x]), *1/ */
/* 			printf("xy = %d %d, ",y,x); */
/* 			printf("r_f = %f\n",r_f); */

/* 		/1* printf( *1/ */
/* 		/1* 		"zbuf.size: %d, idx: %d\n", *1/ */ 
/* 		/1* 		int(sizeof(zbuf)/sizeof(zbuf[0])), int(y*w+x) *1/ */
/* 		/1* ); *1/ */
/* 		if (zz < zbuf[y*w+x]){ */
/*             cout << y*w+x << endl; */
/* 			zbuf[y*w+x] =  zz; */
/* 			rbuf[y*w+x] = int(r_f); */
/* 			/1* gbuf[y*w+x] = int(g_f); *1/ */
/* 			/1* bbuf[y*w+x] = int(b_f); *1/ */
/* 			/1* triangle[y*w+x] = idx; *1/ */
/* 		} else { */
/*             printf("zz = %f, zbuf[y*w+x] = %f\n", zz, zbuf[y*w+x]); */
/*             cout << "Skip" << endl; */
/*         } */
/* 	} */	
/* 	printf("Finish render.\n"); */
/* } */


/* void rgbzbuffer(int h, int w, */ 
/*                 float* points_onface,      // lambda * points_onface (P_2d) = K * P_3d */
/*                 float* points_onface_ori,  // Unused */
/*                 float* points_z, */ 
/*                 float* points_r, float* points_g, float* points_b, */ 
/*                 int len_mesh, */ 
/*                 int* mesh, float* zbuf, int* rbuf, int* gbuf, int* bbuf) */
/* { */

/* 	int * triangle = (int *) malloc(sizeof(int)*h*w); */	
/* 	//float * zbuf = (float *) malloc(sizeof(float)*h*w); */

/* 	for (int i=0; i<h; i++) */
/* 		for (int j=0; j<w; j++){ */
/* 			triangle[i*w+j] = -1, zbuf[i*w+j] = 1e9, rbuf[i*w+j] = gbuf[i*w+j] = bbuf[i*w+j] = 0; */
/* 		} */
	
/* 	int * xys = (int *) malloc(sizeof(int)*h*w*2); */

/* 	for (int i=0; i<len_mesh; i++){ */
/* 		cover_rgbd( */
/* 			points_onface, points_z, points_r, points_g, points_b, mesh[i*3+0], */ 
/* 			mesh[i*3+1], mesh[i*3+2], i, triangle, zbuf, rbuf, gbuf, bbuf, */
/* 			h, w, xys, 0 */ 
/* 		); */
/* 	} */

/* 	for (int i=0; i<h; i++) */
/* 		for (int j=0; j<w; j++){ */
/* 			zbuf[i*w+j] = (zbuf[i*w+j] < 1e8) ? zbuf[i*w+j] : 0; */
/* 		} */
/* 	free(triangle); */
/* 	free(xys); */
/* } */

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