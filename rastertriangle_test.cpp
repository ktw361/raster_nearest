#include <iostream>
#include <cstdio>
#include <cstdlib>

#include "rastertriangle.h"
using namespace std;

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

    return 0;
}
