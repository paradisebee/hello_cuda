#include <math.h>

const int LEN = 4;

// get a line centered at *ctrp, along the direction of *vec
__device__ void line2D(double out[][2*LEN+1], double *ctrp, double *vec) {
    double x = ctrp[1];
    double y = ctrp[0];
    double gx = vec[0];
    double gy = vec[1];
    for (int i = -LEN; i <= LEN; i++){
        out[0][i+LEN] = x+i*gx;
        out[1][i+LEN] = y+i*gy;
    }
}

__global__ void etfWeight(
                double *xout, double *yout, double *outmag,
                double *tx, double *ty,
                double *im, double *gmag,
                int height, int width) {

    // calculate pixels' location
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    // Only execute valid pixels
    if (x>=width || y>=height) {
        return;
    }
    // get parallel line centered at current point
    double ctrp[2] = {y,x};
    // direction vector
    double vec[2] = {tx[y*width+x], ty[y*width+x]};
    double p_line[2][2*LEN+1] = {0.0};
    line2D(p_line, ctrp, vec);

    double sum_inten = 0.0;
    for(int i = 0; i < 2*LEN+1; i++) {
        if (p_line[0][i]>=0 && p_line[0][i]<width &&
            p_line[1][i]>=0 && p_line[1][i]<height){
            int posx = (int)(p_line[0][i]+0.5);
            int posy = (int)(p_line[1][i]+0.5);
            int ind = posy*width+posx;            
            if (ind<width*height){
                sum_inten += im[ind];
            }
        }
    }
    
    __syncthreads();

    outmag[y*width+x] = sum_inten/(2*LEN+1);
    xout[y*width+x] = tx[y*width+x];
    yout[y*width+x] = ty[y*width+x];
}
