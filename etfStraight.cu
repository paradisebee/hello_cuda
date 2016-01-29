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

// single point calculation
__device__ double dire_weight(double *x, double *y) {
    return y[0]*x[0]+y[1]*x[1];
}

__device__ double mag_weight(double x, double y) {
    return 0.5*(1+y-x);
}

__global__ void etfStraight(
                double *xout, double *yout, double *outmag,
                double *tx, double *ty, 
                double *im, double *gmag,
                int height, int width) {

    // calculate pixels' location
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    // Only execute valie pixels
    if (x>=width || y>=height) {
        return;
    }
    // get perpendicular line centered at current point
    double ctrp[2] = {y,x};
    // vector orthogonal to [ctrx,ctry]
    double vec[2] = {tx[y*width+x],ty[y*width+x]};
    double p_line[2][2*LEN+1] = {0.0};
    line2D(p_line, ctrp, vec);
    
    /*
    int idx_line[2][2*LEN+1] = {0};
    for (int i = 0; i < 2*LEN+1; i++){
        idx_line[0][i] = (int)(p_line[0][i]+0.5);
        idx_line[1][i] = (int)(p_line[1][i]+0.5);
    }
    */

    double sum_wd = 0.0;
    double temp[2] = {0.0};
    for (int i = 0; i < 2*LEN+1; i++){
        if (p_line[0][i]>=0 && p_line[0][i]<width && 
            p_line[1][i]>=0 && p_line[1][i]<height){
            int posx = (int)(p_line[0][i]+0.5);
            int posy = (int)(p_line[1][i]+0.5);
            int ind = posy*width+posx; 
            double ctrV[2] = {tx[y*width+x],ty[y*width+x]};
            double winV[2] = {tx[ind],ty[ind]};
            double wd = dire_weight(ctrV, winV); 
            sum_wd += wd;
            double wm = mag_weight(gmag[y*width+x], gmag[ind]);
            temp[0] += wd*wm*tx[ind];
            temp[1] += wd*wm*ty[ind];
        }
    }
    if (sum_wd/9<-0.1){
        temp[0] = -temp[0];
        temp[1] = -temp[1];
    }

    double temp_mag = sqrt(temp[0]*temp[0]+temp[1]*temp[1]);
    if (temp_mag != 0){
        outmag[y*width+x] = temp_mag;
        xout[y*width+x] = temp[0]/temp_mag;
        yout[y*width+x] = temp[1]/temp_mag;
    }
    __syncthreads(); 
}
