#include <math.h>

__device__ size_t calculateGlobalIndex() {
    // Which block are we?
    size_t const globalBlockIndex = blockIdx.x + blockIdx.y * gridDim.x;
    // Which thread are we within the block?
    size_t const localThreadIdx = threadIdx.x + blockDim.x * blockIdx.y;
    // How big is each block?
    size_t const threadsPerBlock = blockDim.x * blockDim.y;
    // Which thread are we overall?
    return localThreadIdx + globalBlockIndex*threadsPerBlock;
}

__device__ void dire_weight(double *wd, double *x, double *y, size_t len) {
    double tx = x[0];
    double ty = x[1];
    for (int i = 0; i < len; i++) {
        wd[i] = y[i]*tx+y[9+i]*ty;
    }
}

__device__ void mag_weight(double *wm, double x, double *y, size_t len) {
    for (int i = 0; i < len; i++) {
        wm[i] = 0.5*(1+y[i]-x);
    }
}


/*__device__ double* local_win(size_t pos, size_t ih, size_t iw,  double *im) {
    double lwin[9];
    int count = 0;
    int locator = pos/iw;
    // 1,0,0,
    // 0,0,0,
    // 0,0,0.
    if( (pos-iw-1)>=0 && ((pos-iw-1)/iw)>=0 ) {
        lwin[count] = im[pos-iw-1];
        count++;
    }
    // 0,0,0,
    // 1,0,0,
    // 0,0,0.
    if( (pos-1)>=0 && ((pos-1)/iw)>=0 ) {
        lwin[count] = im[pos-1];
        count++;
    }
    // 0,0,0,
    // 0,0,0,
    // 1,0,0.
    if((pos+iw-1)>0) {
        lwin[count] = im[pos+iw-1];
        count++;
    }
    // 0,1,0,
    // 0,0,0,
    // 0,0,0.
    if((pos-iw)>0) {
        lwin[count] = im[pos-iw];
        count++;
    }
    // 0,0,0,
    // 0,1,0,
    // 0,0,0.
    lwin[count] = im[pos];
    count++;
    
}
*/
__global__ void etfKernel(
                double *xout, double *yout, double *magout, 
                const double *tx, const double *ty,
                const double *im, const double *gmag,
                int height, int width) {

    // size_t const globalThreadIdx = calculateGlobalIndex();

    // Calculate pixel's location
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    // Only execute valid pixels
    if (x>=width || y>=height) {
         return;
    }

    double ctrx = tx[y*width+x];
    double ctry = ty[y*width+x];
    double ctrgmag = gmag[y*width+x];

    // initialization
   // double win[9] = {0.0};
    double win_tx[9] = {0.0};
    double win_ty[9] = {0.0};
    double win_gmag[9] = {0.0};
    int count = 0;
    // get local win
    for (int j = -1; j < 2; j++) {
        int wy = y+j;
        if (wy>=0 && wy<height){
            for (int i = -1; i < 2; i++) {
                int wx = x+i;
                if (wx>=0 && wx<width){            
                   // win[count] = im[wy*width+wx];
                    win_tx[count] = tx[wy*width+wx];
                    win_ty[count] = ty[wy*width+wx];
                    win_gmag[count] = gmag[wy*width+wx];
                    count++;
                }
            }
        }
    }
    

    // etf operation
    double wm[9] = {0.0};
    mag_weight(wm, ctrgmag, win_gmag, count);
    double ctrxy[2];
    ctrxy[0] = ctrx;
    ctrxy[1] = ctry;
    double winxy[18];
    for(int i = 0; i < count; i++){
        winxy[i] = win_tx[i];
        winxy[i+9] = win_ty[i];
    }
    
    double wd[9] = {0.0};
    dire_weight(wd, ctrxy, winxy, count);
    
    double sum_tx = 0.0;
    double sum_ty = 0.0;
    for (int i = 0; i < count; i++){
        sum_tx += win_tx[i]*wm[i]*wd[i];
        sum_ty += win_ty[i]*wm[i]*wd[i];
    }

    double tmpgmag = sqrt(sum_tx*sum_tx+sum_ty*sum_ty);
    if (tmpgmag != 0){
        sum_tx /= tmpgmag;
        sum_ty /= tmpgmag;
    }
    else {
        sum_tx = 0;
        sum_ty = 0;
    }
/*
    for (int i = 0; i < count; i++){
        xout[y*width+x] += wm[i];
        yout[y*width+x] += wd[i];
    }
*/

    xout[y*width+x] = sum_tx;
    yout[y*width+x] = sum_ty;
    magout[y*width+x] = tmpgmag;
    __syncthreads();
}
