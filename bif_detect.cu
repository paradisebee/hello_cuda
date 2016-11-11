#include <math.h>

// *********** helper functions ***************
__device__ double  circle(double *xunit, double *yunit, 
                          double initX, double initY, 
                          double Radius, double stepSize) {
    double ang[stepSize] = {0.0};
    for (int i = 0; i < stepSize; i++){
        ang[i] = i*M_PI/stepSize;
    }
}


// ********************************************


// find the new center
__device__ getNewCenter()

// main process, to get profile
__device__ bool getProfile(double *centroid, double *curP, double Radius,
                           double *im, double *tx, double *ty) {

}


__global__ void bifurcation_detection(double *bif, double Radius,
                                      double *im, double *mask,
                                      double *tx, double *ty) {


    // calculate pixels' location
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    // Only execute valid pixels
    if (x>=width || y>=height || mask[y*width+x] != 1) {
        return;
    }
    
    // initialze local params
    double curP[2] = {y,x};
    int count = 0, int maxCount = 10;
    double pts[2][maxCount] = {0.0};
    double centroid[2] = {0.0};
    // main loop
    while(true) {
        bool success_flag = getProfile(centroid, curP, Radius, im, tx, ty);
    }

