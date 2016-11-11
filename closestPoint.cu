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

__global__ void closestPoint(double *corrBdy, double *dist,
                             double *bdy, double *mask, 
                             int height, int width) {

    // Calculate pixel's location
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    // Only execut valid pixels
    if (x>=width || y>=height || mask[y*width+x]==0 || bdy[y*width+x]==1) {
        return;
    }

    double min_dist = max(height, width); 
    double cur_dist = 0.0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (bdy[i*width+j] == 0)
                continue;
            
            cur_dist = sqrt(double((x-j)*(x-j))+double((y-i)*(y-i)));
            if (cur_dist < min_dist) {
                min_dist = cur_dist;
                corrBdy[y*width+x] = j*height+i+1;
                if (min_dist < 1.1)
                    break;
            }
        }
    }
    /*int bdyx = corrx[y*width+x];
    int bdyy = corry[y*width+x];
    if (x-bdyx != 0 || y-bdyy != 0) {
        double angle = (double(x-bdyx) * gx[bdyy*width+bdyx] + double(y-bdyy) * gy[bdyy*width+bdyx])/sqrt(double((x-bdyx)*(x-bdyx))+double((y-bdyy)*(y-bdyy)));
        if (angle < 0)
            dist[y*width+x] = 100;
        else
            dist[y*width+x] = min_dist;
    }*/
    dist[y*width+x] = min_dist;
    __syncthreads();
}
