
__global__ void addVector(
                double *c, double *a, double *b,
                int height, int width) {

    // x corresponds to vertical direction, meaning row
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    // y corresponds to horizontal direction, meaning column
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    // only execute valid pixels
    if (x>=width || y>=height) {
        return;
    }
    int wy = y+1;
    if (wy<0 || wy>=height)
        return;

    c[y*width+x] = a[wy*width+x];
   // c[y*width+x] = y;
   // a[y*width+x] = x;
}
