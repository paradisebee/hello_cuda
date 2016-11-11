const int LEN = 4;

__device__ void line3D(double out[][2*LEN+1], double *ctrp, double *vec) {
    double x = ctrp[1];
    double y = ctrp[0];
    double z = ctrp[2];
    double gx = vec[0];
    double gy = vec[1];
    double gz = vec[2];
    for(int i = -LEN; i <= LEN; i++){
        out[0][i+LEN] = x+i*gx;
        out[1][i+LEN] = y+i*gy;
        out[2][i+LEN] = z+i*gz;
    }
}

__device__ double dire_weight(double *a, double *b) {
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; 
}

__device__ double mag_weight(double x, double y) {
    return 0.5*(1+y-x);
}


__global__ void etf3D(
                double *xout, double *yout, double *zout, double *outmag,
                double *tx, double *ty, double *tz,
                double *im, double *gmag,
                int height, int width, int slice) {

    // calculate pixels' location
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int z = (blockIdx.z * blockDim.z) + threadIdx.z;

    // only execute valid pixels
    if(x>=width || y>=height || z>=slice) {
        return;
    }
    // get paralle line centered at current point
    double ctrp[3] = {y,x,z};
    // direction vector
    int ctrpos = z*width*height+y*width+x;
    double ctrV[3] = {tx[ctrpos],
                      ty[ctrpos],
                      tz[ctrpos]};

    double p_line[3][2*LEN+1] = {0.0};
    line3D(p_line, ctrp, ctrV);

    double sum_wd = 0.0;
    double temp[3] = {0.0};
    for (int i = 0; i < 2*LEN+1; i++) {
        if (p_line[0][i]>=0 && p_line[0][i]<width &&
            p_line[1][i]>=0 && p_line[1][i]<height &&
            p_line[2][i]>=0 && p_line[2][i]<slice) {
            int posx = (int)(p_line[0][i]+0.5);
            int posy = (int)(p_line[1][i]+0.5);
            int posz = (int)(p_line[2][i]+0.5);
            int ind = posz*width*height+posy*width+posx;
            if (ind>=height*width*slice) {
                continue;
            }
            double winV[3] = {tx[ind], ty[ind], tz[ind]};
            double wd = dire_weight(ctrV, winV);
            sum_wd += wd;
            double wm = mag_weight(gmag[ctrpos], gmag[ind]);
            temp[0] += wd*wm*tx[ind];
            temp[1] += wd*wm*ty[ind];
            temp[2] += wd*wm*tz[ind];
        }
    }

    if (sum_wd/9<-0.1){
        temp[0] = -temp[0];
        temp[1] = -temp[1];
        temp[2] = -temp[2];
    }
    
    double temp_mag = sqrt(temp[0]*temp[0]+temp[1]*temp[1]+temp[2]*temp[2]);
    if(temp_mag != 0) {
        outmag[ctrpos] = temp_mag;
        xout[ctrpos] = temp[0]/temp_mag;
        yout[ctrpos] = temp[1]/temp_mag;
        zout[ctrpos] = temp[2]/temp_mag;
    }

    __syncthreads();
}
