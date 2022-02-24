//
// Example 1: Single Process, Single Thread, Multiple Devices
//

#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <stdlib.h>
#include <vector>
#include <iostream>
#include "nvToolsExt.h"

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

__global__ void  init1(float *dptr,int i)
{
  int id = threadIdx.x;
  dptr[id] = id;
  printf("GPU: %d,dptr: %f\n",i,dptr[id]);
}
__global__ void gpuFunc(){
  int i=threadIdx.x;
}
int main(int argc, char *argv[]) {

  gpuFunc<<<30,1024>>>();
    // managing 2 devices
    int nDev = 1;
    ncclComm_t comms[nDev];
    const int size = 1024;

    // std::vector<int> devs(nDev);
    // for (int i = 0; i < nDev; ++i)
    // {
    //   devs[i] = i;
    // }
    int devs[1]={0};

    // allocating and initializing device buffers
    float **sendbuff = (float **) malloc(nDev * sizeof(float *));
    float **recvbuff = (float **) malloc(nDev * sizeof(float *));
    float **hptr = (float **) malloc(nDev * sizeof(float *));
    // 有几个设备就创建了几个流
    cudaStream_t *s = (cudaStream_t *) malloc(sizeof(cudaStream_t) * nDev);
    
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        {
          // Device info 
          cudaDeviceProp deviceProp;
          cudaGetDeviceProperties(&deviceProp, i);
          printf("\nDevice %d: \"%s\"%d,%d\n", i, deviceProp.name, deviceProp.major, deviceProp.minor);
        }
        CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
        CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
        CUDACHECK(cudaStreamCreate(s + i));
        // init1<<<1,size>>>(sendbuff[i],i);
    }
        init1<<<1,size>>>(sendbuff[0],0);
    // 见https://gitee.com/liuyin-91/ncclexamples/blob/master/documents/nvdia%E5%AE%98%E6%96%B9documentation.md 创建一个Communicator 章节
    // initializing NCCL
    NCCLCHECK(ncclCommInitAll(comms, nDev, devs));


    // calling NCCL communication API. Group API is required when
    // using multiple devices per thread
    // 详见 https://gitee.com/liuyin-91/ncclexamples/blob/master/documents/nvdia%E5%AE%98%E6%96%B9documentation.md#%E4%BB%8E%E4%B8%80%E4%B8%AA%E7%BA%BF%E7%A8%8B%E7%AE%A1%E7%90%86%E5%A4%9A%E4%B8%AA-gpu 
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; ++i) {
      hptr[i] = (float *)malloc(size * sizeof(float));
      NCCLCHECK(ncclAllReduce((const void *)sendbuff[i],
                              (void *)recvbuff[i], size, ncclFloat, ncclSum,
                              comms[i], s[i]));
    }
    // NCCLCHECK(ncclSend((const void *)sendbuff[0], size, ncclFloat, 1, comms[0], s[0]));
    // NCCLCHECK(ncclRecv(recvbuff[1], size, ncclFloat, 0, comms[1], s[1]));
    // cudaMemcpy(hptr[1], recvbuff[1], size * sizeof(float), cudaMemcpyDeviceToHost);
    // for (int j = 0; j < size; ++j)
    // {
    //   std::cout << "recv-i= " << 1 << " " << hptr[1][j] << "\n";
    // }
    NCCLCHECK(ncclGroupEnd());

    // synchronizing on CUDA streams to wait for completion of NCCL operation
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        // 此函数强制阻塞主机，直到在给定流中的所有操作都完成了
        CUDACHECK(cudaStreamSynchronize(s[i]));
    }

    // free device buffers
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        cudaMemcpy(hptr[i], recvbuff[i], size * sizeof(float), cudaMemcpyDeviceToHost);
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
    }

    // finalizing NCCL
    for (int i = 0; i < nDev; ++i) {
        ncclCommDestroy(comms[i]);
    }

    // for(int i=0;i<size;++i){
    //   for(int j=0;j<nDev;++j)
    //   std::cout<<"i= "<<i<<" "<<hptr[j][i]<<"\n";
    // }
    printf("Success \n");
    return 0;
}