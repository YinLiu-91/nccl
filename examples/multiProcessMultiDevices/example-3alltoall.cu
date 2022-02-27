//
// Multiple Devices per Thread
// 
// 
// compile command: nvcc -g -G ./example-3scatter.cu -o ex3scatter.out -lnccl -lmpi  
//  or remove '-g -G' flag for release version
// 

//
// execute command: mpirun -np 2 ./ex3scatter.out 
//
/* output result:

*/

#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <iostream>

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


static uint64_t getHostHash(const char* string) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}


static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}

__global__ void  init(float *dptr,int myRank)
{
  int id = threadIdx.x;
  dptr[id] = id;
//   printf("kernel-myRank: %d id: %f\n",myRank,dptr[id]);
}

static __inline__ int ncclTypeSize(ncclDataType_t type) {
  switch (type) {
    case ncclInt8:
    case ncclUint8:
      return 1;
    case ncclFloat16:
#if defined(__CUDA_BF16_TYPES_EXIST__)
    case ncclBfloat16:
#endif
      return 2;
    case ncclInt32:
    case ncclUint32:
    case ncclFloat32:
      return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclFloat64:
      return 8;
    default:
      return -1;
  }
}

// 利用ncclGroupStart(),ncclGroupEnd()实现sendrecv
ncclResult_t NCCLSendRecv(void *sendbuff, size_t sendcount, ncclDataType_t datatype, int peer,
                          void *recvbuff,size_t recvcount,ncclComm_t comm, cudaStream_t stream)
{
    ncclGroupStart();
    auto a = ncclSend(sendbuff, sendcount, datatype, peer, comm, stream);
    auto b = ncclRecv(recvbuff, recvcount, datatype, peer, comm, stream);
    ncclGroupEnd();
    if (a||b)
    {
      if(a)
        return a;
      return b;
    }
    return ncclSuccess;
}


ncclResult_t NCCLAlltoall(void *sendbuff, size_t sendcount, ncclDataType_t senddatatype, void *recvbuff,
                         size_t recvcount, ncclDataType_t recvdatatype, int myRank, int nRanks, ncclComm_t comm, cudaStream_t stream)
{

  ncclGroupStart();
  for (int i = 0; i < nRanks; ++i)
  {

    // auto a = ncclSend(sendbuff + i * ncclTypeSize(senddatatype) * sendcount, sendcount, recvdatatype, i, comm, stream);
    auto a=NCCLSendRecv(sendbuff + i * ncclTypeSize(senddatatype) * sendcount,sendcount,senddatatype,i,recvbuff+ i * ncclTypeSize(recvdatatype) * recvcount,
      recvcount,comm,stream);
    if (a)
      return a;
  }
  // for (int i = 0; i < nRanks; ++i)
  // {
  //   auto b = ncclRecv(recvbuff+ i * ncclTypeSize(recvdatatype) * recvcount, recvcount, recvdatatype, i,comm, stream);
  //   if (b)
  //     return b;
  // }
  ncclGroupEnd();
  return ncclSuccess;
}

int main(int argc, char* argv[])
{
    //each process is using two GPUs
    int nDev = 1;
    int size = 3;
    int root = 0;

    int myRank, nRanks, localRank = 0;

    //initializing MPI
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

    //calculating localRank which is used in selecting a GPU
    uint64_t hostHashs[nRanks];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[myRank] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    for (int p = 0; p < nRanks; p++)
    {
      if (p == myRank)
        break;
      if (hostHashs[p] == hostHashs[myRank])
        localRank++;
  }
    std::cout<<"myRank: "<<myRank<<" localRank: "<<localRank<<"\n";

    float **sendbuff = (float **)malloc(nDev * sizeof(float *));
    float **recvbuff = (float **)malloc(nDev * sizeof(float *));
    float **hptr = (float **)malloc(nDev * sizeof(float *));
    cudaStream_t *s = (cudaStream_t *)malloc(sizeof(cudaStream_t) * nDev);

    //picking GPUs based on localRank
    for (int i = 0; i < nDev; ++i)
    {
      CUDACHECK(cudaSetDevice(localRank * nDev + i)); // 给所有设备编号
      CUDACHECK(cudaMalloc(sendbuff + i, nDev * nRanks * size * sizeof(float)));
      CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
      CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
      CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
      CUDACHECK(cudaStreamCreate(s + i));
      hptr[i] = (float *)malloc(nDev * nRanks * size * sizeof(float));
  }


  ncclUniqueId id;
  ncclComm_t comms[nDev];


  //generating NCCL unique ID at one process and broadcasting it to all
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  //initializing NCCL, group API is required around ncclCommInitRank as it is
  //called across multiple GPUs in each thread/process
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; i++)
  {
    CUDACHECK(cudaSetDevice(localRank * nDev + i));
    init<<<1,  nDev * nRanks * size >>>(sendbuff[i], myRank);
    NCCLCHECK(ncclCommInitRank(comms + i, nRanks * nDev, id, myRank * nDev + i));
    cudaMemcpy(hptr[i], sendbuff[i], nDev * nRanks * size * sizeof(float), cudaMemcpyDeviceToHost);
    if (myRank * nDev + i == root)
    {
      std::cout << "myRank" << myRank << " sendbuff[" << i << "]"
                << "\n";
      for (int j = 0; j < nDev * nRanks * size; ++j)
      {
        std::cout << " j: " << j << " hptr[i][j]: " << hptr[i][j] << "\n";
      }
    }
  }
  NCCLCHECK(ncclGroupEnd());


  // scatter Data
  for(int i=0;i<nDev;++i){
    NCCLScather(sendbuff[i], size, ncclFloat, recvbuff[i], size, ncclFloat, root, myRank * nDev + i, nRanks * nDev, comms[i], s[i]);
  }

  for (int i = 0; i < nDev; ++i)
  {
    cudaMemcpy(hptr[i], recvbuff[i],size * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout<<"myRank"<<myRank<<" recvbuff["<<i<<"]"<<"\n";
    for(int j=0;j<size;++j){
        std::cout<<" j: "<<j<<" hptr[i][j]: "<<hptr[i][j]<<"\n";
    }
  }

  //synchronizing on CUDA stream to complete NCCL communication
  for (int i=0; i<nDev; i++)
      CUDACHECK(cudaStreamSynchronize(s[i]));


  //freeing device memory
  for (int i=0; i<nDev; i++) {
     CUDACHECK(cudaFree(sendbuff[i]));
     CUDACHECK(cudaFree(recvbuff[i]));
     free(hptr[i]);
  }


  //finalizing NCCL
  for (int i=0; i<nDev; i++) {
     ncclCommDestroy(comms[i]);
  }


  //finalizing MPI
  MPICHECK(MPI_Finalize());


  printf("[MPI Rank %d] Success \n", myRank);
  return 0;
}