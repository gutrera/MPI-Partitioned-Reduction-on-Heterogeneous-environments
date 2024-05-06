#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"


#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <asm/unistd.h>
#include <signal.h>
#include <sched.h>
#include <complex.h>
#include <errno.h>
#include <netinet/in.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/resource.h>


#define __KERNEL__

#define THREADS_PER_BLOCK_X 16

#define MAX_CPUS_PER_SOCKET 20
#define MAX_CPUS_PER_NODE   40
#define CALLBACKNOTDONE 0
#define CALLBACKDONE 1
#define CALLBACKPROCESSED 2

void printBuff(int* reference, int sendcount);

int calculus = 20;
double f_const = 3.4, d_const= 2.3;
typedef struct {
        double time;
} tdata;

/*************** CUDA ****************/

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

int *addLayer1D_device(const int n) {
  int *buffer = NULL;
  const int buffer_size = sizeof(int) * n;
  cudaError_t error = cudaMalloc((void **)&buffer, buffer_size);
  if (error != cudaSuccess) {
    //printf("Allocation error \n");
    return NULL;
  }
  return buffer;
}

void removeLayer1D_device(int **layer) {
  if ((layer != NULL) && (*layer != NULL)) {
    cudaFree(*layer);
    *layer = NULL;
  }
}

__global__ void calculation_d(int * buffer, int calculus, int count, double f, double g) {

    int id = (threadIdx.x + blockIdx.x * blockDim.x);
    double a = 0;
    // Dummy calculation
    for(int i = 0; i < calculus; i++)
    {
        a += f*g;
    }
    buffer[id%count] = a;
}
/*************** END CUDA ****************/


tdata multilane, native, PR;
int buffer_size;
MPI_Comm shmcomm, opcomm;
int shm_rank, shm_size, op_rank, op_size, rank;

// Headers
int scheme_native ( const void * sendbuf , void * recvbuf , int count ,
                          MPI_Datatype datatype , MPI_Op op , MPI_Comm comm );
int scheme_PR     ( const void * sendbuf , void * recvbuf , int count ,
                          MPI_Datatype datatype , MPI_Op op , MPI_Comm comm );


// Schemes implementation
int scheme_native( void * sendbuf , int * recvbuf , int count ,
                          MPI_Datatype datatype , MPI_Op op , MPI_Comm comm,
                          dim3 grid_dimension, dim3 block_dimension,
                          int *& sendbuf_d
                          )
{

    cudaMemcpy(sendbuf_d, sendbuf, count*sizeof(int), cudaMemcpyHostToDevice);
    cudaError_t err = cudaGetLastError();
    if ( err != cudaSuccess )
    {
       printf("init CUDA Error: %s\n", cudaGetErrorString(err));       
    }

    //Launch the kernel
    calculation_d<<<grid_dimension, block_dimension>>>(sendbuf_d, calculus, count, f_const, d_const);
    err = cudaGetLastError();
    if ( err != cudaSuccess )
    {
       printf("CUDA Error: %s\n", cudaGetErrorString(err));       
    }
    cudaDeviceSynchronize();

    // Use device buffer directly. Possible in latest versions of MPI
    MPI_Allreduce ( sendbuf_d, recvbuf , count , datatype , op , comm );

    return MPI_SUCCESS ;
}

struct infoCallback
{
    int state;
};

void CUDART_CB myStreamCallback(cudaStream_t stream, cudaError_t status, void *data)
{
    infoCallback *d = (infoCallback*)data;

    d->state = CALLBACKDONE;
}

////// SCHEME PROPOSAL:
// 1. Send whole vector
// 2. Offload calculation by partitionining the vector 
// 4. Process results as soon as they are ready, one by one
int scheme_PR ( void * sendbuf , void * recvbuf , int count ,
                          MPI_Datatype datatype , MPI_Op op , MPI_Comm comm,
                          void * sendbuf_d)
{
    int i, block;
    int counts[shm_size], displs[shm_size];
    int extent=1;
    MPI_Request requests[shm_size+1];

    block = count / shm_size ;
    for ( i =0; i < shm_size ; i ++) counts [ i ] = block;
    counts [ shm_size -1] += count % shm_size ;
    displs [0] = 0;
    for ( i =1; i < shm_size ; i ++) displs [ i ] = displs [i -1]+ counts [i -1];

    cudaMemcpy(sendbuf_d, sendbuf, count*sizeof(int), cudaMemcpyHostToDevice);
    cudaError_t err = cudaGetLastError();
    if ( err != cudaSuccess )
    {
       printf("init CUDA Error: %s\n", cudaGetErrorString(err));       
    }

    // Offload input vector for calculation 
    cudaStream_t stream[shm_size];
    infoCallback *info = new infoCallback[shm_size];
    for (int id = 0; id < shm_size; ++id)
    {   
	info[id].state     = CALLBACKNOTDONE;
        checkCuda(cudaStreamCreate(&stream[id]));
        checkCuda(cudaStreamAddCallback(stream[id], myStreamCallback, &info[id], 0));
        dim3 grid_dimension (block, 1, 1 );
        dim3 block_dimension(THREADS_PER_BLOCK_X, 1, 1);
        //Launch the kernel in asynchrounus way
        calculation_d<<<grid_dimension, block_dimension, 0, stream[id]>>>((int*)sendbuf_d+id*block*extent, calculus, block, f_const,d_const);
        cudaError_t err = cudaGetLastError();
        if ( err != cudaSuccess )
        {
           printf("CUDA Error: %s\n", cudaGetErrorString(err));       
           // Possibly: exit(-1) if program cannot continue....
        }
    }
     
    cudaDeviceSynchronize();

    // Copy partitioned results as long as they are available and perform reduction on them
    int processed=0;
    int index=0;
    while (processed < shm_size) {
        if (info[index].state == CALLBACKDONE) { // Partial result available

	    // Copy partition
            cudaMemcpy((int *)recvbuf  + index * block * extent, 
               (int *)sendbuf_d + index * block * extent, 
               block*sizeof(int), 
               cudaMemcpyDeviceToHost);
            cudaError_t err = cudaGetLastError();
            if ( err != cudaSuccess )
            {
               printf("callback return CUDA Error: %s\n", cudaGetErrorString(err));
            }

	    // Intra-group reduction on partition
            MPI_Ireduce ( MPI_IN_PLACE,
                  (int *)recvbuf + index * block * extent ,
                  block, datatype, op, index, shmcomm, &requests[index]);
            info[index].state = CALLBACKPROCESSED;
            if (index == shm_rank) {
		  // Inter-group reduction on partition if I'm in charge of this partition
	          MPI_Iallreduce ( MPI_IN_PLACE ,( int *) recvbuf + shm_rank * block * extent, counts [ shm_rank ] , datatype , op , opcomm, &requests[shm_size] );          
	    }
	    processed++;
        }
        index = (index+1)%shm_size;	
    }
    // Wait for completion of partitioned reductions
    MPI_Status statuses[shm_size+1];
    MPI_Waitall (shm_size+1, requests, statuses);

    //
    // Opportunity to overlap reduction/calculation: Partial reductions are finished and ready to use
    //

    // Re-Build output buffer
    MPI_Allgatherv ( MPI_IN_PLACE , counts [ shm_rank ] , datatype ,
                     recvbuf , counts , displs , datatype , shmcomm );

    for (int i = 0; i < shm_size; ++i)
        checkCuda(cudaStreamDestroy(stream[i]) );


    return MPI_SUCCESS ;
}

void printBuff(int* reference, int sendcount)
{
    for(int i = 0; i < sendcount; i++)
    {
        printf("%d ", reference[i]);
    }
    printf("\n");
}

int main(int argc, char ** argv)
{
    int i, k;
    int *recvbuf;
    int *sendbuf_d, *recvbuf_d;    
    double ini, end;
    int size;
    int sendcount;
    int MAX_LOOPS=3;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //what rank is the current processor

    int color, key;
    int group_size = MAX_CPUS_PER_SOCKET;

   /* parameter: group size */
   if (getenv("GROUP_SIZE")!=NULL)
        group_size = atoi(getenv("GROUP_SIZE"));

   /* inter-node group */
  color = rank%group_size;
    key = rank;
    MPI_Comm_split(MPI_COMM_WORLD, color, key, &opcomm);
    MPI_Comm_size (opcomm, &op_size);
    MPI_Comm_rank (opcomm, &op_rank);

    /* intra-node group */
  color = rank/group_size;
    key = rank;
    MPI_Comm_split(MPI_COMM_WORLD, color, key, &shmcomm);
    MPI_Comm_size (shmcomm, &shm_size);
    MPI_Comm_rank (shmcomm, &shm_rank);

    if (getenv("CALCULUS")!=NULL)
        calculus = atoi(getenv("CALCULUS"));

    /* parameter: size of the message in bytes */
    int ndata_bytes=64;
    if (getenv("NDATA"))
	ndata_bytes = atoi(getenv("NDATA"));

   /* parameter: total number of repetitions of the collective */
   if (getenv("MAX_LOOPS")!=NULL)
        MAX_LOOPS = atoi(getenv("MAX_LOOPS"));

    /* message buffer allocation */
    sendcount=ndata_bytes/sizeof(int);
    int *sendbuf_native = (int*) malloc (ndata_bytes*2);
    int *sendbuf_pr = (int*) malloc (ndata_bytes*2);
    recvbuf = (int*) malloc (ndata_bytes*2);

    /* buffer initialization  */
    for (i=0; i<sendcount; i++) {
	    sendbuf_native[i]=(int)rank;
	    sendbuf_pr[i]=(int)rank;
    }

    /* vector index to avoid cache hit at each reduction iteration */
    int index[MAX_LOOPS];
    for (k=0; k<MAX_LOOPS; k++)
        index[k] = random()%(sendcount);

    /*Alloc memory CUDA on device and copy data from host to device*/
    sendbuf_d = addLayer1D_device(sendcount);
    recvbuf_d = addLayer1D_device(sendcount);
    
    dim3 grid_dimension (sendcount, 1, 1 );
    dim3 block_dimension(THREADS_PER_BLOCK_X, 1, 1);
   
    int * reference = (int*) malloc (ndata_bytes*2);

/********************* NATIVE ******************************/

    ini = MPI_Wtime();
    for (i=0; i<MAX_LOOPS; i++) 
    {
        scheme_native(&sendbuf_native[index[i]], reference, sendcount, MPI_INT, MPI_MIN, 
                       MPI_COMM_WORLD, grid_dimension, block_dimension, sendbuf_d);
    }
    end = MPI_Wtime();

    if (rank==0) printf("\n%d: TEST Native Allreduce   message_size= %7d total_time= %10.4f avg_time= %20.4f\n", 
			rank, sendcount*sizeof(int), end-ini, ((end-ini)/MAX_LOOPS)*1000000);

    native.time = ((end-ini)/MAX_LOOPS)*1000000;

/********************* Partitioned reduction  ******************************/


    ini = MPI_Wtime();
    for (i=0; i<MAX_LOOPS; i++) {
	    scheme_PR (&sendbuf_pr[index[i]], recvbuf, sendcount, MPI_INT, MPI_MIN, 
			    MPI_COMM_WORLD, sendbuf_d);
    }
    end = MPI_Wtime();

    if (rank==0) printf("\n%d: TEST Partitioned Allreduce   message_size= %7d total_time= %10.4f avg_time= %20.4f\n", 
                        rank, sendcount*sizeof(int),  end-ini, ((end-ini)/MAX_LOOPS)*1000000);

    PR.time = ((end-ini)/MAX_LOOPS)*1000000;
 

#ifdef _CHECK   
   if(rank == 0)
     {
	 // printf("Reference   ");
         // printBuff(reference, sendcount);
         // printf("Partitioned ");
         // printBuff(recvbuf, sendcount);
         bool check = true;
         for(int i = 0; i < sendcount; i++)
         {
             if(reference[i] != recvbuf[i])
             {
                 check = false;
                 break;
             }
         }
         if(check)
             printf("Partitioned scheme results ok !");
         else
             printf("Partitioned scheme bad results :(");
     }
#endif

/********************* END  ******************************/

    if (rank == 0) {
       // exec results in microseconds
       printf("\n NUMS; %10d;  %10.4f; %10.4f \n",  ndata_bytes, native.time, PR.time);
    }

    MPI_Finalize();
    free(sendbuf_native);
    free(sendbuf_pr);
    free(recvbuf);
    removeLayer1D_device(&sendbuf_d);
    removeLayer1D_device(&recvbuf_d);

    return 0;
}


