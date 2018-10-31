#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <math.h>
#include <sys/times.h>
#include <sys/resource.h>
#include <limits.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <cuda.h>
#include "utils.h"
#include "load_data.cu"
//#include "load_direct.cu"

#ifndef HANDLE_ERROR
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#endif

/*
--------------External functions-------------
*/
extern int load_image_data(char * filename, unsigned char **X, int *S, int *w, int *h);

/*
--------------------Global Variables---------------
*/

#define U(i,j) U[j][i]
#define DIM 256

unsigned char *X_c;
float *X;
float *U;
unsigned char *V_c;
float *V;

int C;
float m;
int S;
int N;
float epsilon;
int w;
int h;

int max_value;
int write_centroids=0;
int write_members=0;
int write_umatrix=0;
int number_of_iterations;
long seed;

/*
-----------------------Function Declarations------------------
*/

/* Public functions */
int lfcm(float** U_d,float** V_d,float* X_d);


/* Private functions */
int update_centroids();
__global__ void update_umatrix(float*,float*,float*,float*,int,int,int,float);

/* Utilities */
int init(float **U_d, float **V_d, float *X_d);
__device__ int is_example_centroid(float* V_d,float* X_d,int k,int S, int C);
__device__ float distance(float *,float *,int,int,int);

int output_centroids(char*);
int output_umatrix(char*);
int output_members(char*);

/*
-----------------------Main function-------------------------
*/

int main(int argc, char *argv[]){
    number_of_iterations=0;
    m=2.0;
    S=3;
    char *filename;
    float *V_d, *X_d;
    float *U_d;
    int ch;

    //commandline argument parsing
    const char *parser = "hc:m:e:w:s:x:y:";
    while( (ch = getopt(argc, argv, parser)) != -1 ){
        switch(ch){
          case 'h':
                  printf("Usage\n-c number of clusters\n-m fuzziness index\n-e epsilon\n-w write output metrics\n");
                  exit(1);
          case 'c':
                  C = atoi(optarg);
                  break;
          case 'm':
                  m = atof(optarg);
                  break;
          case 'e':
                  epsilon = atof(optarg);
                  break;
          case 'w':
                  if(!strcmp(optarg, "umatrix")) write_umatrix=1;
                  if(!strcmp(optarg, "centroids")) write_centroids=1;
                  if(!strcmp(optarg, "members")) write_members=1;
                  if(!strcmp(optarg, "all")) write_members=write_umatrix=write_centroids=1;
                  break;
          case 's':
                  seed = atol(optarg);
        }
    }

    //Loading the required input data
    filename = argv[optind];
    load_image_data(argv[optind], &X_c, &S, &w, &h);

    //Preprocess data
    int *tempX = (int *)malloc(w*h*S*sizeof(int));
    X = (float *)malloc(w*h*S*sizeof(float));

    for(int i=0;i<w*h;i++){
      //printf("(");
      for(int j=0;j<S;j++){
        tempX[i*S+j] = (int)X_c[i*S+j];
        X[i*S+j] = (float)tempX[i*S+j];
        //printf("%f ", X[i*S+j]);
      }
      //printf(")\n");
    }

    HANDLE_ERROR(cudaMalloc(&X_d, w*h*S*sizeof(float)));
    printf("cuda memory assigned\n");
    HANDLE_ERROR(cudaMemcpy(X_d, X, w*h*S, cudaMemcpyHostToDevice));
    printf("Done loading data\n");

    printf("%d ", w);
    printf("%d\n", h);
    N=w*h;
    /*
    for(int i=0;i<N;i++){
      printf("(");
      for(int j=0;j<S;j++){
        printf("%d ", X[i*S+j]);
      }
      printf(")\n");
    }
    */
    printf("Beginning to cluster here...\n");


  	/* Time the fcm algorithm */
  	//getrusage(RUSAGE_SELF, &start_usage);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
  	lfcm(&U_d,&V_d,X_d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time=0;
    cudaEventElapsedTime(&elapsed_time, start, stop);
  	//getrusage(RUSAGE_SELF, &end_usage);


  	/* Output whatever clustering results we need */
  	if ( write_centroids ) output_centroids(filename);
  	if ( write_umatrix   ) output_umatrix(filename);
  	if ( write_members   ) output_members(filename);


  	/* Output timing numbers */
  	//perf_times=timing_of(start_usage, end_usage);
  	///printf("Timing: %f user, %f system, %f total.\n",
  	//perf_times[0], perf_times[1], perf_times[0] +
  	//perf_times[1]);

  	printf("Clustering required %d iterations.\n", number_of_iterations);
    printf("Code Run time: %f\n", elapsed_time);

  	return 0;
}

int lfcm(float** U_d,float** V_d,float* X_d)
{
	float sqrerror[((N+DIM-1)/DIM)*(C/1)];
	float *sqrerror_d;
	float sqrerror_sum;
  float sqrte = 2*epsilon;
	sqrerror_sum= 2 * epsilon;
	/* Initialize code  */
	init(U_d,V_d,X_d);
	cudaDeviceSynchronize();
	HANDLE_ERROR(cudaMalloc(&sqrerror_d,((N+DIM-1)/DIM)*sizeof(float)));
	printf("Beginning GPU side code\n");

	/* Run the updates iteratively */
	while (sqrte > epsilon ) {
		number_of_iterations++;
    printf("iteration: %d\n", number_of_iterations);
		HANDLE_ERROR(cudaMemcpy(U,*U_d,N*C*sizeof(float),cudaMemcpyDeviceToHost));
    //output_umatrix("");
		update_centroids();
		HANDLE_ERROR(cudaMemcpy(*V_d,V,C*S*sizeof(float),cudaMemcpyHostToDevice));
		update_umatrix<<<(N+DIM-1)/DIM,DIM>>>(sqrerror_d,*U_d,*V_d,X_d,C,N,S,m);
		//HANDLE_ERROR(cudaGetLastError());
		HANDLE_ERROR(cudaMemcpy(sqrerror,sqrerror_d,((N+DIM-1)/DIM)*sizeof(float),cudaMemcpyDeviceToHost));
		sqrerror_sum=0;
		cudaDeviceSynchronize();
		for(int i=0; i<((N+DIM-1)/DIM); i++)
			sqrerror_sum+=sqrerror[i];
    sqrte = sqrt(sqrerror_sum);
    printf("sqrterror: %f\n", sqrte);
	}
  printf("\n");
  printf("%f\n", sqrte);
  update_centroids();

	HANDLE_ERROR(cudaMemcpy(U,*U_d,N*C*sizeof(float),cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(V,*V_d,C*S*sizeof(float),cudaMemcpyDeviceToHost));
	return 0;
}


/*
   update_centroids()
    Given a membership matrix U, recalculate the cluster centroids as the
    "weighted" mean of each contributing example from the dataset. Each
    example contributes by an amount proportional to the membership value.
 */
int update_centroids()
{
	  int i,k,x;
	  float *numerator, *denominator;
	  numerator = (float *)malloc(S*sizeof(float));
	  denominator = (float *)malloc(S*sizeof(float));

	  /* For each cluster */
	  for (i=0; i < C; i++)  {
		/* Zero out numerator and denominator options */
		for (x=0; x < S; x++) {
			numerator[x]=0;
			denominator[x]=0;
		}

		/* Calculate numerator */
		for (k=0; k < N; k++) {
			for (x=0; x < S; x++)
				numerator[x] += powf(U[k*C+i], m) * X[k*S+x];
		}

		/* Calculate denominator */
		for (k=0; k < N; k++) {
			for (x=0; x < S; x++)
				denominator[x] += powf(U[k*C+i], m);
		}
    printf("(");
		/* Calculate V */
		for (x=0; x < S; x++) {
			V[i*S+x]= numerator[x] / denominator[x];
      printf("%f ", V[i*S+x]);
		}
    printf(")\n");
	}  /* endfor: C clusters */

	return 0;
}

__global__ void update_umatrix(float *sqrerror,float* U_d, float* V_d, float* X_d,int C,int N,int S,float m)
{

	int j,k;
	int example_is_centroid;
	float summation, D_ki, D_kj;
	float newU;

	__shared__ float tmp_sqrerror[DIM];
	/* For each example in the dataset */
	k = threadIdx.x + blockIdx.x*blockDim.x;
	int local_offset = threadIdx.x;
	tmp_sqrerror[local_offset]=0;
	if(k<N)
	{
		/* Special case: If Example is equal to a Cluster Centroid,
       then U=1.0 for that cluster and 0 for all others */
		if ( (example_is_centroid=is_example_centroid(V_d,X_d,k,S,C)) != -1 ) {
			for(int i=0; i<C; i++)
			{
			if ( i == example_is_centroid )
				newU=1.0;
			else
				newU=0.0;

      tmp_sqrerror[local_offset] += powf(U_d[k*C+i] - newU, 2);

    	U_d[k*C+i]=newU;
			}
			return;
		}
	/* For each class */
	for(int i=0; i< C; i++)
	{
		summation=0;

		/* Calculate summation */
		for (j=0; j < C; j++) {
			D_ki=distance(X_d, V_d,k*S,i*S,S);
			D_kj=distance(X_d, V_d,k*S,j*S,S);
			summation += powf( D_ki / D_kj , (2.0/ (m-1)));
		}

		/* Weight is 1/sum */
		newU=1.0/summation;

		/* Add to the squareDifference */
		tmp_sqrerror[local_offset] += powf(U_d[k*C+i] - newU, 2);

		U_d[k*C+i]=newU;

	}

	}
	__syncthreads();
	int t= blockDim.x/2;
	while(t>0)
	{
		if(k+t < N && threadIdx.x<t)
			tmp_sqrerror[local_offset] += tmp_sqrerror[local_offset+t];
		t/=2;
		__syncthreads();
	}

	if(threadIdx.x==0)
		sqrerror[blockIdx.x] = tmp_sqrerror[0];

}

/*===================================================
  Utilities

  init()
  checkIfExampleIsCentroid()
  distance()

  ===================================================*/

/* Allocate storage for U and V dynamically. Also, copy over the
   variables that may have been externally set into short names,
   which are private and easier to access.
 */
int init(float** U_d, float** V_d, float* X_d)
{
	int i,j;
  //int max_value = 256;
	/* Allocate necessary storage */
	V=(float *)CALLOC(S*C, sizeof(float));

	U=(float *)CALLOC(C*N,sizeof(float));
	HANDLE_ERROR(cudaMalloc(U_d,N*C*sizeof(float)));
	HANDLE_ERROR(cudaMalloc(V_d,C*S*sizeof(float)));
	/* Place random values in V, then update U matrix based on it */
	srand(seed);
	for (i=0; i < C; i++) {
    printf("(");
		for (j=0; j < S; j++) {
      float temp=((float)rand())/RAND_MAX;
			V[i*S+j]= temp*255;
      printf("%f ", V[i*S+j]);
		}
    printf(")\n");
	}
	float *dummy;
	cudaMalloc(&dummy,N*sizeof(float));
	HANDLE_ERROR(cudaMemcpy(*V_d,V,C*S*sizeof(float),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(X_d,X,N*S*sizeof(float),cudaMemcpyHostToDevice));
	/* Once values are populated in V, update the U Matrix for sane values */
	update_umatrix<<<(N+DIM-1)/DIM,DIM>>>(dummy,*U_d,*V_d,X_d,C,N,S,m);
  cudaDeviceSynchronize();

  //HANDLE_ERROR(cudaGetLastError());
  fprintf(stdout,"Initialization completed.\n");

	return 0;
}


/* If X[k] == V[i] for some i, then return that i. Otherwise, return -1 */
__device__ int is_example_centroid(float* V_d,float* X_d,int k,int S, int C)
{
	int  i,x;

	for (i=0; i < C; i++) {
		for (x=0; x < S; x++) {
			if ( X_d[k*S+x] != V_d[i*S+x] ) break;
		}
		if ( x == S )  /* X==V */
			return i;
	}
	return -1;
}

__device__ float distance(float *v1, float *v2,int startV1,int startV2,int S)
{
	int x,i;
	float sum=0;

	for (x=startV1,i=startV2; x < startV1+S && i<startV2+S; x++, i++)
		sum += (v1[x] - v2[i]) * (v1[x] - v2[i]);

	return sqrtf(sum);
}



/*=====================================================
  Public output utilities

  output_centroids()
  output_umatrix()
  output_members()
  =====================================================*/
  int output_centroids(char *filestem)
  {
  	FILE *fp;
  	char buf[DIM];
  	int i,j;

  	sprintf(buf,"%s.centroids", filestem);
  	fp=FOPEN(buf,"w");
  	for (i=0;i < C ;i++) {
  		for (j=0; j < S; j++)
  			fprintf(fp, "%f\t",V[i*S+j]);
  		fprintf(fp,"\n");
  	}
  	fclose(fp);

  	return 0;
  }

  int output_umatrix(char *filestem)
  {
  	FILE *fp;
  	char buf[DIM];
  	int i,j;

  	sprintf(buf,"%s.umatrix", filestem);
  	fp=FOPEN(buf,"w");
  	for (i=0; i < N; i++) {
  		for (j=0; j < C; j++)
  			fprintf(fp,"%f\t", U[i*C+j]);
  		fprintf(fp,"\n");
  	}
  	fclose(fp);

  	return 0;
  }

  int output_members(char *filestem)
  {
  	FILE *fp;
  	char buf[DIM];
  	int i,j,max;

  	sprintf(buf,"%s.members", filestem);
  	fp=FOPEN(buf,"w");
  	for (i=0; i < N; i++) {
  		for (max=j=0; j < C; j++)
  			if ( U[i*C+j] > U[i*C+max] ) max=j;
  		fprintf(fp,"%d\n",max);
  	}
  	fclose(fp);

  	return 0;
  }
