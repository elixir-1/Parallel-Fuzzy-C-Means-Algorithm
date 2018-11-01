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
#include "serial_load.c"
#include "utils.h"

/*
---------------------External functions---------------------
*/
extern int load_image_data(char * filename, unsigned char **X, int *S, int *w, int *h);


/*
---------------------Global variables-----------------------
*/
#define DIM 256

unsigned char* X_c;
float* X;
unsigned char* V_c;
float* V;
float* U;

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
--------------------Function declarations---------------------
*/
/* Public functions */
int lfcm();


/* Private functions */
int update_centroids();
void update_umatrix(float*);

/* Utilities */
int init();
int is_example_centroid(int k);
float distance(int,int);

int output_centroids(char*);
int output_umatrix(char*);
int output_members(char*);



/*
----------------------Main function---------------------------
*/
int main(int argc, char *argv[]){
    number_of_iterations=0;
    m=2.0;
    S=3;
    char *filename;
    int ch;

    //commandline argument parsing
    const char *parser = "hc:m:e:w:s:";
    while( (ch = getopt(argc, argv, parser)) != -1 ){
        switch(ch){
          case 'h':
                  printf("Usage\n-c number of clusters\n-m fuzziness index\n-e epsilon\n-w write output metrics\n-s seed value\n");
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
    int i,j;
    for(i=0;i<w*h;i++){
      //printf("(");
      for(j=0;j<S;j++){
        tempX[i*S+j] = (int)X_c[i*S+j];
        X[i*S+j] = (float)tempX[i*S+j];
        //printf("%f ", X[i*S+j]);
      }
      //printf(")\n");
    }

    printf("%d ", w);
    printf("%d\n", h);
    N=w*h;

    /*int i,j;
    for(i=0;i<N;i++){
      printf("(");
      for(j=0;j<S;j++){
        printf("%d ", X[i*S+j]);
      }
      printf(")\n");
    }
    */
    printf("Beginning to cluster here...\n");


  	/* Time the fcm algorithm */
  	//getrusage(RUSAGE_SELF, &start_usage);
    clock_t begin = clock();
  	lfcm();
    clock_t end = clock();
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
    double time_spent = (double)(end-begin)/CLOCKS_PER_SEC;
    printf("Computation time: %f\n", time_spent);
    return 0;
}

int lfcm()
{
	float sqrerror_sum;
  float sqrte = 2*epsilon;

	/* Initialize code  */
	init(&sqrerror_sum);
  sqrerror_sum = 2 * epsilon;
	/* Run the updates iteratively */
	while (sqrte > epsilon ) {
    sqrerror_sum=0;
		number_of_iterations++;
    printf("%d\n", number_of_iterations);
		update_centroids();
		update_umatrix(&sqrerror_sum);
    sqrte = sqrt(sqrerror_sum);
    printf("sqrterror: %f\n", sqrte);
	}
  printf("\n");
  printf("%f\n", sqrte);
  update_centroids();

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
			for (x=0; x < S; x++){
        //printf("%f\n", powf(U[k*C+i], m) * X[k*S+x]);
        numerator[x] = numerator[x] + powf(U[k*C+i], m) * X[k*S+x];
      }
		}

		/* Calculate denominator */
		for (k=0; k < N; k++) {
			for (x=0; x < S; x++)
				denominator[x] += powf(U[k*C+i], m);
		}

		/* Calculate V */
    printf("(");
		for (x=0; x < S; x++) {
      //printf("numerator: %f\n", numerator[x]);
      //printf("denominator: %f\n", denominator[x]);
      V[i*S+x]= numerator[x] / denominator[x];
      printf("%f ", V[i*S+x]);
		}
    printf(")\n");
	}  /* endfor: C clusters */

	return 0;
}

void update_umatrix(float *sqrerror_sum)
{
	int i,j,k;
	int example_is_centroid;
	float summation, D_ij, D_ik;
	float newU;
  for(i=0;i<N;i++){
    if( (example_is_centroid=is_example_centroid(i))!=-1 ){
      for(j=0;j<C;j++){
        if( j == example_is_centroid )
          newU=1.0;
        else
          newU=0.0;
        *sqrerror_sum+=powf(U[i*C+j]-newU,2);
        U[i*C+j]=newU;
      }
    }
    else{
      for(j=0;j<C;j++){
        summation=0;

        for(k=0;k<C;k++){
          D_ij = distance(i*S, j*S);
          D_ik = distance(i*S, k*S);
          summation += powf( D_ij / D_ik , (2.0/ (m-1)));
        }

        /* Weight is 1/sum */
        newU=1.0/summation;

        *sqrerror_sum+=powf(U[i*C+j]-newU,2);
        U[i*C+j]=newU;
      }
    }
  }
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
int init(float *sqrerror_sum)
{
	int i,j;
  //int max_value = 256;
	/* Allocate necessary storage */
	V=(float *)calloc(S*C, sizeof(float));

	U=(float *)calloc(C*N,sizeof(float));
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
	/* Once values are populated in V, update the U Matrix for sane values */
	update_umatrix(sqrerror_sum);
  fprintf(stdout,"Initialization completed.\n");

	return 0;
}


/* If X[k] == V[i] for some i, then return that i. Otherwise, return -1 */
int is_example_centroid(int k)
{
	int  i,x;

	for (i=0; i < C; i++) {
		for (x=0; x < S; x++) {
			if ( X[k*S+x] != V[i*S+x] ) break;
		}
		if ( x == S )  /* X==V */
			return i;
	}
	return -1;
}

float distance(int startV1, int startV2){
	int x,y;
	float sum=0;

	for (x=startV1,y=startV2; x < startV1+S && y<startV2+S; x++, y++)
		sum += (X[x] - V[y]) * (X[x] - V[y]);

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

  	sprintf(buf,"%s_serial.centroids", filestem);
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

  	sprintf(buf,"%s_serial.umatrix", filestem);
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

  	sprintf(buf,"%s_serial.members", filestem);
  	fp=FOPEN(buf,"w");
  	for (i=0; i < N; i++) {
  		for (max=j=0; j < C; j++)
  			if ( U[i*C+j] > U[i*C+max] ) max=j;
  		fprintf(fp,"%d\n",max);
  	}
  	fclose(fp);

  	return 0;
  }
