#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
#ifdef ATR_SUPPORT
 #include <tiffio.h>
#endif

int load_img(char *filename, float **ds,float **ds_d, int *s, int *n, int image_width, int image_length){
  FILE *fp;
  int S=3;
  int i,j;
  unsigned short int *buf;
  float *X;
  fprintf(stderr,"Loading MRI image %s...", filename);

  fp=FOPEN(filename,"r");

  /* Allocate storage */
  X=(float *)CALLOC(image_length * image_width * S,sizeof(float));
  HANDLE_ERROR(cudaMalloc(ds_d,image_length * image_width * S*sizeof(float)));
  buf=(unsigned short int *)CALLOC(image_width*image_length, sizeof(unsigned short int));
  for (i=0; i < S; i++) {
    fread(buf,2,image_width*image_length,fp);
    for (j=0; j < image_width*image_length;j++) {
      X[j*S+i]=buf[j];
    }
  }

  fclose(fp);
  fprintf(stderr,"done (%d examples).\n", image_width * image_length);
  *ds=X;
  *s=S;
  *n=image_length*image_width;
  cudaGetErrorString(cudaMemcpy(*ds_d,X,image_length * image_width * S * sizeof(float),cudaMemcpyHostToDevice));

  return 0;
}
