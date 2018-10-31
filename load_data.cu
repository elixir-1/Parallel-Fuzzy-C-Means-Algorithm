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

#define RGB_COMPONENT_COLOR 255

typedef struct {
     int x, y;
     unsigned char *data;
} PPMImage;

/*
----------------------PPM Image Utilities-----------------
*/
static PPMImage *readPPM(char *filename)
{
   char buff[16];
	 PPMImage *img;
	 FILE *fp;
	 int c, rgb_comp_color;
   int S=3;
	 //open PPM file for reading
	 fp = fopen(filename, "rb");
	 if (!fp) {
	      fprintf(stderr, "Unable to open file '%s'\n", filename);
	      exit(1);
	 }

	 //read image format
	 if (!fgets(buff, sizeof(buff), fp)) {
	      perror(filename);
	      exit(1);
	 }

	//check the image format
	if (buff[0] != 'P' || buff[1] != '6') {
	 fprintf(stderr, "Invalid image format (must be 'P6')\n");
	 exit(1);
	}

	//alloc memory form image
	img = (PPMImage *)malloc(sizeof(PPMImage));
	if (!img) {
	 fprintf(stderr, "Unable to allocate memory\n");
	 exit(1);
	}

	//check for comments
	c = getc(fp);
	while (c == '#') {
	while (getc(fp) != '\n') ;
	 c = getc(fp);
	}

	ungetc(c, fp);
	//read image size information
	if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
	 fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
	 exit(1);
	}

	//read rgb component
	if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
	 fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
	 exit(1);
	}

	//check rgb component depth
	if (rgb_comp_color!= RGB_COMPONENT_COLOR) {
	 fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
	 exit(1);
	}
	while (fgetc(fp) != '\n') ;
	//memory allocation for pixel data
  /*int i,j;
  unsigned short int *buf;
  int S=3;
  img->data=(float *)calloc(img->y * img->x * S,sizeof(float));
  buf=(unsigned short int *)calloc(img->x*img->y, sizeof(unsigned short int));
  for (i=0; i < S; i++) {
    fread(buf,2,img->x*img->y,fp);
    for (j=0; j < img->x*img->y;j++) {
      img->data[j*S+i]=buf[j];
    }
  }*/

  img->data = (unsigned char*)malloc(img->x * img->y * sizeof(unsigned char) * S);

	if (!img) {
	 fprintf(stderr, "Unable to allocate memory\n");
	 exit(1);
	}

	//read pixel data from file
	fread(img->data, S * img->x, img->y, fp);

	fclose(fp);
	return img;
}

void writePPM(const char *filename, PPMImage *img)
{
    FILE *fp;
    //open file for output
    fp = fopen(filename, "wb");
    if (!fp) {
         fprintf(stderr, "Unable to open file '%s'\n", filename);
         exit(1);
    }

    //write the header file
    //image format
    fprintf(fp, "P6\n");


    //image size
    fprintf(fp, "%d %d\n",img->x,img->y);

    // rgb component depth
    fprintf(fp, "%d\n",RGB_COMPONENT_COLOR);

    // pixel data
    fwrite(img->data, 3 * img->x, img->y, fp);
    fclose(fp);
}


int load_image_data(char * filename, unsigned char **X, int *S, int *w, int *h){
  PPMImage *image;
  printf("Loading data file\n");
  //const char *f = (const char *)argv[optind];
  printf("before read\n" );
  image = readPPM(filename);
  *w = image->x;
  *h = image->y;
  int bytes = image->x*image->y*(*S);
  printf("file successfully read\n");
  *X = (unsigned char *)malloc(bytes*sizeof(char));
  *X = image->data;
  printf("x done\n");
  return 0;
}
