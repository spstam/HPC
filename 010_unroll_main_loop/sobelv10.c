// This will apply the sobel filter and return the PSNR between the golden sobel and the produced sobel
// sobelized image
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>

#define SIZE	4096
#define INPUT_FILE	"input.grey"
#define OUTPUT_FILE	"output_sobel.grey"
#define GOLDEN_FILE	"golden.grey"


//initialise Look Up Table for calc roots
#define MAX_P 65025
unsigned short sqrt_LUT[MAX_P + 1];

double sobel(unsigned char *input, unsigned char *output, unsigned char *golden);
int convolution2D(int posy, int posx, const unsigned char *input, char operator[][3]);

/* The arrays holding the input image, the output image and the output used *
 * as golden standard. The luminosity (intensity) of each pixel in the      *
 * grayscale image is represented by a value between 0 and 255 (an unsigned *
 * character). The arrays (and the files) contain these values in row-major *
 * order (element after element within each row and row after row. 			*/
unsigned char input[SIZE*SIZE], output[SIZE*SIZE], golden[SIZE*SIZE];


/* Implement a 2D convolution of the matrix with the operator */
/* posy and posx correspond to the vertical and horizontal disposition of the *
 * pixel we process in the original image, input is the input image and       *
 * operator the operator we apply (horizontal or vertical). The function ret. *
 * value is the convolution of the operator with the neighboring pixels of the*
 * pixel we process.														  */



/* The main computational function of the program. The input, output and *
 * golden arguments are pointers to the arrays used to store the input   *
 * image, the output produced by the algorithm and the output used as    *
 * golden standard for the comparisons.									 */
double sobel(unsigned char *input, unsigned char * restrict output, unsigned char * restrict golden)
{
	double PSNR = 0, t;
	int i, j;
	unsigned int p;
	int help_p_vert, help_p_horiz;
	//int res;
	struct timespec  tv1, tv2;
	FILE *f_in, *f_out, *f_golden;
    // unsigned int input_idx, help_4_mult;
	// help_4_mult = SIZE;

	/* The first and last row of the output array, as well as the first  *
     * and last element of each column are not going to be filled by the *
     * algorithm, therefore make sure to initialize them with 0s.		 */
	memset(output, 0, SIZE*sizeof(unsigned char));
	memset(&output[SIZE*(SIZE-1)], 0, SIZE*sizeof(unsigned char));
	for (i = 1; i < SIZE-1; i++) {
		output[i*SIZE] = 0;
		output[i*SIZE + SIZE - 1] = 0;
	}

	/* Open the input, output, golden files, read the input and golden    *
     * and store them to the corresponding arrays.						  */
	f_in = fopen(INPUT_FILE, "r");
	if (f_in == NULL) {
		printf("File " INPUT_FILE " not found\n");
		exit(1);
	}
  
	f_out = fopen(OUTPUT_FILE, "wb");
	if (f_out == NULL) {
		printf("File " OUTPUT_FILE " could not be created\n");
		fclose(f_in);
		exit(1);
	}  
  
	f_golden = fopen(GOLDEN_FILE, "r");
	if (f_golden == NULL) {
		printf("File " GOLDEN_FILE " not found\n");
		fclose(f_in);
		fclose(f_out);
		exit(1);
	}    

	fread(input, sizeof(unsigned char), SIZE*SIZE, f_in);
	fread(golden, sizeof(unsigned char), SIZE*SIZE, f_golden);
	fclose(f_in);
	fclose(f_golden);
  
	/* This is the main computation. Get the starting time. */
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
	/* For each pixel of the output image */

	unsigned int root=0;
	unsigned int next_square = 1;

	//fill Look Up Table until the max_pixel_value^2
	for (i = 0; i <= MAX_P - 16; i += 16) {
		if (i == next_square) {
			root++;
			next_square = (root + 1) * (root + 1);
		}
		sqrt_LUT[i] = root;
		if (i + 1 == next_square) {
			root++;
			next_square = (root + 1) * (root + 1);
		}
		sqrt_LUT[i + 1] = root;
		if (i + 2 == next_square) {
			root++;
			next_square = (root + 1) * (root + 1);
		}
		sqrt_LUT[i + 2] = root;
		if (i + 3 == next_square) {
			root++;
			next_square = (root + 1) * (root + 1);
		}
		sqrt_LUT[i + 3] = root;
		if (i +4== next_square) {
			root++;
			next_square = (root + 1) * (root + 1);
		}
		sqrt_LUT[i+4] = root;
		if (i + 5 == next_square) {
			root++;
			next_square = (root + 1) * (root + 1);
		}
		sqrt_LUT[i + 5] = root;
		if (i + 6 == next_square) {
			root++;
			next_square = (root + 1) * (root + 1);
		}
		sqrt_LUT[i + 6] = root;
		if (i + 7 == next_square) {
			root++;
			next_square = (root + 1) * (root + 1);
		}
		sqrt_LUT[i + 7] = root;

		if (i +8== next_square) {
			root++;
			next_square = (root + 1) * (root + 1);
		}
		sqrt_LUT[i + 8] = root;
		if (i + 9 == next_square) {
			root++;
			next_square = (root + 1) * (root + 1);
		}
		sqrt_LUT[i + 9] = root;
		if (i + 10 == next_square) {
			root++;
			next_square = (root + 1) * (root + 1);
		}
		sqrt_LUT[i + 10] = root;
		if (i + 11 == next_square) {
			root++;
			next_square = (root + 1) * (root + 1);
		}
		sqrt_LUT[i + 11] = root;
		if (i +12== next_square) {
			root++;
			next_square = (root + 1) * (root + 1);
		}
		sqrt_LUT[i+12] = root;
		if (i + 13 == next_square) {
			root++;
			next_square = (root + 1) * (root + 1);
		}
		sqrt_LUT[i + 13] = root;
		if (i + 14 == next_square) {
			root++;
			next_square = (root + 1) * (root + 1);
		}
		sqrt_LUT[i + 14] = root;
		if (i + 15 == next_square) {
			root++;
			next_square = (root + 1) * (root + 1);
		}
		sqrt_LUT[i + 15] = root;
	}

	// 2. Tail loop for the remainder (0 to 3 elements)
	// 'i' now holds the starting index for the remaining elements
	for (; i <= MAX_P; i++) {
		if (i == next_square) {
			root++;
			next_square = (root + 1) * (root + 1);
		}
		sqrt_LUT[i] = root;
	}

	//To reduce output idx ops(i*SIZE+j)
	unsigned int output_idx = SIZE +1;
	unsigned char *p_in_tl;
	unsigned char *p_in_ml;
	unsigned char *p_in_bl;

	//INTERCHANGED THIS LOOP
	for (i=1; i<SIZE-1; i+=1) {
	p_in_tl = &input[(i - 1) * SIZE];
	p_in_ml = p_in_tl + SIZE; // Middle row
	p_in_bl = p_in_ml + SIZE; // Bottom row
	for (j=1; j<SIZE-4; j+=4 ) {
		/* Apply the sobel filter and calculate the magnitude *
		* of the derivative.								  */
			

			help_p_horiz = (p_in_tl[2] - p_in_tl[0]) +
                            ((p_in_ml[2] - p_in_ml[0]) << 1) +
                            (p_in_bl[2] - p_in_bl[0]);

                // Gy
            help_p_vert = (p_in_tl[0] + (p_in_tl[1] << 1) + p_in_tl[2]) -
                        (p_in_bl[0] + (p_in_bl[1] << 1) + p_in_bl[2]);

			p = help_p_horiz * help_p_horiz + help_p_vert * help_p_vert;
			//p = pow(convolution2D(i, j, input, horiz_operator), 2) + 
			//	pow(convolution2D(i, j, input, vert_operator), 2);
			
			//res = (int)sqrt(p);
			
			/* If the resulting value is greater than 255, clip it *
			 * to 255.											   */
			//omit unnecessary sqrt ops
			 if (p > 65025)
				output[output_idx] = 255;      
			else
				output[output_idx] = (unsigned char)sqrt_LUT[p];

			//output_idx++ because j++ 	
			output_idx +=1;

			//increment pointers
			p_in_bl++;
			p_in_ml++;
			p_in_tl++;

			help_p_horiz = (p_in_tl[2] - p_in_tl[0]) +
                            ((p_in_ml[2] - p_in_ml[0]) << 1) +
                            (p_in_bl[2] - p_in_bl[0]);

                // Gy
            help_p_vert = (p_in_tl[0] + (p_in_tl[1] << 1) + p_in_tl[2]) -
                        (p_in_bl[0] + (p_in_bl[1] << 1) + p_in_bl[2]);

			p = help_p_horiz * help_p_horiz + help_p_vert * help_p_vert;

			/* If the resulting value is greater than 255, clip it *
			 * to 255.											   */
			//omit unnecessary sqrt ops
			 if (p > 65025)
				output[output_idx] = 255;      
			else
				output[output_idx] = (unsigned char)sqrt_LUT[p];

			//output_idx++ because j++ 	
			output_idx +=1;

			//increment pointers
			p_in_bl++;
			p_in_ml++;
			p_in_tl++;
			
			//loop 3,4 
			help_p_horiz = (p_in_tl[2] - p_in_tl[0]) +
                            ((p_in_ml[2] - p_in_ml[0]) << 1) +
                            (p_in_bl[2] - p_in_bl[0]);

                // Gy
            help_p_vert = (p_in_tl[0] + (p_in_tl[1] << 1) + p_in_tl[2]) -
                        (p_in_bl[0] + (p_in_bl[1] << 1) + p_in_bl[2]);

			p = help_p_horiz * help_p_horiz + help_p_vert * help_p_vert;
			//p = pow(convolution2D(i, j, input, horiz_operator), 2) + 
			//	pow(convolution2D(i, j, input, vert_operator), 2);
			
			//res = (int)sqrt(p);
			
			/* If the resulting value is greater than 255, clip it *
			 * to 255.											   */
			//omit unnecessary sqrt ops
			 if (p > 65025)
				output[output_idx] = 255;      
			else
				output[output_idx] = (unsigned char)sqrt_LUT[p];

			//output_idx++ because j++ 	
			output_idx +=1;

			//increment pointers
			p_in_bl++;
			p_in_ml++;
			p_in_tl++;

			help_p_horiz = (p_in_tl[2] - p_in_tl[0]) +
                            ((p_in_ml[2] - p_in_ml[0]) << 1) +
                            (p_in_bl[2] - p_in_bl[0]);

                // Gy
            help_p_vert = (p_in_tl[0] + (p_in_tl[1] << 1) + p_in_tl[2]) -
                        (p_in_bl[0] + (p_in_bl[1] << 1) + p_in_bl[2]);

			p = help_p_horiz * help_p_horiz + help_p_vert * help_p_vert;

			/* If the resulting value is greater than 255, clip it *
			 * to 255.											   */
			//omit unnecessary sqrt ops
			 if (p > 65025)
				output[output_idx] = 255;      
			else
				output[output_idx] = (unsigned char)sqrt_LUT[p];

			//output_idx++ because j++ 	
			output_idx +=1;

			//increment pointers
			p_in_bl++;
			p_in_ml++;
			p_in_tl++;

		}

		for (; j<SIZE-1; j+=1 ) {
		/* Apply the sobel filter and calculate the magnitude *
		* of the derivative.								  */
			help_p_horiz = (p_in_tl[2] - p_in_tl[0]) +
                            ((p_in_ml[2] - p_in_ml[0]) << 1) +
                            (p_in_bl[2] - p_in_bl[0]);

                // Gy
            help_p_vert = (p_in_tl[0] + (p_in_tl[1] << 1) + p_in_tl[2]) -
                        (p_in_bl[0] + (p_in_bl[1] << 1) + p_in_bl[2]);

			p = help_p_horiz * help_p_horiz + help_p_vert * help_p_vert;
			//p = pow(convolution2D(i, j, input, horiz_operator), 2) + 
			//	pow(convolution2D(i, j, input, vert_operator), 2);
			
			//res = (int)sqrt(p);
			
			/* If the resulting value is greater than 255, clip it *
			 * to 255.											   */
			//omit unnecessary sqrt ops
			 if (p > 65025)
				output[output_idx] = 255;      
			else
				output[output_idx] = (unsigned char)sqrt_LUT[p];

			//output_idx++ because j++ 	
			output_idx +=1;

			//increment pointers
			p_in_bl++;
			p_in_ml++;
			p_in_tl++;

		}
		//to skip padding pixels(columns)
		output_idx+=2;
		//update i*SIZE
		// help_4_mult += SIZE;
	}

	/* Now run through the output and the golden output to calculate *
	 * the MSE and then the PSNR.									 */
	output_idx = SIZE +1;	
	 for (i=1; i<SIZE-1; i++) {
		for ( j=1; j<SIZE-16; j+=16 ) {
			if ((output[output_idx] - golden[output_idx]) != 0){
				t = (output[output_idx] - golden[output_idx]);
				PSNR += t*t;
			}
			output_idx += 1;

			if ((output[output_idx] - golden[output_idx]) != 0){
				t = (output[output_idx] - golden[output_idx]);
				PSNR += t*t;
			}
			output_idx += 1;

			if ((output[output_idx] - golden[output_idx]) != 0){
				t = (output[output_idx] - golden[output_idx]);
				PSNR += t*t;
			}
			output_idx += 1;

			if ((output[output_idx] - golden[output_idx]) != 0){
				t = (output[output_idx] - golden[output_idx]);
				PSNR += t*t;
			}
			output_idx += 1;

			//5

			if ((output[output_idx] - golden[output_idx]) != 0){
				t = (output[output_idx] - golden[output_idx]);
				PSNR += t*t;
			}
			output_idx += 1;

			if ((output[output_idx] - golden[output_idx]) != 0){
				t = (output[output_idx] - golden[output_idx]);
				PSNR += t*t;
			}
			output_idx += 1;

			if ((output[output_idx] - golden[output_idx]) != 0){
				t = (output[output_idx] - golden[output_idx]);
				PSNR += t*t;
			}
			output_idx += 1;

			if ((output[output_idx] - golden[output_idx]) != 0){
				t = (output[output_idx] - golden[output_idx]);
				PSNR += t*t;
			}
			output_idx += 1;

			//9

			if ((output[output_idx] - golden[output_idx]) != 0){
				t = (output[output_idx] - golden[output_idx]);
				PSNR += t*t;
			}
			output_idx += 1;

			if ((output[output_idx] - golden[output_idx]) != 0){
				t = (output[output_idx] - golden[output_idx]);
				PSNR += t*t;
			}
			output_idx += 1;

			if ((output[output_idx] - golden[output_idx]) != 0){
				t = (output[output_idx] - golden[output_idx]);
				PSNR += t*t;
			}
			output_idx += 1;

			if ((output[output_idx] - golden[output_idx]) != 0){
				t = (output[output_idx] - golden[output_idx]);
				PSNR += t*t;
			}
			output_idx += 1;
			
			//13

			if ((output[output_idx] - golden[output_idx]) != 0){
				t = (output[output_idx] - golden[output_idx]);
				PSNR += t*t;
			}
			output_idx += 1;

			if ((output[output_idx] - golden[output_idx]) != 0){
				t = (output[output_idx] - golden[output_idx]);
				PSNR += t*t;
			}
			output_idx += 1;

			if ((output[output_idx] - golden[output_idx]) != 0){
				t = (output[output_idx] - golden[output_idx]);
				PSNR += t*t;
			}
			output_idx += 1;

			if ((output[output_idx] - golden[output_idx]) != 0){
				t = (output[output_idx] - golden[output_idx]);
				PSNR += t*t;
			}
			output_idx += 1;
		}

		for(; j<SIZE -1; j++){
			if ((output[output_idx] - golden[output_idx]) != 0){
				t = (output[output_idx] - golden[output_idx]);
				PSNR += t*t;
			}
			output_idx += 1;
		}

		output_idx+=2;
	}
	


	PSNR /= (double)(SIZE*SIZE);
	PSNR = 10*log10(65536/PSNR);

	/* This is the end of the main computation. Take the end time,  *
	 * calculate the duration of the computation and report it. 	*/
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);

	printf ("Total time = %10g seconds\n",
			(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
			(double) (tv2.tv_sec - tv1.tv_sec));

  
	/* Write the output file */
	fwrite(output, sizeof(unsigned char), SIZE*SIZE, f_out);
	fclose(f_out);
  
	return PSNR;
}


int main(int argc, char* argv[])
{
	double PSNR;
	PSNR = sobel(input, output, golden);
	printf("PSNR of original Sobel and computed Sobel image: %g\n", PSNR);
	printf("A visualization of the sobel filter can be found at " OUTPUT_FILE ", or you can run 'make image' to get the jpg\n");

	return 0;
}

