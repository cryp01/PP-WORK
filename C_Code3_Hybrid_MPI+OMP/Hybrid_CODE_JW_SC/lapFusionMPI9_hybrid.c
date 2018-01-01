
// Libraries
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <omp.h> 


float stencil ( float v1, float v2, float v3, float v4)
{
  return (v1 + v2 + v3 + v4) * 0.25f;
}

float max_error ( float prev_error, float old, float new )
{
  float t= fabsf( new - old );
  return t>prev_error? t: prev_error;
}

/*Initialisation of the grid - this sets the internal points set to 0
  and boundary conditions*/
void laplace_init(float *in, int n)
{
  int i;
  const float pi  = 2.0f * asinf(1.0f);
  memset(in, 0, n*n*sizeof(float));
  for (i=0; i<n; i++) {
    float V = in[i*n] = sinf(pi*i / (n-1));
    in[ i*n+n-1 ] = V*expf(-pi);
  }
}

float laplace_MPI_step(float *in, float *out, int n, int ri, int rf)
{
  int i, j;
  float my_error=0.0f;
  #pragma omp for
  for ( j=ri; j < rf; j++ )
  #pragma omp simd reduction(max:my_error)
    for ( i=1; i < n-1; i++ )
    {
      out[j*n+i]= stencil(in[j*n+i+1], in[j*n+i-1], in[(j-1)*n+i], in[(j+1)*n+i]);
      my_error = max_error( my_error, out[j*n+i], in[j*n+i] );
    }
  return my_error;
}

void copy_A_to_temp(float *A, float * *temp, int nrows, int n){
    for(int j = 0; j < nrows; j++){
      for(int i = 0; i < n; i++){
        (*temp)[j*n +i] = A[j*n +i];
      }
    }
}

int main(int argc, char** argv)
{   

  int n = 4096; 
  int iter_max = 1000; 
  float *A, *temp; 
    
  const float tol = 1.0e-5f; 
  float error = 1.0f;   

  int nprocs, rank, jwsc, my_nrows, nrows, tag = 1, my_size; 
  float *my_A, *my_temp; 
  float my_error= 1.0f; 
  
  MPI_Status status; 
  int ri, rf, provided;
  
 /* HYBRID Activated using...>>> "MPI_Init_thread (&argc, &argv, MPI_THREAD_FUNNELED, &provided);"
 and not .... "MPI_Init (&argc, &argv);" */ 
 /* MPI Level of Thread Safety - "MPI_THREAD_FUNNELED": multithreaded, but only the main thread 
     makes MPI calls (the one that called MPI_Init_thread)*/
 
  jwsc = MPI_Init_thread (&argc, &argv, MPI_THREAD_FUNNELED, &provided);  ;
  if (jwsc != MPI_SUCCESS)
    {
      printf ("Error starting MPI program. Terminating.\n");
      MPI_Abort (MPI_COMM_WORLD, jwsc);
      return -1;
    }
  double T0, T1;
  T0 = MPI_Wtime();
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
   
  if(nprocs < 2){
    printf ("This program works with 2 or more processes (-np 'Enter N >= 2').\n");
    MPI_Abort (MPI_COMM_WORLD, 1);
    return -1;
  } 

  // get runtime arguments 
  if (argc>1) {  n        = atoi(argv[1]); }
  if (argc>2) {  iter_max = atoi(argv[2]); }
  
  // Allocate memory for A and temp
  if( ( A = (float*) malloc(n*n*sizeof(float)) ) == NULL ){
    printf ("Error when allocating memory for A.\n");
    MPI_Abort (MPI_COMM_WORLD, 1);
    return -1;
  }
  if( ( temp = (float*) malloc(n*n*sizeof(float)) ) == NULL ){
    printf ("Error when allocating memory for temp.\n");
    MPI_Abort (MPI_COMM_WORLD, 1);
    return -1;
  }
  
  MPI_Bcast(&error, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

  my_nrows = n/nprocs; 
  my_size = n*(my_nrows+2);
  
// Allocate memory for my_A and my_temp
   if( ( my_A = (float*) malloc( my_size*sizeof(float)) ) == NULL ){
    printf ("Error when allocating memory for my_A.\n");
    MPI_Abort (MPI_COMM_WORLD, 1);
    return -1;
  }
  if( ( my_temp = (float*) malloc(my_size*sizeof(float)) ) == NULL ){
    printf ("Error when allocating memory for my_temp.\n");
    MPI_Abort (MPI_COMM_WORLD, 1);
    return -1;
  }
  
  // set boundary conditions
  laplace_init (A, n);
  laplace_init (temp, n);
  A[(n/128)*n+n/128] = 1.0f; 
  printf("Jacobi relaxation Calculation: %d x %d mesh,"
         " maximum of %d iterations\n", 
         n, n, iter_max );
 
  MPI_Scatter(A, my_nrows*n,  MPI_FLOAT, my_A+n, my_nrows*n, MPI_FLOAT, 0, MPI_COMM_WORLD);
  
  float * my_temp_n = my_temp+n;
  copy_A_to_temp(my_A+n, &my_temp_n, my_nrows, n);
  
 int iter = 0;  
 nrows = my_nrows +2;
  
 while ( error > tol*tol && iter < iter_max ) {
	iter++;
	
/* All MPI calls are made by the master thread
   Outside the OpenMP parallel regions and
    In OpenMP master regions */
	
/* omp begin parallel */

 #pragma omp parallel
 {
	
    if(rank > 0){
	#pragma omp barrier
    #pragma omp master
	{
      MPI_Send(my_A+n, n, MPI_FLOAT, rank-1, tag, MPI_COMM_WORLD);
      MPI_Recv(my_A, n, MPI_FLOAT, rank-1, tag, MPI_COMM_WORLD, &status);
	  /*printf( " [%d] received(%d) from %d!\n", rank, nprocs, status.MPI_SOURCE );*/
	}
	#pragma omp barrier
	  }
 

    if(rank < nprocs -1 ){
	#pragma omp barrier
    #pragma omp master
	{
       MPI_Recv( (my_A+n*(my_nrows+1) ), n, MPI_FLOAT, rank+1, tag, MPI_COMM_WORLD, &status);
	   /*printf( " [%d] received(%d) from %d!\n", rank, nprocs, status.MPI_SOURCE );*/
	   MPI_Send(  (my_A+ n*(my_nrows)), n, MPI_FLOAT, rank+1, tag, MPI_COMM_WORLD);
	}
	#pragma omp barrier
      } 
	   
     
    if(rank == 0){
      ri =2;
      rf = nrows-1;
    }
    else if(rank == (nprocs - 1)){
      ri = 1;
      rf = nrows -2;
    }
    else{ 
      ri = 1;
      rf = nrows-1;   
	}
 } 
  /* omp end parallel */
  
  my_error= laplace_MPI_step(my_A, my_temp, n, ri, rf); 
  MPI_Reduce(&my_error, &error, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD); 
 
  float *swap= my_A; my_A=my_temp; my_temp= swap; // swap pointers my_A & my_temp 
   
 } 
 
  MPI_Gather(my_A+n, my_nrows*n, MPI_FLOAT, A, my_nrows*n, MPI_FLOAT, 0, MPI_COMM_WORLD);
  
    if(rank == 0){
    error = sqrtf( error );
    printf("Total Iterations: %5d, ERROR: %0.6f, ", iter, error);
    printf("A[%d][%d]= %0.6f\n", n/128, n/128, A[(n/128)*n+n/128]);
    }
	
  // recieve time details
   if(rank == 0){
   T1 = MPI_Wtime();
   printf( " [%d]  \t\ttime: %lf\n", rank, T1 - T0 );
   }
 
 
 free(A); free(temp); free(my_A); free(my_temp);
 MPI_Finalize();
 return 0;

}
