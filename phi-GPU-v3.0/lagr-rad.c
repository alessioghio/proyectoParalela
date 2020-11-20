/***************************************************************************** 
  File Name      : "lagr-rad.c"
  Contents       : Lagrange radius calc. using Quicksort vs. Select procedures...
  Coded by       : Peter Berczik
  Last redaction : 1/31/2005 2:21PM
*****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Some "good" functions and constants... */
#define SIG(x)   ( ((x)<0) ? (-1):(1) )
#define ABS(x)   ( ((x)<0) ? (-x):(x) )

#define MAX(a,b) ( ((a)>(b)) ? (a):(b) )
#define MIN(a,b) ( ((a)<(b)) ? (a):(b) )

#define SQR(x)   ( (x)*(x) )

#define Pi      3.141592653589793238462643
#define G     	6.6704E-11
#define R_gas 	8.31434E+03
#define k_gas 	1.38066E-23
#define Mp    	1.6726E-27
#define Msol  	1.9891E+30
#define pc    	3.085678E+16
#define kpc   	(1.0E+03*3.085678E+16)
#define km    	1.0E+03
#define cm3   	1.0E+06
#define Year  	3.1556926E+07
#define Myr  	(1.0E+06*3.1556926E+07)

#define KB       1024
#define MB       1048576
#define N_MAX    20*KB

/***************************************************************/

char    name[8], fname[12], tmp_char[12];

int     i, i, j, tmp_i, N, ii,
        ind[N_MAX];

double  t, 
        mass_frac, 
	lagr_rad1, lagr_rad2, lagr_rad3, 
        lagr_rad4, lagr_rad5, lagr_rad6,
        lagr_rad7, lagr_rad8, lagr_rad9,
        m[N_MAX], d[N_MAX],
        X[N_MAX],  Y[N_MAX],  Z[N_MAX],
        VX[N_MAX], VY[N_MAX], VZ[N_MAX], 
	MCM,
	XCM, YCM, ZCM,
	VXCM, VYCM, VZCM,
	X_CM, Y_CM, Z_CM,
        tmp;
	
double R_LIM, R_Lim[] = { 10.0, 5.0, 1.0, 0.5, 0.1 }; 

FILE    *inp, *out;

/***************************************************************/

/***************************************************************/
void d_swap(double *a, double *b)
{
register double tmp;
tmp = *a; *a = *b; *b = tmp;
}

void i_swap(int *a, int *b)
{
register int tmp;
tmp = *a; *a = *b; *b = tmp;
}
/***************************************************************/

/***************************************************************/
void my_sort(int l, int r, double *arr, int *ind)
{

int    i, j, cikl;
double tmp;

i = l; j = r;
tmp = arr[(l+r)/2];

cikl = 1;

while (cikl)
  {
    while (arr[i]<tmp) i++;
    while (tmp<arr[j]) j--;

    if (i<=j)
     {
	d_swap(&arr[i],&arr[j]);
	i_swap(&ind[i],&ind[j]);
	i++; j--;
     }
    else
     {
	cikl = 0;
     }

  }
				      
  if (l<j) my_sort(l, j, arr, ind);
  if (i<r) my_sort(i, r, arr, ind);				      
}
/***************************************************************/

/***************************************************************/
double my_select(int n_1, int n_2, int k, double *arr, int *ind)
{

int   i, ir, j, l, mid, a_ind;
double a;

l  = n_1;
ir = n_2;

for(;;)
  {

  if (ir <= l+1) 
    {

    if (ir == l+1 && arr[ir] < arr[l]) 
      {
      d_swap(&arr[l],&arr[ir]);
      i_swap(&ind[l],&ind[ir]);
      }

    return(arr[k]);

    } 
    else 
    {

    mid=(l+ir) >> 1;

    d_swap(&arr[mid],&arr[l+1]);
    i_swap(&ind[mid],&ind[l+1]);

    if (arr[l+1] > arr[ir]) 
      {
      d_swap(&arr[l+1],&arr[ir]);
      i_swap(&ind[l+1],&ind[ir]);
      }

    if (arr[l] > arr[ir]) 
      {
      d_swap(&arr[l],&arr[ir]);
      i_swap(&ind[l],&ind[ir]);
      }

    if (arr[l+1] > arr[l])
      {
      d_swap(&arr[l+1],&arr[l]);
      i_swap(&ind[l+1],&ind[l]);
      }

    i = l+1;
    j = ir;
    a = arr[l];
    a_ind = ind[l];

    for (;;) 
     {
     do i++; while (arr[i] < a);
     do j--; while (arr[j] > a);
     if (j < i) break;
     d_swap(&arr[i],&arr[j]);
     i_swap(&ind[i],&ind[j]);
     }

    arr[l] = arr[j];
    ind[l] = ind[j];
    arr[j] = a;
    ind[j] = a_ind;
    if (j >= k) ir = j-1;
    if (j <= k) l = i;

    }
  }
}
/***************************************************************/

/***************************************************************/
double lagrange_radius(double percent)
{ 
int    Nb;
double tmp;

Nb = (int)(percent * N);

//my_sort(0, N-1, d, ind);
   
tmp = my_select(0, N-1, Nb-1, d, ind); d[Nb-1] = tmp;
		
return(tmp);
}
/***************************************************************/

/***************************************************************/
/***************************************************************/
/***************************************************************/

int main(int argc, char *argv[])
{

if(argc == 2)
  {
  j = atoi(argv[1]);
  }
else
  {
  j = 0;
//  exit(argc);
  }


if(j==0)
  {
  out = fopen("lagr-rad.dat","w");
  fclose(out);
  }  


sprintf(name,"%04d",j);

strcpy(fname,name);
strcat(fname,".dat");
inp = fopen(fname,"r");


fscanf(inp,"%d \n %d \n %lE \n", &tmp_i, &N, &t);

fclose(inp);


while(inp != NULL)
  {

  sprintf(name,"%04d",j);

  strcpy(fname,name);
  strcat(fname,".dat");
  inp = fopen(fname,"r");

  if(inp != NULL)
    {

    fscanf(inp,"%d \n %d \n %lE \n", &tmp_i, &tmp_i, &t);
//    for(i=0;i<N;i++) fscanf(inp,"%d %lE %lE %lE %lE %lE %lE %lE %lE %lE %lE %d \n", &ind[i], &m[i], &X[i], &Y[i], &Z[i], &VX[i], &VY[i], &VZ[i], &tmp, &tmp, &tmp, &tmp_i);
    for(i=0;i<N;i++) fscanf(inp,"%d %lE %lE %lE %lE %lE %lE %lE \n", &tmp_i, &m[i], &X[i], &Y[i], &Z[i], &VX[i], &VY[i], &VZ[i]);

    fclose(inp);



    /* DEF CM for the particles inside R_LIM */
    
    X_CM = Y_CM = Z_CM = 0.0;
    
    /* Do sizeof(R_Lim)/sizeof(R_Lim[0]) iterations for R_LIM */

    for(ii=0; ii < sizeof(R_Lim)/sizeof(R_Lim[0]); ii++)
      {

      R_LIM = R_Lim[ii]; 

      MCM = 0.0;
      XCM  = YCM  = ZCM  = 0.0;
      VXCM = VYCM = VZCM = 0.0;

      for(i=0;i<N;i++)
        {

        tmp = sqrt(X[i]*X[i] + Y[i]*Y[i] + Z[i]*Z[i]);

        if(tmp < R_LIM)
          {
          MCM += m[i];
        
          XCM += m[i]*X[i]; 
          YCM += m[i]*Y[i]; 
          ZCM += m[i]*Z[i]; 

          VXCM += m[i]*VX[i]; 
          VYCM += m[i]*VY[i]; 
          VZCM += m[i]*VZ[i]; 
          } /* if(tmp < R_lim) */

        } /* i */


      XCM  /= MCM; YCM  /= MCM; ZCM  /= MCM;
      VXCM /= MCM; VYCM /= MCM; VZCM /= MCM;


      /* Correct the particles positions */

      for(i=0;i<N;i++)
        {
        X[i] -= XCM; Y[i] -= YCM; Z[i] -= ZCM;
        VX[i] -= VXCM; VY[i] -= VYCM; VZ[i] -= VZCM;
        }

      X_CM += XCM; 
      Y_CM += YCM;
      Z_CM += ZCM;
      
//      printf("%d %.6E \t % .6E % .6E % .6E \n", ii, R_LIM, X_CM, Y_CM, Z_CM);
//      fflush(stdout);
      
      } /* Do sizeof(R_Lim)/sizeof(R_Lim[0]) iterations for R_LIM */


    for(i=0; i<N; i++)
      {
      d[i] = sqrt(SQR(X[i]) + SQR(Y[i]) + SQR(Z[i]));    
      } /* i */

    out = fopen("lagr-rad.dat","a");

//    printf("\n");



//my_sort(0, N-1, d, ind); 

//Nb = (int)(mass_frac * N);
//lagr_rad1 = d[Nb-1];


    mass_frac = 0.005;
    lagr_rad1 = lagrange_radius(mass_frac);

    mass_frac = 0.01;
    lagr_rad2 = lagrange_radius(mass_frac);

    mass_frac = 0.02;
    lagr_rad3 = lagrange_radius(mass_frac);

    mass_frac = 0.05;
    lagr_rad4 = lagrange_radius(mass_frac);

    mass_frac = 0.10;
    lagr_rad5 = lagrange_radius(mass_frac);

    mass_frac = 0.25;
    lagr_rad6 = lagrange_radius(mass_frac);

    mass_frac = 0.50;
    lagr_rad7 = lagrange_radius(mass_frac);

    mass_frac = 0.75;
    lagr_rad8 = lagrange_radius(mass_frac);

    mass_frac = 0.90;
    lagr_rad9 = lagrange_radius(mass_frac);

//    mass_frac = 0.99;
//    lagr_rad9 = lagrange_radius(mass_frac);


    printf("%.3E \t r_lagr = %.4f  %.4f  %.4f  %.4f  %.4f \n", t, lagr_rad1, lagr_rad2, lagr_rad3, lagr_rad4, lagr_rad5);
    fprintf(out,"%.6E \t %.6E \t %.6E \t %.6E \t %.6E \t %.6E \t %.6E \t %.6E \t %.6E \t %.6E \n", 
    t, lagr_rad1, lagr_rad2, lagr_rad3, lagr_rad4, lagr_rad5, lagr_rad6, lagr_rad7, lagr_rad8, lagr_rad9);

    fclose(out);

    } /* if(inp != NULL) */

  j = j+1;

//  printf(".");  fflush(stdout);

  } /* while(inp != NULL) */

return(0);

}

/***************************************************************/
/***************************************************************/
/***************************************************************/
