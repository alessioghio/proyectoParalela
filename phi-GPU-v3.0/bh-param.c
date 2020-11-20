/*
  bh-param.c - Written by BERCZIK PETER
  Last redaction - 3/4/2006 8:04PM
*/

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

#define Pi    3.14159265358979323846
#define G     6.6704E-11
#define R_gas 8.31434E+03
#define k_gas 1.38066E-23
#define Mp    1.6726E-27
#define Msol  1.9891E+30
#define pc    3.085678E+16
#define kpc   (1.0E+03*3.085678E+16)
#define km    1.0E+03
#define cm3   1.0E+06
#define Year  3.1556926E+07

#define KB        1024
#define N_MAX     2

char    name[8], fname[12];

int     i, j, tmp_i, N;

double  eps=1.0E-04;

double  t, m[N_MAX], dt[N_MAX], 
        X[N_MAX],  Y[N_MAX],  Z[N_MAX],
        VX[N_MAX], VY[N_MAX], VZ[N_MAX],
        POT[N_MAX],
        AX[N_MAX], AY[N_MAX], AZ[N_MAX],
        JX[N_MAX], JY[N_MAX], JZ[N_MAX],
        R[3], V[3], L[3], mju, 
	XCM_BH, YCM_BH, ZCM_BH,	
	VXCM_BH, VYCM_BH, VZCM_BH,
        dr, dv, dr2, dv2, dl, dl2,
        a, e, e2, over_a, EB, T_orb,
        tmp;

FILE    *inp, *out, *out_evol;

/***************************************************************/
/***************************************************************/
/***************************************************************/

int main(int argc, char *argv[])
{

/*
if(argc == 2)
  {
  j = atoi(argv[1]);
  }
else
  {
  j = 0;
  exit(argc);
  }
*/

out = fopen("bh-param.dat","w");
fclose(out);


inp = fopen("bh.dat","r");
out = fopen("bh-param.dat","a");


while(feof(inp) == NULL)
  {

  fscanf(inp,"%lE \n", &t);

  fscanf(inp,"%d  %lE  %lE %lE %lE  %lE %lE %lE  %lE  %lE %lE %lE  %lE %lE %lE \n", 
  &tmp_i, &m[0], &X[0], &Y[0], &Z[0], &VX[0], &VY[0], &VZ[0], &POT[0], &AX[0], &AY[0], &AZ[0], &JX[0], &JY[0], &JZ[0]);

  fscanf(inp,"%d  %lE  %lE %lE %lE  %lE %lE %lE  %lE  %lE %lE %lE  %lE %lE %lE \n", 
  &tmp_i, &m[1], &X[1], &Y[1], &Z[1], &VX[1], &VY[1], &VZ[1], &POT[1], &AX[1], &AY[1], &AZ[1], &JX[1], &JY[1], &JZ[1]);

  fscanf(inp,"\n");


  /* BH evolution in the BH's center of mass... */

  XCM_BH = (m[0]*X[0] + m[1]*X[1]) / (m[0] + m[1]);
  YCM_BH = (m[0]*Y[0] + m[1]*Y[1]) / (m[0] + m[1]);
  ZCM_BH = (m[0]*Z[0] + m[1]*Z[1]) / (m[0] + m[1]);

  VXCM_BH = (m[0]*VX[0] + m[1]*VX[1]) / (m[0] + m[1]);
  VYCM_BH = (m[0]*VY[0] + m[1]*VY[1]) / (m[0] + m[1]);
  VZCM_BH = (m[0]*VZ[0] + m[1]*VZ[1]) / (m[0] + m[1]);

/*
  X[0] -= XCM_BH; Y[0] -= YCM_BH; Z[0] -= ZCM_BH;
  X[1] -= XCM_BH; Y[1] -= YCM_BH; Z[1] -= ZCM_BH;

  VX[0] -= VXCM_BH; VY[0] -= VYCM_BH; VZ[0] -= VZCM_BH;
  VX[1] -= VXCM_BH; VY[1] -= VYCM_BH; VZ[1] -= VZCM_BH;
*/


  mju = m[0]*m[1]/(m[0] + m[1]);


  /* relative quantities... */

  R[0] = (X[1] - X[0]);
  R[1] = (Y[1] - Y[0]);
  R[2] = (Z[1] - Z[0]);

  V[0] = (VX[1] - VX[0]);
  V[1] = (VY[1] - VY[0]);
  V[2] = (VZ[1] - VZ[0]);

  L[0] = (R[1]*V[2] - R[2]*V[1]);
  L[1] = (R[2]*V[0] - R[0]*V[2]);
  L[2] = (R[0]*V[1] - R[1]*V[0]);


  dr2 = SQR(R[0]) + SQR(R[1]) + SQR(R[2]) + SQR(eps);
  dr  = sqrt(dr2);

  dv2 = SQR(V[0]) + SQR(V[1]) + SQR(V[2]);
  dv  = sqrt(dv2);

  dl2 = SQR(L[0]) + SQR(L[1]) + SQR(L[2]);
  dl  = sqrt(dl2);

  EB = -(m[0] + m[1]) / dr + 0.5 * dv2;

  a = -0.5 * (m[0] + m[1])/EB;

  over_a = 1.0/a;

  e2 = 1.0 - dl2/(m[0] + m[1])  * over_a;


  if(e2 >= 0.0) 
    e = sqrt(e2);
  else
    e = 0.0;


  if(a >= 0.0) 
    T_orb = 2.0*Pi * sqrt( a*a*a / (m[0] + m[1]) );
  else
    T_orb = 0.0;


  printf("%.8E  % .4E % .4E % .4E % .4E % .4E  %.4E \n", 
  t, dr, a, over_a, EB, e, T_orb);

  fprintf(out,"%.8E \t %.6E \t % .6E % .6E \t % .6E \t % .6E \t %.6E \t % .6E % .6E % .6E \t % .6E % .6E % .6E \n", 
  t, dr, a, over_a, EB, e, T_orb, XCM_BH, YCM_BH, ZCM_BH, VXCM_BH, VYCM_BH, VZCM_BH);


  } /* while(feof(inp) == NULL) */


fclose(out);
fclose(inp);


return(0);

}

/***************************************************************/
/***************************************************************/
/***************************************************************/
