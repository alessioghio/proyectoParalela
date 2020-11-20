/*
  bh-evol.c - Written by BERCZIK PETER
  Last redaction - 7/27/2005 12:37PM
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define ABS(x)   (((x)<0)?(-x):(x))

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
#define N_MAX     1000000

char    name[8], fname[12];

int     i, j, tmp_i, N;

double  t, m[N_MAX], 
        X[N_MAX],  Y[N_MAX],  Z[N_MAX],
        VX[N_MAX], VY[N_MAX], VZ[N_MAX], 
	XCM_BH, YCM_BH, ZCM_BH,	VXCM_BH, VYCM_BH, VZCM_BH,
        v1, v2, dr, over_a, over_a_0 = 0.0, d_over_a, eps = 1.0E-04,
        hx, hy, hz, h2, EB, e, 
	MTCM, XTCM, YTCM, ZTCM, 
        MLCM, XLCM, YLCM, ZLCM, 
	XDC, YDC, ZDC, 
	X_BHC, Y_BHC, Z_BHC, 
	XBH0, YBH0, ZBH0, XBH1, YBH1, ZBH1,
        XBH_TCM, YBH_TCM, ZBH_TCM, XBH_DC, YBH_DC, ZBH_DC,         
        tmp;

FILE    *inp, *out, *out_evol;

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

/*
out = fopen("bh-evol.dat","w");
fclose(out);

out_evol = fopen("def-den-evol.dat","w");
fclose(out_evol);

out_evol = fopen("BH-Brown.dat","w");
fclose(out_evol);

out = fopen("bh-coord.dat","w");
fclose(out);
*/

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
    for(i=0;i<2;i++) fscanf(inp,"%d %lE %lE %lE %lE %lE %lE %lE \n", &tmp_i, &m[i], &X[i], &Y[i], &Z[i], &VX[i], &VY[i], &VZ[i]);

    fclose(inp);


    /* BH evolution in the BH's center of mass... */

    XCM_BH = (m[0]*X[0] + m[1]*X[1]) / (m[0] + m[1]);
    YCM_BH = (m[0]*Y[0] + m[1]*Y[1]) / (m[0] + m[1]);
    ZCM_BH = (m[0]*Z[0] + m[1]*Z[1]) / (m[0] + m[1]);

    VXCM_BH = (m[0]*VX[0] + m[1]*VX[1]) / (m[0] + m[1]);
    VYCM_BH = (m[0]*VY[0] + m[1]*VY[1]) / (m[0] + m[1]);
    VZCM_BH = (m[0]*VZ[0] + m[1]*VZ[1]) / (m[0] + m[1]);

    X[0] -= XCM_BH; Y[0] -= YCM_BH; Z[0] -= ZCM_BH;
    X[1] -= XCM_BH; Y[1] -= YCM_BH; Z[1] -= ZCM_BH;

    VX[0] -= VXCM_BH; VY[0] -= VYCM_BH; VZ[0] -= VZCM_BH;
    VX[1] -= VXCM_BH; VY[1] -= VYCM_BH; VZ[1] -= VZCM_BH;

    dr = (X[0] - X[1])*(X[0] - X[1]) + 
         (Y[0] - Y[1])*(Y[0] - Y[1]) + 
         (Z[0] - Z[1])*(Z[0] - Z[1]);

    dr = sqrt(dr);

    v1 = VX[0]*VX[0] + VY[0]*VY[0] + VZ[0]*VZ[0];
    v2 = VX[1]*VX[1] + VY[1]*VY[1] + VZ[1]*VZ[1];

    over_a = 2.0/sqrt(dr*dr + eps*eps) - v1/m[1] - v2/m[0];

    d_over_a = over_a - over_a_0;

    EB = -m[0]*0.5 * over_a;

    hx = (Y[0]*VZ[0] - Z[0]*VY[0]) + (Y[1]*VZ[1] - Z[1]*VY[1]);
    hy = (Z[0]*VX[0] - X[0]*VZ[0]) + (Z[1]*VX[1] - X[1]*VZ[1]);
    hz = (X[0]*VY[0] - Y[0]*VX[0]) + (X[1]*VY[1] - Y[1]*VX[1]);

    h2 = hx*hx + hy*hy + hz*hz;

    e = 1.0 - 2.0*h2/m[0] * over_a;

    if(e >= 0.0) 
      e = sqrt(e);
    else
      e = 0.0;
       
    printf("%04d \t %.4E \t %.6E \t % .6E \n", j, t, dr, over_a);

/*
    out = fopen("bh-evol.dat","a");    
    fprintf(out,"%.6E \t %.6E \t % .6E \t % .6E \t % .6E \t %04d \n", t, dr, over_a, EB, e, j);
    fclose(out);
*/
    over_a_0 = over_a;






    /* read again all particles */

    inp = fopen(fname,"r");

    fscanf(inp,"%d \n %d \n %lE \n", &tmp_i, &tmp_i, &t);
    for(i=0;i<N;i++) fscanf(inp,"%d %lE %lE %lE %lE %lE %lE %lE \n", &tmp_i, &m[i], &X[i], &Y[i], &Z[i], &VX[i], &VY[i], &VZ[i]);

    fclose(inp);
    

    /* Def the BH's center in the IC */

    X_BHC = (m[0]*X[0] + m[1]*X[1]) / (m[0] + m[1]);
    Y_BHC = (m[0]*Y[0] + m[1]*Y[1]) / (m[0] + m[1]);
    Z_BHC = (m[0]*Z[0] + m[1]*Z[1]) / (m[0] + m[1]);
    

    /* Setting the BH's masses to 0.0 */

    m[0] = 0.0;
    m[1] = 0.0;



    /* DEF TOTAL CM for all particles together */

    XTCM = 0.0;  YTCM = 0.0;  ZTCM = 0.0;  MTCM = 0.0;

    for(i=0;i<N;i++)
      {
      tmp = sqrt(X[i]*X[i] + Y[i]*Y[i] + Z[i]*Z[i]);

      XTCM += m[i]*X[i]; YTCM += m[i]*Y[i]; ZTCM += m[i]*Z[i]; MTCM += m[i];
      }

    XTCM /= MTCM;  YTCM /= MTCM;  ZTCM /= MTCM;


    /* moving the coordinates to the system connected with the TOTAL CM */
/*
    for(i=0;i<N;i++)
      {
      X[i] -= XTCM; Y[i] -= YTCM; Z[i] -= ZTCM;
      }
*/



    /* Define the Density Center (DC) coordinates */

    XDC = 0.0; YDC = 0.0; ZDC = 0.0;


    /* DEF LCM inside 100 */

    XLCM = 0.0;  YLCM = 0.0;  ZLCM = 0.0;  MLCM = 0.0;

    for(i=0;i<N;i++)
      {
      tmp = sqrt(X[i]*X[i] + Y[i]*Y[i] + Z[i]*Z[i]);

      if(tmp < 100.0) 
        {
        XLCM += m[i]*X[i]; YLCM += m[i]*Y[i]; ZLCM += m[i]*Z[i]; MLCM += m[i];
        }
      }

    XLCM /= MLCM;  YLCM /= MLCM;  ZLCM /= MLCM;

    for(i=0;i<N;i++)
      {
      X[i] -= XLCM; Y[i] -= YLCM; Z[i] -= ZLCM;
      }

    XDC += XLCM; YDC += YLCM; ZDC += ZLCM; 



    /* DEF LCM inside 10 */

    XLCM = 0.0;  YLCM = 0.0;  ZLCM = 0.0;  MLCM = 0.0;

    for(i=0;i<N;i++)
      {
      tmp = sqrt(X[i]*X[i] + Y[i]*Y[i] + Z[i]*Z[i]);

      if(tmp < 10.0)
        {
        XLCM += m[i]*X[i]; YLCM += m[i]*Y[i]; ZLCM += m[i]*Z[i]; MLCM += m[i];
        }
      }

    XLCM /= MLCM;  YLCM /= MLCM;  ZLCM /= MLCM;

    for(i=0;i<N;i++)
      {
      X[i] -= XLCM; Y[i] -= YLCM; Z[i] -= ZLCM;
      }

    XDC += XLCM; YDC += YLCM; ZDC += ZLCM; 



    /* DEF LCM inside 5 */

    XLCM = 0.0;  YLCM = 0.0;  ZLCM = 0.0;  MLCM = 0.0;

    for(i=0;i<N;i++)
      {
      tmp = sqrt(X[i]*X[i] + Y[i]*Y[i] + Z[i]*Z[i]);

      if(tmp < 5.0)
        {
        XLCM += m[i]*X[i]; YLCM += m[i]*Y[i]; ZLCM += m[i]*Z[i]; MLCM += m[i];
        }
      }

    XLCM /= MLCM;  YLCM /= MLCM;  ZLCM /= MLCM;

    for(i=0;i<N;i++)
      {
      X[i] -= XLCM; Y[i] -= YLCM; Z[i] -= ZLCM;
      }

    XDC += XLCM; YDC += YLCM; ZDC += ZLCM;



    /* DEF LCM inside 1 */

    XLCM = 0.0;  YLCM = 0.0;  ZLCM = 0.0;  MLCM = 0.0;

    for(i=0;i<N;i++)
      {
      tmp = sqrt(X[i]*X[i] + Y[i]*Y[i] + Z[i]*Z[i]);

      if(tmp < 1.0) 
        {
        XLCM += m[i]*X[i]; YLCM += m[i]*Y[i]; ZLCM += m[i]*Z[i]; MLCM += m[i];
        }
      }

    XLCM /= MLCM;  YLCM /= MLCM;  ZLCM /= MLCM;

    for(i=0;i<N;i++)
      {
      X[i] -= XLCM; Y[i] -= YLCM; Z[i] -= ZLCM;
      }

    XDC += XLCM; YDC += YLCM; ZDC += ZLCM; 



    /* DEF LCM inside 0.5 */

    XLCM = 0.0;  YLCM = 0.0;  ZLCM = 0.0;  MLCM = 0.0;

    for(i=0;i<N;i++)
      {
      tmp = sqrt(X[i]*X[i] + Y[i]*Y[i] + Z[i]*Z[i]);
      
      if(tmp < 0.5)
        {
        XLCM += m[i]*X[i]; YLCM += m[i]*Y[i]; ZLCM += m[i]*Z[i]; MLCM += m[i];
        }
      }

    XLCM /= MLCM;  YLCM /= MLCM;  ZLCM /= MLCM;

    for(i=0;i<N;i++)
      {
      X[i] -= XLCM; Y[i] -= YLCM; Z[i] -= ZLCM;
      }

    XDC += XLCM; YDC += YLCM; ZDC += ZLCM;



    /* DEF LCM inside 0.1 */

    XLCM = 0.0;  YLCM = 0.0;  ZLCM = 0.0;  MLCM = 0.0;

    for(i=0;i<N;i++)
      {
      tmp = sqrt(X[i]*X[i] + Y[i]*Y[i] + Z[i]*Z[i]);
      
      if(tmp < 0.1)
        {
        XLCM += m[i]*X[i]; YLCM += m[i]*Y[i]; ZLCM += m[i]*Z[i]; MLCM += m[i];
        }
      }

    XLCM /= MLCM;  YLCM /= MLCM;  ZLCM /= MLCM;

    for(i=0;i<N;i++)
      {
      X[i] -= XLCM; Y[i] -= YLCM; Z[i] -= ZLCM;
      }

    XDC += XLCM; YDC += YLCM; ZDC += ZLCM;




/*
    out_evol = fopen("def-den-evol.dat","a");
    fprintf(out_evol,"%.6E \t % .6E % .6E % .6E \n",
                      t, XDC, YDC, ZDC);
    fclose(out_evol);


    out_evol = fopen("BH-Brown.dat","a");
    fprintf(out_evol,"%.6E \t % .6E % .6E % .6E \t % .6E % .6E % .6E \n",
                      t,
                      X_BHC, Y_BHC, Z_BHC,
                      X_BHC-XDC, Y_BHC-YDC, Z_BHC-ZDC);
    fclose(out_evol);
*/

    out = fopen("bh-coord.dat","a");
    fprintf(out,"%.6E \t % .6E % .6E % .6E \t % .6E % .6E % .6E \t % .6E % .6E % .6E \n",
                  t,
                  X_BHC, Y_BHC, Z_BHC,
                  XDC, YDC, ZDC,
                  X_BHC-XDC, Y_BHC-YDC, Z_BHC-ZDC);
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
