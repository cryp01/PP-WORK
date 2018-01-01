#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef REAL
#define REAL float
#endif

#ifndef M_PI
#define M_PI (3.1415926535897932384626)
#endif

#define F1(x,y,z) F1[((ny)*(nz)*(x))+((nz)*(y))+(z)]
#define F2(x,y,z) F2[((ny)*(nz)*(x))+((nz)*(y))+(z)]

void diffusion (REAL * F1, REAL * F2,
                int nx, int ny, int nz,
                REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
                REAL cb, REAL cc, int time)
{
  int t, x, y, z;
  for (t = 0; t < time; ++t)
  {
    for (x=0; x < nx; x++)
      for (y = 0; y < ny; y++)
        for (z = 0; z < nz;)
        {
          int west, east, north, south, top, down;
          if (x == 0)    west = x;  else west = x - 1;
          if (x == nx-1) east = x;  else east = x + 1;
          if (y == 0)    north= y;  else north= y - 1;
          if (y == ny-1) south= y;  else south= y + 1;
          if (z == 0)    down= 0;  else down= z - 1;
          if (z == nz-1) top=  z;  else top = z + 1;
          F2(x,y,z) =
             F1(x,y,z)     *cc +  F1(west,y,z)  *cw +
             F1(east,y,z)  *ce +  F1(x,north,z) *cn +
             F1(x,south,z) *cs +  F1(x,y,down)  *cb +
             F1(x,y,top)   *ct;
          z++;

          if (x == 0)    west = x;  else west = x - 1;
          if (x == nx-1) east = x;  else east = x + 1;
          if (y == 0)    north= y;  else north= y - 1;
          if (y == ny-1) south= y;  else south= y + 1;
          if (z == 0)    down= 0;  else down= z - 1;
          if (z == nz-1) top=  z;  else top = z + 1;
          F2(x,y,z) =
             F1(x,y,z)     *cc +  F1(west,y,z)  *cw +
             F1(east,y,z)  *ce +  F1(x,north,z) *cn +
             F1(x,south,z) *cs +  F1(x,y,down)  *cb +
             F1(x,y,top)   *ct;
         z++;
         }
    REAL *tt = F1;   F1 = F2;  F2 = tt;  // swap matrices
  }
}

void init (REAL *F1, const int nx, const int ny, const int nz,
           const REAL kx, const REAL ky, const REAL kz,
           const REAL dx, const REAL dy, const REAL dz,
           const REAL kappa, const REAL time)
{
  REAL ax, ay, az;
  int jx, jy, jz;
  ax = exp(-kappa*time*(kx*kx));
  ay = exp(-kappa*time*(ky*ky));
  az = exp(-kappa*time*(kz*kz));
  for (jx = 0; jx < nx; jx++)
    for (jy = 0; jy < ny; jy++)
      for (jz = 0; jz < nz; jz++)
      {
        REAL x = dx*((REAL)(jx + 0.5));
        REAL y = dy*((REAL)(jy + 0.5));
        REAL z = dz*((REAL)(jz + 0.5));
        REAL f0 = (REAL)0.125
          *(1.0 - ax*cos(kx*x))
          *(1.0 - ay*cos(ky*y))
          *(1.0 - az*cos(kz*z));
        F1(jx,jy,jz) = f0;
      }
}

REAL sum_values (REAL *F1, const int nx, const int ny, const int nz)
{
  REAL sum=0.0;
  int jx, jy, jz;
  for (jx = 0; jx < nx; jx++)
    for (jy = 0; jy < ny; jy++)
      for (jz = 0; jz < nz; jz++)
        sum += F1(jx,jy,jz);
  return sum;
}

int main(int argc, char *argv[])
{
  int  jx, jy, jz;
  int  NX=128, NY=128, NZ=128;

  if (argc>1) { NX= atoi(argv[1]); } // get  first command line parameter
  if (argc>2) { NY= atoi(argv[2]); } // get second command line parameter
  if (argc>3) { NZ= atoi(argv[3]); } // get  third command line parameter
  if (NX < 1 || NY < 1 || NZ < 1)
  {
    printf("arguments: NX NY NZ\n");
    return 1;
  }

  REAL *f1 = (REAL *) malloc(sizeof(REAL)*NX*NY*NZ);
  REAL *f2 = (REAL *) malloc(sizeof(REAL)*NX*NY*NZ);

  REAL *f_final = NULL;

  REAL  time  = 0.0;
  int   count = 0;

  REAL l, dx, dy, dz, kx, ky, kz, kappa, dt;
  REAL ce, cw, cn, cs, ct, cb, cc;

  l = 1.0;
  kappa = 0.1;
  dx = l / NX;  dy = l / NY;  dz = l / NZ;
  kx = ky = kz = 2.0 * M_PI;
  dt = dx*dy*dz / kappa;
  count = 0.01 / dt;
  f_final = (count % 2)? f2 : f1;

  init(f1, NX, NY, NZ, kx, ky, kz, dx, dy, dz, kappa, time);

  REAL err = sum_values(f1, NX, NY, NZ);

  ce = cw = kappa*dt/(dx*dx);
  cn = cs = kappa*dt/(dy*dy);
  ct = cb = kappa*dt/(dz*dz);
  cc = 1.0 - (ce + cw + cn + cs + ct + cb);

  printf("Running diffusion kernel with NX=%d, NY=%d, NZ=%d, %d times\n",
         NX, NY, NZ, count);

  diffusion(f1, f2, NX, NY, NZ, ce, cw, cn, cs, ct, cb, cc, count);

  err = err - sum_values(f_final, NX, NY, NZ);
  fprintf(stderr, "Accuracy     : %E\n", err);

  free(f1); free(f2);
  return 0;
}