#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <time.h>
#include <string.h>
#include <list>
#include <thrust/device_vector.h>
#include "CudaCoreFunctions.h"

void GetCudaInfo(){
  int nDevices;

  cudaError_t cudaStat1 ;
  cudaStat1= cudaGetDeviceCount(&nDevices);
  if(cudaStat1 == cudaErrorInsufficientDriver){
      printf("Insufficent Driver\n");
    }
  else if(cudaStat1 == cudaErrorNoDevice){
      printf("No Device\n");
    }
  else{
      printf("Cuda Installation Detected:\n");
    }

  std::cout<<"# Devices: "<<nDevices<<std::endl;

  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("_____________________________________\n");
    printf("Device Number: %d\n", i);
    printf("Device name: %s\n", prop.name);
    printf("MaxThreats name: %d\n", prop.maxThreadsPerBlock);
    printf("Major: %d\n",prop.major);
    printf("Minor: %d\n",prop.minor);
    }
}

// __device__
// double D_CorrFuncIntegrand(double x, IntegrandParameter Parameter){
//   double r = Parameter.r;
//   int n   = Parameter.n;
//   double Zeta = Parameter.zeta;
//   double Eta = Parameter.eta;
//   double q1 = Parameter.q1;
//   double IntegrandVal = 0;
//   double t = r/Zeta;
//   IntegrandVal = 2.0*Eta/pow(q1,2) * ( 1.0- (j0( (sqrt(2.0*x)*t) )   * pow( sqrt(1+pow(x,2)) - x , 2*n)) ) / (x*sqrt(1+pow(x,2)));//
//   return( IntegrandVal );
// }

// __device__
// double D_Exp1(double x, IntegrandParameter Parameter){
//   double r = Parameter.r;
//   int n   = Parameter.n;
//   double Zeta = Parameter.zeta;
//   double Eta = Parameter.eta;
//   double q1 = Parameter.q1;
//   double IntegrandVal = 0;
//   double t = r/Zeta;
//   IntegrandVal = exp(-x)/x;
//   return( IntegrandVal );
// }


// __device__
// double D_SimpsonRule(Integrand_t Integrand, IntegrandParameter Parameter, double aa, double a, double b, double NumIntervals)
// {
//   // float (*Integrand)(float, IntegrandParameter);
//   double y=0;
//   double xstep = (b-a)/NumIntervals;
//   for(double x=a;x<b;x=x+xstep){
//      y += xstep/2*((*Integrand)((1-x)/x,Parameter)/pow(x,2)+(*Integrand)((1-(x+xstep))/(x+xstep),Parameter)/pow((x+xstep),2));
//      // y += xstep/6*( (*Integrand)(aa+(1-x)/x,Parameter)/pow(x,2) + 4*(*Integrand)(aa+(1-(x+xstep/2))/(x+xstep/2),Parameter)/pow((x+xstep/2),2) + (*Integrand)(aa+(1-(x+xstep))/(x+xstep),Parameter)/pow((x+xstep),2) ); // Simpson Rule
//   }
//   return(y);
// }

__device__
double D_SimpsonRule2(Integrand_t Integrand, HrIntegrandParameter Parameter, double aa, double a, double b, double NumIntervals)
{
  double y=0;
  double xstep = (b-a)/NumIntervals;
  for(double x=a;x<b;x=x+xstep){
     y += xstep/2*((*Integrand)(x,Parameter)+(*Integrand)((x+xstep),Parameter));
  }
  return(y);
}

__device__
double D_Trapz(HankelIntegrand_t Integrand, HankelParameter Parameter, double a, double b, double xstep)
{
  double y=0;
  for(double x=a;x<b;x=x+xstep){
     y += xstep/2*((*Integrand)(x,Parameter)+(*Integrand)((x+xstep),Parameter));
  }
  return(y);
}

// __device__
// double D_QAGS_Integration(Integrand_t Integrand, IntegrandParameter Parameter)
// {
//   // float (*Integrand)(float, IntegrandParameter);
//   double y=0;
//   double xstep = .001;
//   y = D_SimpsonRule(Integrand,Parameter, xstep,1-xstep,998);
//   // double xstep = .001;
//   // for(double x=xstep;x<1-xstep;x=x+xstep){
//   //    // y += xstep/2*((*Integrand)((1-x)/x,Parameter)/pow(x,2)+(*Integrand)((1-(x+xstep))/(x+xstep),Parameter)/pow((x+xstep),2));
//   //    y += xstep/6*( (*Integrand)((1-x)/x,Parameter)/pow(x,2) + 4*(*Integrand)((1-(x+xstep/2))/(x+xstep/2),Parameter)/pow((x+xstep/2),2) + (*Integrand)((1-(x+xstep))/(x+xstep),Parameter)/pow((x+xstep),2) ); // Simpson Rule
//   // }
//   return(y);
// }

__device__
double D_Gaussian(double x, double mu, double sigma)
{
  double y=0;
  y = 1/sigma*exp( -pow((x-mu),2) / (2*sigma*sigma) );
  return(y);
}

__device__
double D_Fr(double r)
{
  if(r<=1){
    return( acos(r)-r*sqrt( 1-pow(r,2) ) );
  }
  else{
    return(0);
  }
}


// __device__
// double D_Corrfunc_Cale(IntegrandParameter Parameter)
// {
//   double y=0;
//   double xstep = .001;
//   Integrand_t Integrand = &D_Exp1;
//   double r = Parameter.r;
//   int n   = Parameter.n;
//   double Zeta = Parameter.zeta;
//   double Eta = Parameter.eta;
//   double q1 = Parameter.q1;
//   y    = 4*Eta/(q1*q1)*(0.5772156649 + logf(r/Zeta) + 0.5*D_SimpsonRule(Integrand,Parameter, (r*r)/(4*n*Zeta*Zeta),0.001,1,1000));
//   // y    = 4*Eta/(q1*q1)*(0.5772156649 + logf(r/Zeta));
//   return(y);
// }


// __device__
// double D_QAGS_Integration(Integrand_t Integrand, IntegrandParameter Parameter)
// {
//   double y=0;
//   double xstep = .001;
//   y    = D_SimpsonRule(Integrand,Parameter, 0, xstep,1-xstep,998);
//   return(y);
// }

__device__
double D_HrIntegrand(double Lr, HrIntegrandParameter Parameter)
{
  double r = Parameter.r;
  double y=0;
  double Avg_LR=Parameter.Avgr;
  double SigmaR=Parameter.sigmar;
    y    = D_Gaussian(Lr,Avg_LR,SigmaR)*Lr*Lr*D_Fr(r/Lr);
  return(y);
}

// __device__
// double D_HzIntegrand(double Lz, IntegrandParameter Parameter)
// {
//   double z = Parameter.r;
//   double D = 2*M_PI/Parameter.q1;
//   double y=0;
//   double Avg_Lz=1e4;
//   double Sigmaz=3.3e3;
//     y    = D_Gaussian(Lz,Avg_Lz,Sigmaz)*(Lz-z)/D;
//   return(y);
// }


__device__
double D_StructurefactorIntegrand(double r, HankelParameter Parameter)
{
  double y;
   if(r==0)
   {
      return(0);
   }
   else{
    y    = D_Interpolate(Parameter.hr,r,Parameter.rStart,Parameter.rStep)*D_Interpolate(Parameter.G,r/Parameter.zeta,Parameter.rStart,Parameter.rStep)*j0(Parameter.qr*r)*r;
    // y    = D_Interpolate(Parameter.hr,r,Parameter.rStart,Parameter.rStep)*D_Interpolate(Parameter.G,r/Parameter.zeta,Parameter.rStart,Parameter.rStep)*r;
    // y    = D_Interpolate(Parameter.G,r/Parameter.zeta,Parameter.rStart,Parameter.rStep)*j0(Parameter.qr*r);
  return(y);
  }
}


__device__
double D_Hr(double r, double AvgLr, double sigmaR)
{
  double y=0;
  HrIntegrandParameter Parameter = {r,0,0,0,0,AvgLr,sigmaR};
  Integrand_t Integrand = &D_HrIntegrand;
  y    = D_SimpsonRule2(Integrand,Parameter, 0, r,1e5,1000);
  return(y);
}

//
// __device__
// double D_Hz(double z)
// {
//   double y=0;
//   double xstep = .001;
//   IntegrandParameter Parameter = {z,0,0,0,0.1};
//   // double Avg_LR=1e4;
//   // double SigmaR=3.3e3;
//   Integrand_t Integrand = &D_HzIntegrand;
//   y    = D_SimpsonRule2(Integrand,Parameter, 0, z,1e5,1000);
//   return(y);
// }


__device__
double D_Interpolate(double *function, double r, double rStart, double rStep)
{
  int index;
  index = floorf((log10(r)-rStart)/rStep);
  double x1 = pow(10.0,rStart+index*rStep);
  double x2 = pow(10.0,rStart+(index+1)*rStep);
  double m = (function[index+1]-function[index])/(x2-x1);
  double b = function[index]-m*x1;
  return(m*r+b);
  // return(x2);
}


// __global__
// void G_PreProcessCorrFunc(double *xt, int Nmax, int NumR, double rStart, double rStep, double Zeta, double Eta, double dz)
// {
//   int index;
//   int n;
//   index = blockIdx.x * blockDim.x + threadIdx.x;
//   if(index<=(Nmax+1)*(NumR+1)){
//     n = floorf(index/(NumR+1));
//     int roffset = index%(NumR+1);
//     float r=pow(10.0,rStart+roffset*rStep);
//     IntegrandParameter Parameter = {r,n,Zeta,Eta,dz};
//
//     Integrand_t Integrand = &D_CorrFuncIntegrand;
//     xt[index] = D_QAGS_Integration(Integrand,Parameter);
//   }
//
// }

__global__
void G_PreProcessHr(double *xt, int NumR, double rStart, double rStep, double Avg_LR, double SigmaR)
{
  int index;
  // index = blockIdx.x * blockDim.x + threadIdx.x;
  index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index<=(NumR+1)){
    float r=pow(10.0,rStart+index*rStep);
      xt[index] = D_Hr(r,Avg_LR,SigmaR);
  }

}


// __global__
// void G_PreProcessHz(double *xt, double D, int NMax)
// {
//   int n;
//   n = threadIdx.x;
//   if(n<=(NMax+1)){
//     xt[n] = D_Hz(n*D);
//   }
//
// }

// __global__
// void G_PreProcessJr(double *xt, int NumR, double rStart, double rStep)
// {
//   int index;
//   index = blockIdx.x * blockDim.x + threadIdx.x;
//   if(index<=(NumR+1)){
//   float r=pow(10.0,rStart+index*rStep);
//     xt[index] = j0(r);
//   }
//
// }


// __global__
// void G_InterpolateHr(double *xt, double *hrarray, int NumR, double rStart, double rStep)
// {
//   int index;
//   index = blockIdx.x * blockDim.x + threadIdx.x;
//   if(index<=(NumR+1)){
//   float r=1+index;//pow(10.0,rStart+index*rStep);
//     xt[index] = D_Interpolate(hrarray,r,rStart,rStep);
//   }
//
// }


__global__
void G_CalculateNSummation(double *xt, double *cf,double *hz, int NumR, double eta, double qz, int NMax, double q1)
{
  int index;
  index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index<=(NumR+1)){
    xt[index] =0;
    // for(int n =0;n<NMax+1;n++){
    //   xt[index] += cos(qz*n*2*M_PI/q1)*hz[n]*exp(-(qz*qz*eta* cf[index+n*(NumR)] )/2);
    //   // xt[index] += exp(-(qz*qz*eta* cf[index+n*(NumR)] )/2);
    //   // xt[index] = xt[index] + cos(qz*n*2*M_PI/q1)*hz[n];
    // }
    for(int n =-NMax;n<=NMax;n++){
      // xt[index] += cos(qz*n*2*M_PI/q1)*hz[abs(n)]*exp(-(qz*qz*eta* cf[index+abs(n)*(NumR)] )/2);
      xt[index] += cos(qz*n*2*M_PI/q1)*hz[abs(n)]*exp(-(qz*qz*eta* cf[index+abs(n)*(NumR)]+n*n*2*M_PI/q1*2*M_PI/q1*0.004*0.002)/(2*(1+cf[index+abs(n)*(NumR)]*0.004*0.002) ) );
      // xt[index] += exp(-(qz*qz*eta* cf[index+n*(NumR)] )/2);
      // xt[index] = xt[index] + cos(qz*n*2*M_PI/q1)*hz[n];
    }
  }
}


__global__
void G_HankelTransform(double *xt, double *G,double *hr, int NumR, double eta,double zeta,double qz, int NMax,double rStart,double rStep)
{
  double qr = 0;
  int index;
  index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index<=(NumR+1)){
  qr = floorf(index/1000)*.001;
  int offset = index%1000;
  HankelParameter Parameter = {zeta,eta,qr,qz,hr,G,rStart,rStep};
  HankelIntegrand_t Integrand = &D_StructurefactorIntegrand;
  // xt[index] = D_Trapz(Integrand, Parameter, 1+offset*100, (offset+1)*100, 1);//D_StructurefactorIntegrand(r,Parameter);
  // xt[index] = D_Trapz(Integrand, Parameter, .00001+offset*100, (offset+1)*100, .1);//D_StructurefactorIntegrand(r,Parameter);
  xt[index] = D_Trapz(Integrand, Parameter, offset*100, (offset+1)*100, 1);//D_StructurefactorIntegrand(r,Parameter);
  }
}


// __global__
// void G_StructurFactorIntegrand(double *xt, double *hrarray, double qr, int NumR, double rStart, double rStep)
// {
//   int index;
//   index = blockIdx.x * blockDim.x + threadIdx.x;
//   if(index<=(NumR+1)){
//   float r=1+index;//pow(10.0,rStart+index*rStep);
//     xt[index] = j0(qr*r)*D_Interpolate(hrarray,r,rStart,rStep);
//   }
//
// }
