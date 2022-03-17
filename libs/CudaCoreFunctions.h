// MyKernel.h
#ifndef CudaCoreFunctions_h
#define CudaCoreFunctions_h

struct IntegrandParameter {float r; int n; double zeta; double eta; double q1;};
struct HrIntegrandParameter {float r; int n; double zeta; double eta; double q1; double Avgr; double sigmar;};
struct HankelParameter {double zeta; double eta; double qr; double qz; double* hr; double* G; double rStart; double rStep;};

typedef double (*Integrand_t)(double, HrIntegrandParameter);
typedef double (*HankelIntegrand_t)(double, HankelParameter);


void GetCudaInfo();
__device__ double D_CorrFuncIntegrand(double, IntegrandParameter);
__device__ double D_Gaussian(double x, double mu, double sigma);
__device__ double D_SimpsonRule(Integrand_t, IntegrandParameter, double, double, double);
__device__ double D_SimpsonRule2(Integrand_t, HrIntegrandParameter, double, double, double, double);
__device__ double D_QAGS_Integration(Integrand_t, IntegrandParameter);
__device__ double D_Corrfunc_Cale(IntegrandParameter);
__device__ double D_Interpolate(double *, double, double, double);
__device__ double D_Trapz(HankelIntegrand_t, HankelParameter, double, double, double);
__device__ double D_Gaussian(double, double, double);
__device__ double D_Fr(double);
__device__ double D_HrIntegrand(double, HrIntegrandParameter);
__device__ double D_StructurefactorIntegrand(double, HankelParameter);
__device__ double D_Hr(double, double, double);
__device__ double D_Interpolate(double *, double, double, double);
__global__ void G_PreProcessCorrFunc(double*, int, int, double, double, double, double, double);
__global__ void G_PreProcessHr(double*, int, double , double, double, double);
__global__ void G_PreProcessJr(double*, int, double , double );
__global__ void G_InterpolateHr(double *, double *, int , double , double );
__global__ void G_StructurFactorIntegrand(double*, double*, double, int, double, double);
__global__ void G_PreProcessHz(double*, double, int);
__global__ void G_CalculateNSummation(double*, double*, double*,int, double, double, int, double);
__global__ void G_HankelTransform(double*, double*,double*, int, double, double, double, int, double, double);
#endif
