#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_sf_airy.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_sf_expint.h>
// #include <gsl/gsl_filter.h>
#include <string>
#include <iostream>
#include <time.h>
#include <sstream>
#include <iomanip>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_dht.h>
#include "diffuselibCuda.h"
#include "diffusetoolbox.h"
#include "CudaCoreFunctions.h"
#include "GraphicsLib.h"

void DiffuseXRD::ListSimParameters(){
  std::cout<<"|"<<std::setw(20)<<"Parameter"<<"|"<<std::setw(15)<<"Value"<<"|"<<std::setw(10)<<"Unit"<<"|"<<std::endl;
  for(int k=1; k<=49; k++){
    std::cout<<"_";
  }
  std::cout<<std::endl;
  std::cout<<"|"<<std::setw(20)<<"Dspacing"<<"|"<<std::setw(15)<<m_Dspacing<<"|"<<std::setw(10)<<"Ang"<<"|"<<std::endl;
  std::cout<<"|"<<std::setw(20)<<"Q1"<<"|"<<std::setw(15)<<m_q1<<"|"<<std::setw(10)<<"Ang^-1"<<"|"<<std::endl;
  std::cout<<"|"<<std::setw(20)<<"Zeta"<<"|"<<std::setw(15)<<m_zeta<<"|"<<std::setw(10)<<"Ang"<<"|"<<std::endl;
  std::cout<<"|"<<std::setw(20)<<"delta Zeta"<<"|"<<std::setw(15)<<m_dzeta<<"|"<<std::setw(10)<<"Ang"<<"|"<<std::endl;
  std::cout<<"|"<<std::setw(20)<<"Eta"<<"|"<<std::setw(15)<<m_eta<<"|"<<std::setw(10)<<"-"<<"|"<<std::endl;
  std::cout<<"|"<<std::setw(20)<<"delta Eta"<<"|"<<std::setw(15)<<m_deta<<"|"<<std::setw(10)<<"-"<<"|"<<std::endl;
  std::cout<<"|"<<std::setw(20)<<"B"<<"|"<<std::setw(15)<<m_B<<"|"<<std::setw(10)<<"kTAng^-4"<<"|"<<std::endl;
  std::cout<<"|"<<std::setw(20)<<"delta B"<<"|"<<std::setw(15)<<m_dB<<"|"<<std::setw(10)<<"kTAng^-4"<<"|"<<std::endl;
  std::cout<<"|"<<std::setw(20)<<"Kc"<<"|"<<std::setw(15)<<m_Kc<<"|"<<std::setw(10)<<"kT"<<"|"<<std::endl;
  std::cout<<"|"<<std::setw(20)<<"delta Kc"<<"|"<<std::setw(15)<<m_dKc<<"|"<<std::setw(10)<<"kT"<<"|"<<std::endl;
  std::cout<<"|"<<std::setw(20)<<"Avg Lr"<<"|"<<std::setw(15)<<m_AvgLr<<"|"<<std::setw(10)<<"Ang"<<"|"<<std::endl;
  std::cout<<"|"<<std::setw(20)<<"Sigma R"<<"|"<<std::setw(15)<<m_SigmaR<<"|"<<std::setw(10)<<"Ang"<<"|"<<std::endl;
  std::cout<<"|"<<std::setw(20)<<"Avg Lz"<<"|"<<std::setw(15)<<m_AvgLz<<"|"<<std::setw(10)<<"Ang"<<"|"<<std::endl;
  std::cout<<"|"<<std::setw(20)<<"Sigma Z"<<"|"<<std::setw(15)<<m_SigmaZ<<"|"<<std::setw(10)<<"Ang"<<"|"<<std::endl;
  std::cout<<"|"<<std::setw(20)<<"Beam Width"<<"|"<<std::setw(15)<<m_bwdt<<"|"<<std::setw(10)<<"Ang^-1"<<"|"<<std::endl;
  std::cout<<"|"<<std::setw(20)<<"initial Qz"<<"|"<<std::setw(15)<<m_qzstart<<"|"<<std::setw(10)<<"Ang^-1"<<"|"<<std::endl;
  std::cout<<"|"<<std::setw(20)<<"final Qz"<<"|"<<std::setw(15)<<m_qzstop<<"|"<<std::setw(10)<<"Ang^-1"<<"|"<<std::endl;
  std::cout<<"|"<<std::setw(20)<<"Qz stepwidth"<<"|"<<std::setw(15)<<m_qzstep<<"|"<<std::setw(10)<<"Ang^-1"<<"|"<<std::endl;
  std::cout<<"|"<<std::setw(20)<<"initial R"<<"|"<<std::setw(15)<<m_StartR<<"|"<<std::setw(10)<<"Ang"<<"|"<<std::endl;
  std::cout<<"|"<<std::setw(20)<<"final R"<<"|"<<std::setw(15)<<m_StopR<<"|"<<std::setw(10)<<"Ang"<<"|"<<std::endl;
  std::cout<<"|"<<std::setw(20)<<"R stepwidth"<<"|"<<std::setw(15)<<m_StepSize<<"|"<<std::setw(10)<<"Ang"<<"|"<<std::endl;
  std::cout<<"|"<<std::setw(20)<<"CF R-length"<<"|"<<std::setw(15)<<m_Rlength<<"|"<<std::setw(10)<<"-"<<"|"<<std::endl;
  std::cout<<"|"<<std::setw(20)<<"Hr R-Length"<<"|"<<std::setw(15)<<m_RlengthHr<<"|"<<std::setw(10)<<"-"<<"|"<<std::endl;
  std::cout<<"|"<<std::setw(20)<<"RIntMax"<<"|"<<std::setw(15)<<m_RIntMax<<"|"<<std::setw(10)<<"-"<<"|"<<std::endl;
  std::cout<<"|"<<std::setw(20)<<"RIntStep"<<"|"<<std::setw(15)<<m_RIntStep<<"|"<<std::setw(10)<<"-"<<"|"<<std::endl;
  std::cout<<"|"<<std::setw(20)<<"Qr Intervals"<<"|"<<std::setw(15)<<m_QrIntervals<<"|"<<std::setw(10)<<"-"<<"|"<<std::endl;
  std::cout<<"|"<<std::setw(20)<<"IntIntervals"<<"|"<<std::setw(15)<<m_IntIntervals<<"|"<<std::setw(10)<<"-"<<"|"<<std::endl;
  std::cout<<"|"<<std::setw(20)<<"NHankelTransform"<<"|"<<std::setw(15)<<m_NHankelTransform<<"|"<<std::setw(10)<<"-"<<"|"<<std::endl;
  std::cout<<"|"<<std::setw(20)<<"Fitdatasets"<<"|"<<std::setw(15)<<m_NumFitDatSets<<"|"<<std::setw(10)<<"-"<<"|"<<std::endl;
  std::cout<<"|"<<std::setw(20)<<"Fitdatalines"<<"|"<<std::setw(15)<<m_NumFitDatLines<<"|"<<std::setw(10)<<"-"<<"|"<<std::endl;
  std::cout<<std::endl;
}

double DiffuseXRD::CorrFuncApprox(double r, int n){
  r = r/m_zeta;
  int  index;
  if (r<m_r[0]){
    return(m_eta*m_CorrFuncTable[n]);
  }
  if (r<=m_r[m_Rlength]){
    index = floorf((log10(r)-m_StartR)/m_StepSize);
    double slope     = (m_CorrFuncTable[index+1] - m_CorrFuncTable[index])/(m_r[index+1]-m_r[index]);
    double intersept = m_CorrFuncTable[index] - slope*m_r[index];
    return(m_eta*(slope*r+intersept));
  }
  return(0);
}

void DiffuseXRD::PreProcessCorrFunc(int nmax){
  PrintProcessInfo("PreProcessCorrFunc","CPU",0,0,0);
  std::cout<<std::endl;
  for(int n = 0; n<=nmax; n++){
    PrintProgress( ((double)n) / ((double)nmax) * 100 );
    double q1 = m_q1;
    for(int k=0; k<=m_Rlength; k++){
      if((n<=30)&&(m_r[k]<=1000))
      {
        m_CorrFuncTable[n*(m_Rlength+1) + k] = CorrFunc(m_r[k],1,1,n,q1);
      }
      else{
        m_CorrFuncTable[n*(m_Rlength+1) + k] = CalleApproximation_pre(m_r[k],n,q1);
      }
    }
  }
  std::cout<<"\n"<<std::endl;
}


void DiffuseXRD::LoadCorrFunc(char* FileName){
  PrintProcessInfo("Load Correlation Function","CPU",0,0,0);
  std::cout<<std::endl;
  ReadBinFile(m_CorrFuncTable,m_Rlength+1,CNMAX+1,FileName);
}

void DiffuseXRD::PreProcessHz(int nmax){
  PrintProcessInfo("PreProcessHz","CPU",0,0,0);
  double NormFactor = 1;//Hz(0*m_Dspacing,m_AvgLz,m_SigmaZ,m_Dspacing);
  for(int n = 0; n<=nmax; n++){
    m_HzTable[n] = 1/NormFactor*Hz(n*m_Dspacing,m_AvgLz,m_SigmaZ,m_Dspacing);
  }
}

void DiffuseXRD::PreProcessHr(){

  int CurrDevice = 0;
  cudaSetDevice(CurrDevice);
  double *x;
  double *x_CPU;
  int N = (m_RlengthHr+1);
  std::cerr<<"Allocate Memory"<<std::endl;
  x_CPU = (double*)malloc(sizeof(double) * N);
  cudaMalloc(&x, N*sizeof(double));
  std::cerr<<"Memory Allocated"<<std::endl;

  cudaError_t err = cudaGetLastError();        // Get error code

   if ( err != cudaSuccess )
   {
     std::cerr<<"CUDA Error:"<<cudaGetErrorString(err)<<std::endl;
      exit(-1);
   }
  // initialize x arrays on the host
  for (int i = 0; i < N; i++) {
       x_CPU[i] = 0.0;
   }

  cudaMemcpy(x, x_CPU, sizeof(double) * N, cudaMemcpyHostToDevice);

  // Run kernel on all elements on the GPU
   int blockSize;
   int numBlocks;

   blockSize = 1024;
   numBlocks = N / blockSize+1;;
   cudaGetDevice(&CurrDevice);
   PrintProcessInfo("PreProcessHr","GPU",CurrDevice,numBlocks,blockSize);
   G_PreProcessHr<<<numBlocks, blockSize>>>(x,m_RlengthHr,m_StartR,m_StepSize,m_AvgLr,m_SigmaR);
   cudaDeviceSynchronize();
   cudaMemcpy(x_CPU, x, sizeof(double) * N, cudaMemcpyDeviceToHost);
   double NormFactor = 1;//x[0];
   for(int n = 0; n<N; n++){
       m_HrTable[n] = M_PI*(double)x_CPU[n]/NormFactor;
   }
   std::cerr<<"Data copied\n";
   cudaFree(x);
   std::cerr<<"Unified memory freed\n";
}


void DiffuseXRD::PostProcessHankel(){

  for(int k = 0; k<m_QrIntervals; k++){
    m_QrScan[k] = 0;
    for(int j = 0;j<m_IntIntervals;j++){
      m_QrScan[k] += m_HankelTransform[k*m_IntIntervals+j];
    }
  }

  int NumBlurPoints = 19;//!!!

  double* Kernel;
  double* BlurringData;
  double qrKernel = 0;
  double sigma = m_bwdt; //!!!
  Kernel = new double [2*NumBlurPoints+1];
  BlurringData = new double [m_QrIntervals+2*NumBlurPoints];

  for(int k =0; k<2*NumBlurPoints+1; k++)
  {
    qrKernel = -0.019+k*0.001;
    Kernel[k] = 1/sigma/sqrt(2*M_PI)*exp(-(qrKernel*qrKernel/2/(sigma*sigma)));
    // std::cout<<qrKernel<<"\t"<<Kernel[k]<<std::endl;
  }
  // std::cout<<"\n\n";
  for(int k =0;k<m_QrIntervals+2*NumBlurPoints;k++){
    if(k<=NumBlurPoints){
      BlurringData[k]=m_QrScan[NumBlurPoints-k];
    }
    else if(k>=m_QrIntervals+NumBlurPoints){
      BlurringData[k] = m_QrScan[m_QrIntervals-1];
    }
    else{
      BlurringData[k] = m_QrScan[k-NumBlurPoints];
    }
    // std::cout<<BlurringData[k]<<std::endl;
  }

  for(int k =0;k<m_QrIntervals;k++) {
    m_QrScan[k]=0;
    for(int j =0;j<2*NumBlurPoints+1;j++) {
      m_QrScan[k]+=BlurringData[k+j]*Kernel[j];
      // if(k==0){
      //   // std::cout<<j<<"\t"<<BlurringData[k+j]<<"\t"<<Kernel[j]<<std::endl;
      // }
    }
  }

}


void DiffuseXRD::PreProcessNSummation(double qz){

  int CurrDevice = 0;
  cudaSetDevice(CurrDevice);
  double *x;
  double *y;
  double *hz;
  double *x_CPU;
  double *y_CPU;
  double *hz_CPU;
  int N = (m_Rlength+1);


  x_CPU = (double*)malloc(N*sizeof(double));
  cudaMalloc(&x, N*sizeof(double));
  y_CPU = (double*)malloc(((m_Rlength+1)*(CNMAX+1))*sizeof(double));
  cudaMalloc(&y, ((m_Rlength+1)*(CNMAX+1))*sizeof(double));
  hz_CPU = (double*)malloc((CNMAX+1)*sizeof(double));
  cudaMalloc(&hz, (CNMAX+1)*sizeof(double));

  // initialize x arrays on the host
  for (int i = 0; i < N; i++) {
       x_CPU[i] = 0.0;
   }
   for (int i = 0; i < ((m_Rlength+1)*(CNMAX+1)); i++) {
        y_CPU[i] = m_CorrFuncTable[i];
    }
    for (int i = 0; i < ((CNMAX+1)); i++) {
         hz_CPU[i] = m_HzTable[i];
     }

    cudaMemcpy(x, x_CPU, sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(y, y_CPU, ((m_Rlength+1)*(CNMAX+1))*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(hz, hz_CPU, (CNMAX+1)*sizeof(double), cudaMemcpyHostToDevice);
  // Run kernel on all elements on the GPU
   int blockSize;
   int numBlocks;
   // std::cout<<"MEta ="<<m_eta<<std::endl;
   blockSize = 1024;
   numBlocks = N / blockSize+1;;
   cudaGetDevice(&CurrDevice);
   G_CalculateNSummation<<<numBlocks, blockSize>>>(x,y,hz,N,m_eta,qz,CNMAX,m_q1);
   cudaDeviceSynchronize();
   cudaMemcpy(x_CPU,x, sizeof(double) * N, cudaMemcpyDeviceToHost);

   for(int n = 0; n<N; n++){
       m_SummationTable[n] = (double)x_CPU[n];
   }
   cudaFree(x);
   cudaFree(y);
   cudaFree(hz);
   free(x_CPU);
   free(y_CPU);
   free(hz_CPU);
}


void DiffuseXRD::HankelTransformation(double qz){
    int CurrDevice = 0;
    cudaSetDevice(CurrDevice);

    double *x;
    double *y;
    double *hr;
    double *x_CPU;
    double *y_CPU;
    double *hr_CPU;
    int N = (m_Rlength+1);

    y_CPU = (double*)malloc(N*sizeof(double));
    cudaMalloc(&y, N*sizeof(double));
    hr_CPU = (double*)malloc(N*sizeof(double));
    cudaMalloc(&hr, N*sizeof(double));

    // initialize x arrays on the host
    for (int i = 0; i < N; i++) {
         y_CPU[i] = m_SummationTable[i];
         hr_CPU[i] = m_HrTable[i];
     }

     cudaMemcpy(y, y_CPU, N*sizeof(double), cudaMemcpyHostToDevice);
     cudaMemcpy(hr, hr_CPU, N*sizeof(double), cudaMemcpyHostToDevice);

     N = m_NHankelTransform+1;
     x_CPU = (double*)malloc(N*sizeof(double));
     cudaMalloc(&x, N*sizeof(double));
     for (int i = 0; i < N; i++) {
          x_CPU[i] = 0.0;
      }

    cudaMemcpy(x, x_CPU, N*sizeof(double), cudaMemcpyHostToDevice);

  // Run kernel on all elements on the GPU
   int blockSize;
   int numBlocks;

   blockSize = 1024;
   numBlocks = N / blockSize+1;
   // std::cout<<"here"<<numBlocks<<"\t"<<blockSize<<"\t"<<N<<std::endl;
   cudaGetDevice(&CurrDevice);
   G_HankelTransform<<<numBlocks, blockSize>>>(x,y,hr,N,m_eta,m_zeta,qz,CNMAX,m_StartR,m_StepSize);
   cudaDeviceSynchronize();

   cudaMemcpy(x_CPU, x, N*sizeof(double), cudaMemcpyDeviceToHost);

   for(int n = 0; n<N; n++){
       m_HankelTransform[n] = (double)x_CPU[n];
   }
   cudaFree(x);
   cudaFree(y);
   cudaFree(hr);
   free(x_CPU);
   free(y_CPU);
   free(hr_CPU);
}



double DiffuseXRD::InterpolateQrScan(double qpar){
  // qr = k*.001; //!!! fixe values
  int k = qpar/0.001;//!!!
  if(qpar>0&&qpar<100*0.001){
  double x1 = k*0.001;
  double x2 = (k+1)*0.001;
  double slope     = (m_QrScan[k+1] - m_QrScan[k])/(x2-x1);
  double intersept = m_QrScan[k] - slope*(k*0.001);
  return((slope*qpar+intersept));
  }
  else if(qpar<=0){
    return(m_QrScan[0]);
  }
  else if(qpar>=100*0.001){
    return(m_QrScan[99]);
  }
  return(0);
}


void DiffuseXRD::NormalizeQrScan(){
  double NormFactor = m_QrScan[0];
  for(int k=0;k<QRuns;k++){
    m_QrScan[k]=m_QrScan[k]/NormFactor;
  }
}

double DiffuseXRD::XiSquareCuda(const gsl_vector *v, void* parameter){
  double XiSquare;
  double ExpectedValue;
  double Normalisation = 0;
  double* qpardata = m_FitQr;
  double* Data = m_FitData;
  double ScanLength = m_NumFitDatLines;
  double qz = m_qzstart;

  m_eta = gsl_vector_get(v,0);
  m_zeta = gsl_vector_get(v,1);
  ConvertUnitsCale();

  PreProcessNSummation(qz);
  HankelTransformation(qz);
  PostProcessHankel();
  Normalisation = InterpolateQrScan(qpardata[0]);
  XiSquare = 0;
  for(int k = 1; k<ScanLength-1; k++){
    ExpectedValue = InterpolateQrScan(qpardata[k])/Normalisation;
    XiSquare = XiSquare + pow(Data[k]-ExpectedValue,2)/ExpectedValue;
  }
  return(XiSquare);
}

void DiffuseXRD::QzScan(){
  int scancount;
  scancount = (int)((m_qzstop-m_qzstart)/m_qzstep);
  double qztmp = 0;
  for(int l =0;l<scancount;l++){
    PrintProgress( ((double)l) / ((double)scancount) * 100 );
    qztmp = m_qzstart+l*m_qzstep;
    PreProcessNSummation(qztmp);
    HankelTransformation(qztmp);
    PostProcessHankel();
    m_2DData[l][QRuns] = m_QrScan[0];
    for(int k=1;k<QRuns-1;k++){
      m_2DData[l][QRuns-k] = m_QrScan[k];
      m_2DData[l][QRuns+k] = m_QrScan[k];
    }
   }
}

void DiffuseXRD::InitiateQzScan(){
  int scancount;
  scancount = (int)((m_qzstop-m_qzstart)/m_qzstep);
  m_2DData = new double [scancount][2*QRuns];
}

void DiffuseXRD::WriteQrVector(char* FileName){
  WriteHeader(FileName,0,0);
  ofstream OutputFile;
  OutputFile.open(FileName,ios::app);
  for(int k = 0; k<QRuns; k++){
    OutputFile<<k*.001<<"\t"<<m_QrScan[k]<<"\n";
    // OutputFile<<m_r[k]<<"\t"<<m_SummationTable[k]<<"\n";
  }
  OutputFile.close();
}

void DiffuseXRD::WriteFitData(char* FileName, int DatIndex){
  WriteHeader(FileName,0,0);
  ofstream OutputFile;
  OutputFile.open(FileName,ios::app);
  double NormFactor = InterpolateQrScan(m_FitQr[0]);
  double Yk;
  for(int k = 0; k<m_NumFitDatLines; k++){
    Yk = (InterpolateQrScan(m_FitQr[k]))/NormFactor;
    OutputFile<<m_FitQr[k]<<"\t"<<gsl_matrix_get(m_FitData_comb,DatIndex,k)<<"\t"<<gsl_matrix_get(m_FitErrData_comb,DatIndex,k)<<"\t"<<Yk<<"\n";
    // OutputFile<<m_r[k]<<"\t"<<m_SummationTable[k]<<"\n";
  }
  OutputFile.close();
}

void DiffuseXRD::WriteFitData2(char* FileName){
  WriteHeader(FileName,0,0);
  ofstream OutputFile;
  OutputFile.open(FileName,ios::app);
  double NormFactor = InterpolateQrScan(m_FitQr[0]);
  double Yk;
  for(int k = 0; k<m_NumFitDatLines; k++){
    Yk = (InterpolateQrScan(m_FitQr[k]))/NormFactor;
    OutputFile<<m_FitQr[k]<<"\t"<<gsl_matrix_get(m_FitData_comb,1,k)<<"\t"<<gsl_matrix_get(m_FitErrData_comb,1,k)<<"\t"<<Yk<<"\n";
    // OutputFile<<m_r[k]<<"\t"<<m_SummationTable[k]<<"\n";
  }
  OutputFile.close();
}

void DiffuseXRD::WriteHeader(char* FileName, int SizeX, int SizeY){
  ofstream OutputFile;
  OutputFile.open(FileName);
  OutputFile<<"{"<<std::endl;
  OutputFile<<"NUMLINES="<<27<<std::endl;
  OutputFile<<"FILENAME="<<FileName<<std::endl;
  OutputFile<<"SOFTWAREVERSION="<<DIFFVERSION<<std::endl;
  OutputFile<<"DSPACING="<<m_Dspacing<<std::endl;
  OutputFile<<"Q1="<<m_q1<<std::endl;
  OutputFile<<"ZETA="<<m_zeta<<std::endl;
  OutputFile<<"DZETA="<<m_dzeta<<std::endl;
  OutputFile<<"ETA="<<m_eta<<std::endl;
  OutputFile<<"DETA="<<m_deta<<std::endl;
  OutputFile<<"B="<<m_B<<std::endl;
  OutputFile<<"DB="<<m_dB<<std::endl;
  OutputFile<<"KC="<<m_Kc<<std::endl;
  OutputFile<<"DKC="<<m_dKc<<std::endl;
  OutputFile<<"AVGLR="<<m_AvgLr<<std::endl;
  OutputFile<<"DAVGLR="<<"NAN"<<std::endl;
  OutputFile<<"SIGMAR="<<m_SigmaR<<std::endl;
  OutputFile<<"AVGLZ="<<m_AvgLz<<std::endl;
  OutputFile<<"DAVGLZ="<<"NAN"<<std::endl;
  OutputFile<<"SIGMAZ="<<m_SigmaZ<<std::endl;
  OutputFile<<"SIZEX="<<SizeX<<std::endl;
  OutputFile<<"SIZEY="<<SizeY<<std::endl;
  OutputFile<<"QRSTART="<<0<<std::endl;//!!!!
  OutputFile<<"QRSTOP="<<QRuns*0.001<<std::endl;//!!!!
  OutputFile<<"QRSTEP="<<0.001<<std::endl;//!!!!
  OutputFile<<"QZSTART="<<m_qzstart<<std::endl;
  OutputFile<<"QZSTOP="<<m_qzstop<<std::endl;
  OutputFile<<"QZSTEP="<<m_qzstep<<std::endl;
  OutputFile<<"XI2="<<m_xi2<<std::endl;
  OutputFile<<"N="<<m_NumFitDatLines<<std::endl;
  OutputFile<<"}"<<std::endl;
  OutputFile.close();
}


void DiffuseXRD::Save2DScan(char* FileName){
  int n = (int)((m_qzstop-m_qzstart)/m_qzstep);
  int m = 2*QRuns;
  WriteHeader(FileName,n,m);
  fstream OutputFile;
  OutputFile.open(FileName,ios::app|ios::binary);
  for(int k = 0; k<n; k++){
    for(int l = 0; l<m; l++){
      OutputFile.write((char*)&(m_2DData[k][l]),sizeof(double));
    }
  }
  OutputFile.close();
}

void DiffuseXRD::SetBeamWidth(double bwdt){
    m_bwdt = bwdt;
}

void DiffuseXRD::SetZeta(double zeta){
    m_zeta = zeta;
}

void DiffuseXRD::SetDZeta(double value){
    m_dzeta = value;
}

void DiffuseXRD::SetEta(double eta){
    m_eta = eta;
}

void DiffuseXRD::SetDEta(double value){
    m_deta = value;
}

void DiffuseXRD::SetQ1(double q1){
    m_Dspacing = 2*M_PI/q1;
    m_q1 = q1;
}

void DiffuseXRD::SetQzStart(double qzstart){
  m_qzstart = qzstart;
}

void DiffuseXRD::SetQzStop(double qzstop){
  m_qzstop = qzstop;
}

void DiffuseXRD::SetQzStep(double qzstep){
  m_qzstep = qzstep;
}

void DiffuseXRD::SetAvgLr(double value){
  m_AvgLr = value;
}

void DiffuseXRD::SetSigmaR(double value){
  m_SigmaR = value;
}
void DiffuseXRD::SetAvgLz(double value){
  m_AvgLz = value;
}
void DiffuseXRD::SetSigmaZ(double value){
  m_SigmaZ = value;
}

void DiffuseXRD::SetXI2(double value){
  m_xi2= value;
}

double DiffuseXRD::GetZeta(){
  return(m_zeta);
}

double DiffuseXRD::GetEta(){
  return(m_eta);
}

double DiffuseXRD::GetQ1(){
  return(m_q1);
}

double DiffuseXRD::GetQzStart(){
  return(m_qzstart);
}

double DiffuseXRD::GetQzStart2(){
  return(m_qzstart2);
}

double DiffuseXRD::GetQzStop(){
  return(m_qzstop);
}

double DiffuseXRD::GetQzStep(){
  return(m_qzstep);
}

double DiffuseXRD::GetAvgLr(){
  return(m_AvgLr);
}

double DiffuseXRD::GetAvgLz(){
  return(m_AvgLz);
}

double DiffuseXRD::GetSigmaR(){
  return(m_SigmaR);
}

double DiffuseXRD::GetSigmaZ(){
  return(m_SigmaZ);
}

double DiffuseXRD::GetNumDataLines(){
  return(m_NumFitDatLines);
}

double DiffuseXRD::GetNumDataSets(){
  return(m_NumFitDatSets);
}

double DiffuseXRD::GetXI2(){
  return(m_xi2);
}

double DiffuseXRD::GetKc(){
  return(m_Kc);
}

double DiffuseXRD::GetB(){
  return(m_B);
}

double DiffuseXRD::GetDKc(){
  return(m_dKc);
}

double DiffuseXRD::GetDB(){
  return(m_dB);
}

void DiffuseXRD::ConvertUnitsCale(){
  m_B = 4*M_PI/8/m_eta/m_zeta/m_zeta/m_Dspacing/m_Dspacing; // in units of kBT: B=4pikBT/(8 eta zeta^2 D^2)
  m_Kc = m_zeta*m_zeta*4*M_PI/8/m_eta/m_Dspacing/m_Dspacing; // in units of kBT: Kc=4pi zeta^2 kBT/(8 eta D^2)

  double dkcde = -m_zeta*m_zeta*4*M_PI/8/m_eta/m_eta/m_Dspacing/m_Dspacing;
  double dkcdz= -m_zeta*8*M_PI/8/m_eta/m_Dspacing/m_Dspacing;

  double dBde = 4*M_PI/8/m_eta/m_eta/m_zeta/m_zeta/m_Dspacing/m_Dspacing;
  double dBdz= 2*M_PI/8/m_eta/m_zeta/m_zeta/m_zeta/m_Dspacing/m_Dspacing;

  m_dB = sqrt( pow(dBde*m_deta,2)+pow(dBdz*m_dzeta,2) );
  m_dKc = sqrt( pow(dkcde*m_deta,2)+pow(dkcdz*m_dzeta,2) );

}

void DiffuseXRD::ReadDataFile(char* FileName){
  // float datatmp, errdatatmp, datatmp2, errdatatmp2, qrtmp, relerror;
  float relerror;
  int NumDatPoints,NumHeaderLines;
  // double q1tmp,qzbtmp,NormFactor,SubFactor;
  double q1tmp,qzbtmp,NormFactor;
  // double qzbtmp;
  char lineText[1000];
  ifstream in(FileName);
  // double NormFactor;
  //Read Header
  in.getline(lineText, 99);
  in.getline(lineText, 99);
  sscanf(lineText, "NUMLINES=%d",&NumHeaderLines);
  in.getline(lineText, 99);
  in.getline(lineText, 99);
  sscanf(lineText, "NUMDATLINES=%d",&NumDatPoints);
  in.getline(lineText, 99);
  sscanf(lineText, "Q1=%lf",&q1tmp);
  SetQ1(q1tmp);
  m_NumFitDatLines = NumDatPoints;
  m_NumFitDatSets = NumHeaderLines-6;
  m_FitData_comb = gsl_matrix_alloc (m_NumFitDatSets, m_NumFitDatLines);
  m_FitErrData_comb = gsl_matrix_alloc (m_NumFitDatSets, m_NumFitDatLines);
  m_FitData_qzVals = gsl_vector_alloc (m_NumFitDatSets);
  // m_FitData_qparVals = gsl_vector_alloc (m_NumFitDatLines);

  for(int k=6;k<=NumHeaderLines-1;k++){
    in.getline(lineText, 99);
    sscanf(lineText, "QZSTART=%lf",&qzbtmp);
    gsl_vector_set(m_FitData_qzVals,k-6,qzbtmp);
  }
  // in.getline(lineText, 99);
  // sscanf(lineText, "QZSTART=%lf",&qzbtmp);
  m_qzstart = gsl_vector_get(m_FitData_qzVals,0);
  // in.getline(lineText, 99);
  // sscanf(lineText, "QZSTART=%lf",&qzbtmp);
  m_qzstart2 = gsl_vector_get(m_FitData_qzVals,1);
  in.getline(lineText, 1000);


  //Allocate Memory
  m_FitQr = new double [m_NumFitDatLines];
  m_FitData = new double [m_NumFitDatLines];
  m_FitErrData = new double [m_NumFitDatLines];
  m_FitData2 = new double [m_NumFitDatLines];
  m_FitErrData2 = new double [m_NumFitDatLines];

  //Read Data
  // std::cout<<m_NumFitDatLines<<std::endl;
  // std::cout<<m_NumFitDatSets<<std::endl;
  for(int k =0; k<m_NumFitDatLines; k++){
        in.getline(lineText, 1000);
        // std::cout<<lineText<<std::endl;
        std::istringstream ss(lineText);
        std::string token;
        // int tempval;
        // stringstream ss(lineText); // convert string into a stream
        int ColumnCount = 0;
        while (std::getline(ss, token, ' '))     // convert each word on the stream into an int
        {
            // std::cout<<token.c_str()<<std::endl;
            if(ColumnCount==0){
              m_FitQr[k] = (double)(atof(token.c_str()));
            }
            else{
              if(ColumnCount%2==1){
                // std::cout<<token<<" data"<<std::endl;
                gsl_matrix_set(m_FitData_comb,floor(ColumnCount/2),k,(double)(atof(token.c_str())));
              }
              else{
                // std::cout<<token<<" error"<<std::endl;
                gsl_matrix_set(m_FitErrData_comb,floor(ColumnCount/2)-1,k,(double)(atof(token.c_str())));
              }
            }
            ColumnCount++;
        }
        // std::cout<<"  "<<std::endl;
        // std::cout<<"  "<<std::endl;

        // std::cout<<std::endl;
        // // std::cout<<lineText<<std::endl;
        // sscanf(lineText, "%f\t%f\t%f\t%f\t%f",&qrtmp,&datatmp,&errdatatmp,&datatmp2,&errdatatmp2);
        // // m_FitQr[k] = (double)qrtmp;
        // m_FitData[k] = (double)datatmp;
        // m_FitErrData[k] = (double)errdatatmp;
        // m_FitData2[k] = (double)datatmp2;
        // m_FitErrData2[k] = (double)errdatatmp2;
        // std::cout<<m_FitQr[k]<<"\t"<<m_FitData[k]<<m_FitQr[k]<<"\t"<<m_FitData2[k]<<std::endl;
    }
    in.close();
    // std::cout<<std::endl;
    // std::cout<<std::endl;
    for( int l = 0;l<m_NumFitDatSets;l++){

      NormFactor = gsl_matrix_get(m_FitData_comb,l,0);
      for(int k =0;k<m_NumFitDatLines;k++){

            relerror = gsl_matrix_get(m_FitErrData_comb,l,k)/gsl_matrix_get(m_FitData_comb,l,k);
            if (gsl_matrix_get(m_FitData_comb,l,k) ==0)
            {
              relerror = 0.01;
            }
            gsl_matrix_set(m_FitData_comb,l,k,gsl_matrix_get(m_FitData_comb,l,k)/NormFactor);
            gsl_matrix_set(m_FitErrData_comb,l,k,gsl_matrix_get(m_FitData_comb,l,k)*relerror);
            if(gsl_matrix_get(m_FitErrData_comb,l,k)<0.002){
              gsl_matrix_set(m_FitErrData_comb,l,k,0.01);
              m_FitErrData[k] = 0.01;
            }
            // std::cout<<gsl_matrix_get(m_FitData_comb,l,k)<<"\t"<<gsl_matrix_get(m_FitErrData_comb,l,k)<<std::endl;
        }
    }


    // SubFactor = 0;//m_FitData[NumDatPoints-1];
    // std::cout<<"SubFactor Data = "<<SubFactor<<std::endl;
    // NormFactor = m_FitData[0]-SubFactor;
    // for(int k =0;k<NumDatPoints;k++){
    //       relerror = m_FitErrData[k]/m_FitData[k];
    //       if (m_FitData[k] ==0)
    //       {
    //         relerror = 0.01;
    //       }
    //       m_FitData[k]    = (m_FitData[k]-SubFactor)/NormFactor;
    //       // m_FitErrData[k] = 0.05;
    //       m_FitErrData[k] = relerror*m_FitData[k];
    //       if(m_FitErrData[k]<0.002){
    //         m_FitErrData[k] = 0.01;
    //       }
    //       //   m_FitErrData[k] = 0.01*m_FitData[k];
    //       // }
    //       // else{
    //       //   m_FitErrData[k] = 0.01;
    //       // }
    //       std::cout<<m_FitData[k]<<"\t"<<NormFactor<<std::endl;
    //   }
    // SubFactor = 0;//m_FitData[NumDatPoints-1];
    // std::cout<<"SubFactor Data = "<<SubFactor<<std::endl;
    // NormFactor = m_FitData2[0]-SubFactor;
    // for(int k =0;k<NumDatPoints;k++){
    //       relerror = m_FitErrData2[k]/m_FitData2[k];
    //       if (m_FitData2[k] ==0)
    //       {
    //         relerror = 0.01;
    //       }
    //       m_FitData2[k]    = (m_FitData2[k]-SubFactor)/NormFactor;
    //       // m_FitErrData[k] = 0.05;
    //       m_FitErrData2[k] = relerror*m_FitData2[k];
    //       if(m_FitErrData2[k]<0.002){
    //         m_FitErrData2[k] = 0.01;
    //       }
    //       //   m_FitErrData[k] = 0.01*m_FitData[k];
    //       // }
    //       // else{
    //       //   m_FitErrData[k] = 0.01;
    //       // }
    //       std::cout<<m_FitData2[k]<<"\t"<<NormFactor<<std::endl;
    //   }
}


///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////



double CorrFunc_Integrand(double x, void * parameter){
  CorrFuncParameter * params = (CorrFuncParameter *)parameter;
  double r = (params->r);
  int n = (params->n);
  double Integrand = ( 1- gsl_sf_bessel_J0( (sqrt(2*x)*r))*pow( sqrt(1+pow(x,2)) - x , 2*n) ) / (x*sqrt(1+pow(x,2)));
  return(Integrand);
}

double CorrFunc(double r, double eta, double zeta, int n, double q1){
  double result, error;
  CorrFuncParameter Param_tmp {r , eta, zeta, n};
  gsl_function CorrFunc_Integrand_GSL;
  CorrFunc_Integrand_GSL.function = &CorrFunc_Integrand;
  gsl_integration_workspace * w = gsl_integration_workspace_alloc (100000);

  CorrFunc_Integrand_GSL.params = &Param_tmp;
  gsl_integration_qagiu(&CorrFunc_Integrand_GSL,0,0,1e-4, 100000, w, &result, &error);
  gsl_integration_workspace_free (w);
  return((2)*result);
}

double CalleApproximation_pre(double r, int n,double q1){
  if(n!=0&&(r*r/4/n)<=500){
    return( (4/pow(q1,2))*(M_EULER+log(r)+0.5*gsl_sf_expint_E1(r*r/4/n)) );
  }
  else{
    return( (4/pow(q1,2))*(M_EULER+log(r)) );
  }
}


void WriteArrayToBinFile(double * Array, int n, int m, char* FileName){
  fstream OutputFile;
  OutputFile.open(FileName,ios::out|ios::binary);
  for(int k = 0; k<n; k++){
    for(int l = 0; l<m; l++){
      OutputFile.write((char*)&(Array[k*m+l]),sizeof(double));
    }
  }
  OutputFile.close();
}

void ReadBinFile(double * Array, int n, int m, char* FileName){
  fstream OutputFile;
  OutputFile.open(FileName,ios::in|ios::binary);
  for(int k = 0; k<n; k++){
    for(int l = 0; l<m; l++){
      OutputFile.read((char*)&(Array[k*m+l]),sizeof(double));
    }
  }
  OutputFile.close();
}

void BackupFile(char* FileName){
  ifstream f(FileName);
  char* BackUpName;
  BackUpName = new char [200];
  sprintf(BackUpName,"#%s#",FileName);
  if(f.good()){
      std::cout<<"Back Off! I just backed up your file: "<<FileName<<"\nto "<<BackUpName<<std::endl;
      rename(FileName,BackUpName);
  }
}

double Gaussian(double Lz,double AvgLz, double sigma){
  return( (1/sigma)*exp( -pow((Lz-AvgLz),2)/pow(sigma,2)/2 ) );
}


double Hz_Integrand(double Lz,void* parameter){
  HzParameter * params = (HzParameter *)parameter;
  double AvgLz = params->AvgLz;
  double z = params->z;
  double sigma = params->sigma;
  double dspacing =params->dspacing;
  double integrand = Gaussian(Lz,AvgLz,sigma)*(Lz-z)*(1/dspacing); //!!! Lz removed
  return(integrand);
}

double Hz(double z, double AvgLz, double sigma, double D){
  double result,error;
  HzParameter Param_tmp {z , AvgLz, sigma,D};
  gsl_function Hz_Integrand_GSL;
  gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);
  Hz_Integrand_GSL.function = &Hz_Integrand;
  Hz_Integrand_GSL.params = &Param_tmp;
  gsl_integration_qag(&Hz_Integrand_GSL,z,1e6,0,1e-4, 10000,3, w, &result, &error);
  gsl_integration_workspace_free (w);
  return(result);
}


// double Gauss(double qr){
//   double sigmas = pow(0.004,2);
//   double output;
//   output = 1/sqrt(2*M_PI*sigmas) * exp(- 1.0/2.0*pow(qr,2)/sigmas ) ;
//   return( output);
// }
