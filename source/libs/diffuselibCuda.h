#include "diffusetoolbox.h"
#include "CudaCoreFunctions.h"
#include <cmath>

#ifndef __DIFFUSE_INCL__
#define __DIFFUSE_INCL__

#define CNMAX 1000
#define QRuns  100


using namespace std;

enum Modus{
  e_pccf    = 0,
  e_cd      = 1,
  e_fitd    = 2,
  e_fitc    = 3,
  e_2Ds     = 4,
  e_convc   = 5,
  e_h       = 6,
  e_UNKNOWN = 7,
};


struct CorrFuncParameter{double r; double eta; double zeta; int n;};
struct HzParameter{double z; double AvgLz; double sigma; double dspacing;};

double CorrFunc_Integrand(double x, void * parameter);
double CalleApproximation_pre(double, int ,double);
double CorrFunc(double, double, double, int, double);
void WriteArrayToBinFile(double*, int, int, char*);
void ReadBinFile(double*, int, int, char*);
void BackupFile(char*);

double Gaussian(double,double,double);
double Hz_Integrand(double,void*);
double Hz(double, double, double,double);
double Gauss(double);


class DiffuseXRD{
private:
  double m_qzstart;
  double m_qzstart2;
  double m_qzstop;
  double m_qzstep;
  double m_Dspacing;
  double m_q1;
  double m_zeta;
  double m_eta;
  double m_dzeta;
  double m_deta;
  double m_Kc;
  double m_B;
  double m_dKc;
  double m_dB;
  double m_StartR;
  double m_StopR;
  double m_StepSize;
  int    m_RlengthHr;
  int    m_RIntMax;
  int    m_RIntStep;
  int    m_QrIntervals;
  int    m_IntIntervals;
  int    m_NHankelTransform;
  double m_AvgLz;
  double m_SigmaZ;
  double m_AvgLr;
  double m_SigmaR;
  double* m_r;
  double* m_rhr;
  double *m_SummationTable;
  double *m_HzTable;
  double *m_HrTable;
  double *m_HankelTransform;
  double (*m_2DData)[2*QRuns];
  int m_NumFitDatLines;
  int m_NumFitDatSets;
  double m_xi2;

public:
  int    m_Rlength;
  double *m_CorrFuncTable;
  double *m_FitData;
  double *m_FitQr;
  double *m_FitErrData;
  double *m_FitData2;
  double *m_FitQr2;
  gsl_matrix *m_FitData_comb;
  gsl_matrix *m_FitErrData_comb;
  gsl_vector *m_FitData_qzVals;
  // gsl_vector *m_FitData_qparVals;
  double *m_FitErrData2;
  double *m_QrScan;
  DiffuseXRD(double Dspacing, double zeta, double eta){
    m_Dspacing = Dspacing;
    m_q1       = 2*M_PI/m_Dspacing;
    m_zeta     = zeta;
    m_eta      = eta;
    m_Kc     = 0;
    m_B      = 0;
    m_dzeta     = 0;
    m_deta      = 0;
    m_dKc     = 0;
    m_dB      = 0;
    m_StartR   = -4;
    m_StepSize = 0.001;
    m_StopR    = 6;
    m_Rlength  = (m_StopR - m_StartR)/m_StepSize;
    m_RIntMax  = 1e4;
    m_RIntStep = .1;
    m_QrIntervals = 100;
    m_IntIntervals = 1000;
    m_qzstart = 0;
    m_qzstop = 0;
    m_qzstep = 0;
    m_AvgLz = 1e4;
    m_SigmaZ = 3.3e3;
    m_AvgLr = 1e4;
    m_SigmaR = 3.3e3;
    m_NHankelTransform = m_QrIntervals*m_IntIntervals;
    m_r = new double [m_Rlength+1];
    m_CorrFuncTable = new double [(m_Rlength+1)*(CNMAX+1)];
    m_SummationTable = new double [(m_Rlength+1)];
    CreateLogScaleVec(m_r, m_StartR,m_StopR,m_StepSize);
    m_HzTable = new double [CNMAX+1];
    m_RlengthHr  = (6 - m_StartR)/m_StepSize;
    m_rhr = new double [m_RlengthHr+1];
    m_HrTable = new double [m_RlengthHr+1];
    m_HankelTransform = new double [m_NHankelTransform+1];
    m_QrScan = new double [m_QrIntervals];
    CreateLogScaleVec(m_rhr, m_StartR,6,m_StepSize);
  }

  ~DiffuseXRD(){
    delete m_r;
    delete m_CorrFuncTable;
    delete m_rhr;
    delete m_HzTable;
    delete m_HrTable;
    delete m_SummationTable;
    delete m_HankelTransform;
    delete m_QrScan;
  }

  void LoadFitData();
  void PostProcessHankel();
  void PreProcessNSummation(double);
  void HankelTransformation(double);
  void PreProcessCorrFunc(int);
  void LoadCorrFunc(char*);
  void PreProcessHz(int);
  void PreProcessHr();
  void SetZeta(double);
  void SetEta(double);
  void SetDZeta(double);
  void SetDEta(double);
  void SetQ1(double);
  void SetQzStart(double);
  void SetQzStop(double);
  void SetQzStep(double);
  void SetAvgLr(double);
  void SetSigmaR(double);
  void SetAvgLz(double);
  void SetSigmaZ(double);
  void SetXI2(double);
  double GetZeta();
  double GetEta();
  double GetQ1();
  double GetQzStart();
  double GetQzStart2();
  double GetQzStop();
  double GetQzStep();
  double GetAvgLr();
  double GetSigmaR();
  double GetAvgLz();
  double GetSigmaZ();
  double GetNumDataLines();
  double GetNumDataSets();
  double GetXI2();
  double XiSquareCuda(const gsl_vector*, void*);
  void WriteQrVector(char*);
  void WriteFitData(char*);
  void WriteFitData2(char*);
  void WriteHeader(char*,int,int);
  void QzScan();
  void InitiateQzScan();
  void Save2DScan(char*);
  void ListSimParameters();
  double InterpolateQrScan(double);
  void NormalizeQrScan();
  void ReadDataFile(char* FileName);
  void ConvertUnitsCale();
  double CorrFuncApprox(double, int);
};


#endif
