#include <math.h>
#include <stdio.h>
#include <cmath>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_sf_airy.h>
#include <gsl/gsl_roots.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlin.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_dht.h>


#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <time.h>
#include <string.h>

#include "libs/diffuselibCuda.h"
#include "libs/CudaCoreFunctions.h"
#include "libs/GraphicsLib.h"

#include <boost/program_options.hpp>

namespace po = boost::program_options;




// void ReadDataFile(double*, double*, char*);
double wrapper(const gsl_vector*, void*);
Modus ParseModus(std::string);
// void GetCudaInfo();

DiffuseXRD Simulation(62.8319, 1, 1);




//
// struct data {
//   size_t n;
//   double * y;
//   double * sigma;
// };

struct datadiffuse {
  size_t n;
  // double * y;
  // double * y2;
  gsl_matrix * data;
  double * sigma;
  double * sigma2;
  double * qpar;
};


// int print_state (size_t, gsl_multifit_fdfsolver *);

// int
// expb_f (const gsl_vector * x, void *params,
//         gsl_vector * f)
// {
//   size_t n = ((struct data *)params)->n;
//   double *y = ((struct data *)params)->y;
//   double *sigma = ((struct data *) params)->sigma;
//
//   double A = gsl_vector_get (x, 0);
//   double lambda = gsl_vector_get (x, 1);
//   double b = 0;
//
//   size_t i;
//
//   for (i = 0; i < n; i++)
//     {
//       /* Model Yi = A * exp(-lambda * i) + b */
//       double t = i;
//       double Yi = A * exp (-lambda * t) + b;
//       gsl_vector_set (f, i, (Yi - y[i])/sigma[i]);
//     }
//
//   return GSL_SUCCESS;
// }


int
xi_f (const gsl_vector * x, void *params,
        gsl_vector * f)
{
  size_t n = ((struct datadiffuse *)params)->n;
  // double *y = ((struct datadiffuse *)params)->y;
  // double *y2 = ((struct datadiffuse *)params)->y2;
  gsl_matrix *data = ((struct datadiffuse *)params)->data;
  double *sigma = ((struct datadiffuse *) params)->sigma;
  double *sigma2 = ((struct datadiffuse *) params)->sigma2;
  double *qpardata = ((struct datadiffuse *) params)->qpar;

  gsl_vector* xiS1 = gsl_vector_calloc(n);
  gsl_vector* xiS2 = gsl_vector_calloc(n);

  double eta = gsl_vector_get (x, 0);
  double zeta = gsl_vector_get (x, 1);

  double qz = Simulation.GetQzStart();

  double Normalisation;
  Simulation.SetEta(eta);
  Simulation.SetZeta(zeta);

  qz = Simulation.GetQzStart();
  // std::cout<<qz<<std::endl;
  Simulation.ConvertUnitsCale();
  Simulation.PreProcessNSummation(qz);
  Simulation.HankelTransformation(qz);
  Simulation.PostProcessHankel();
  double SubFactor = 0;//Simulation.InterpolateQrScan(qpardata[n-1]);

  Normalisation = Simulation.InterpolateQrScan(qpardata[0])-SubFactor;
  size_t i;

  for (i = 0; i < n; i++)
  {
      double Yi = (Simulation.InterpolateQrScan(qpardata[i])-SubFactor)/Normalisation;
      gsl_vector_set (xiS1, i, (Yi - gsl_matrix_get(data,0,i))/sigma[i]); //!!!!
  }

  // qz = 0.3306;
  qz = Simulation.GetQzStart2();
  // std::cout<<qz<<std::endl;
  Simulation.ConvertUnitsCale();
  Simulation.PreProcessNSummation(qz);
  Simulation.HankelTransformation(qz);
  Simulation.PostProcessHankel();
  SubFactor = 0;//Simulation.InterpolateQrScan(qpardata[n-1]);

  Normalisation = Simulation.InterpolateQrScan(qpardata[0])-SubFactor;
  // size_t i;

  for (i = 0; i < n; i++)
  {
      double Yi = (Simulation.InterpolateQrScan(qpardata[i])-SubFactor)/Normalisation;
      gsl_vector_set (xiS2, i, (Yi - gsl_matrix_get(data,1,i))/sigma2[i]); //!!!!
  }

  for (i = 0; i < n; i++)
  {
      gsl_vector_set (f, i, gsl_vector_get(xiS1, i)+ gsl_vector_get(xiS2, i));
  }

  gsl_vector_free(xiS1);
  gsl_vector_free(xiS2);
  return GSL_SUCCESS;
}


int
xi_df (const gsl_vector * x, void *params,
         gsl_matrix * J)
{
  size_t n = ((struct datadiffuse *)params)->n;
  // double *sigma = ((struct datadiffuse *) params)->sigma;
  // double *sigma = ((struct data *) params)->sigma;

    double eta = gsl_vector_get (x, 0);
    double zeta = gsl_vector_get (x, 1);
    gsl_vector* xtmp = gsl_vector_calloc(2);
    gsl_vector* fzp = gsl_vector_calloc(n);
    gsl_vector* fzm = gsl_vector_calloc(n);
    gsl_vector* ftmp = gsl_vector_calloc(n);
    gsl_vector* fep = gsl_vector_calloc(n);
    gsl_vector* fem = gsl_vector_calloc(n);


    gsl_vector_memcpy(xtmp, x);
    // xtmp = x;

    // double Normalisation;
    // double qz = Simulation.GetQzStart();

    gsl_vector_set(xtmp,0,eta+0.05);
    xi_f (xtmp, params, fep);
    gsl_vector_set(xtmp,0,eta-0.05);
    xi_f (xtmp, params, fem);
    gsl_vector_set(xtmp,0,eta);

    gsl_vector_set(xtmp,1,zeta+3);
    xi_f (xtmp, params, fzp);
    gsl_vector_set(xtmp,1,zeta-3);
    xi_f (xtmp, params, fzm);

    gsl_vector_set(xtmp,1,zeta);
    xi_f (xtmp, params, ftmp);

  size_t i;

  // std::cout<<"fzpi"<<"\t"<<"fzmi"<<"\t"<<"fepi"<<"\t"<<"femi"<<"\t"<<"ftmpi"<<"\t"<<"dfzi"<<"\t"<<"dfei"<<std::endl;
  for (i = 0; i < n; i++)
    {
      double fzpi = gsl_vector_get (fzp, i);
      double fzmi = gsl_vector_get (fzm, i);
      double fepi = gsl_vector_get (fep, i);
      double femi = gsl_vector_get (fem, i);
      double ftmpi = gsl_vector_get (ftmp, i);
      double dfzi = ((fzpi-ftmpi)/1+(ftmpi-fzmi)/1)/2; //!!!
      double dfei = ((fepi-ftmpi)/.01+(ftmpi-femi)/.01)/2; //!!!
      gsl_matrix_set (J, i, 0, dfei/2);
      gsl_matrix_set (J, i, 1, dfzi/2);
      // std::cout<<fzpi<<"\t"<<fzmi<<"\t"<<fepi<<"\t"<<femi<<"\t"<<ftmpi<<"\t"<<dfzi<<"\t"<<dfei<<std::endl;
      // gsl_matrix_set (J, i, 2, 1/s);

    }

    gsl_vector_free(fzp);
    gsl_vector_free(fzm);
    gsl_vector_free(ftmp);
    gsl_vector_free(fep);
    gsl_vector_free(fem);
    gsl_vector_free(xtmp);
  return GSL_SUCCESS;
}

int
xi_fdf (const gsl_vector * x, void *params,
          gsl_vector * f, gsl_matrix * J)
{
  xi_f (x, params, f);
  xi_df (x, params, J);

  return GSL_SUCCESS;
}


// int
// xi_corf (const gsl_vector * x, void *params,
//         gsl_vector * f)
// {
//   size_t n = ((struct datadiffuse *)params)->n;
//   double *y = ((struct datadiffuse *)params)->y;
//   double *sigma = ((struct datadiffuse *) params)->sigma;
//   double *rdata = ((struct datadiffuse *) params)->qpar;
//
//   double eta = gsl_vector_get (x, 0);
//   double zeta = gsl_vector_get (x, 1);
//
//   double qz = Simulation.GetQzStart();
//
//   // double Normalisation;
//   Simulation.SetEta(eta);
//   Simulation.SetZeta(zeta);
//   Simulation.ConvertUnitsCale();
//
//   size_t i;
//
//   for (i = 0; i < n; i++)
//     {
//       // /* Model Yi = A * exp(-lambda * i) + b */
//       double Yi = (Simulation.CorrFuncApprox(rdata[i],0));
//       gsl_vector_set (f, i, (Yi - y[i])/sigma[i]); //!!!!
//     }
//   return GSL_SUCCESS;
// }
//
//
// int
// xi_dcorf (const gsl_vector * x, void *params,
//          gsl_matrix * J)
// {
//   size_t n = ((struct datadiffuse *)params)->n;
//
//     double eta = gsl_vector_get (x, 0);
//     double zeta = gsl_vector_get (x, 1);
//     gsl_vector* xtmp = gsl_vector_calloc(2);
//     gsl_vector* fzp = gsl_vector_calloc(n);
//     gsl_vector* fzm = gsl_vector_calloc(n);
//     gsl_vector* ftmp = gsl_vector_calloc(n);
//     gsl_vector* fep = gsl_vector_calloc(n);
//     gsl_vector* fem = gsl_vector_calloc(n);
//
//
//     gsl_vector_memcpy(xtmp, x);
//     gsl_vector_set(xtmp,0,eta+0.01);
//     xi_f (xtmp, params, fep);
//     gsl_vector_set(xtmp,0,eta-0.01);
//     xi_f (xtmp, params, fem);
//     gsl_vector_set(xtmp,0,eta);
//
//     gsl_vector_set(xtmp,1,zeta+1);
//     xi_f (xtmp, params, fzp);
//     gsl_vector_set(xtmp,1,zeta-1);
//     xi_f (xtmp, params, fzm);
//
//     gsl_vector_set(xtmp,1,zeta);
//     xi_f (xtmp, params, ftmp);
//
//   size_t i;
//
//   for (i = 0; i < n; i++)
//     {
//       double fzpi = gsl_vector_get (fzp, i);
//       double fzmi = gsl_vector_get (fzm, i);
//       double fepi = gsl_vector_get (fep, i);
//       double femi = gsl_vector_get (fem, i);
//       double ftmpi = gsl_vector_get (ftmp, i);
//       double dfzi = ((fzpi-ftmpi)/1+(ftmpi-fzmi)/1)/2; //!!!
//       double dfei = ((fepi-ftmpi)/.01+(ftmpi-femi)/.01)/2; //!!!
//       gsl_matrix_set (J, i, 0, dfei/2);
//       gsl_matrix_set (J, i, 1, dfzi/2);
//     }
//
//     gsl_vector_free(fzp);
//     gsl_vector_free(fzm);
//     gsl_vector_free(ftmp);
//     gsl_vector_free(fep);
//     gsl_vector_free(fem);
//     gsl_vector_free(xtmp);
//   return GSL_SUCCESS;
// }
//
// int
// xi_corfdcorf (const gsl_vector * x, void *params,
//           gsl_vector * f, gsl_matrix * J)
// {
//   xi_corf (x, params, f);
//   xi_dcorf (x, params, J);
//
//   return GSL_SUCCESS;
// }



int main(int argc, char const *argv[]) {

  char* FileName;
  char* FileNameBase;
  char* InputFile;
  FileName = new char [200];
  FileNameBase = new char [200];
  InputFile = new char [200];
  Modus SysModus;

  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "produce help message")
      ("mode,m", po::value<std::string>(), "Set Modus: pccf, cd, fitd")
      ("zeta,z", po::value<double>(), "set initial zeta")
      ("eta,e", po::value<double>(), "set initial eta")
      ("q1,q", po::value<double>(), "set d-spcing in q")
      ("qzb", po::value<double>(), "initial qz")
      ("qze", po::value<double>(), "final qz")
      ("qzs", po::value<double>(), "qz step size")
      ("Lz", po::value<double>()->default_value(1e4), "Average z domain size")
      ("sz", po::value<double>()->default_value(3.3e3), "Average z domain size")
      ("Lr", po::value<double>()->default_value(1e4), "Average z domain size")
      ("sr", po::value<double>()->default_value(3.3e3), "Average z domain size")
      ("output,o", po::value<std::string>()->default_value("UNKNOWN"), "Provide Outputfilename without extension")
      ("file,f", po::value<std::string>(), "Provide InputFile without extension")
  ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  PrintLogo();

  if (vm.count("help")) {
      std::cout<<"Usage: "<<argv[0]<<" [options]"<<std::endl;
      std::cout << desc << "\n";
      return 1;
  }

  if (vm.count("mode")) {
      std::cout << "Mode Selected: "<<vm["mode"].as<std::string>()<<"\n";
      SysModus = ParseModus(vm["mode"].as<std::string>());
  }
 else {
      std::cout << "Mode must be selected\n";
      return(0);
  }

  if(SysModus==e_fitd){
    if (!vm.count("zeta")||!vm.count("eta")||!vm.count("file")) {
        std::cout << "Not enough arguments. The program requires to set zeta, eta, q1, qzb and file\n";
        return(0);
    }
    std::cout<<"Selected Option: Fit Diffuse Profile\nInitialize System...\n"<<std::endl;
    Simulation.SetZeta(roundf(vm["zeta"].as<double>()*10000)/10000);
    Simulation.SetEta(roundf(vm["eta"].as<double>()*10000)/10000);
    Simulation.SetAvgLr(vm["Lr"].as<double>());
    Simulation.SetSigmaR(vm["sr"].as<double>());
    Simulation.SetAvgLz(vm["Lz"].as<double>());
    Simulation.SetSigmaZ(vm["sz"].as<double>());
    strcpy(FileNameBase,vm["output"].as<std::string>().c_str());
    strcpy(InputFile,vm["file"].as<std::string>().c_str());
    Simulation.ReadDataFile(InputFile);
  }
  else if (SysModus==e_cd){
    if (!vm.count("zeta")||!vm.count("eta")||!vm.count("q1")||!vm.count("qzb")) {
        std::cout << "Not enough arguments. The program requires to set zeta, eta, q1, qzb\n";
        return(0);
    }
    std::cout<<"Selected Option: Calculate Line Scan\nInitialize System...\n"<<std::endl;
    Simulation.SetZeta(roundf(vm["zeta"].as<double>()*10000)/10000);
    Simulation.SetEta(roundf(vm["eta"].as<double>()*10000)/10000);
    Simulation.SetQ1(roundf(vm["q1"].as<double>()*10000)/10000);
    Simulation.SetQzStart(roundf(vm["qzb"].as<double>()*10000)/10000);
    strcpy(FileNameBase,vm["output"].as<std::string>().c_str());
  }
  else if (SysModus==e_2Ds){
    if (!vm.count("zeta")||!vm.count("eta")||!vm.count("q1")||!vm.count("qze")||!vm.count("qzb")||!vm.count("qzs")) {
        std::cout << "Not enough arguments. The program requires to set zeta, eta, q1, qzb, qze, qzs\n";
        return(0);
    }
    std::cout<<"Selected Option: Calculate 2D Scan\nInitialize System...\n"<<std::endl;
    Simulation.SetZeta(roundf(vm["zeta"].as<double>()*1000)/1000);
    Simulation.SetEta(roundf(vm["eta"].as<double>()*1000)/1000);
    Simulation.SetQ1(roundf(vm["q1"].as<double>()*10000)/10000);
    Simulation.SetQzStart(roundf(vm["qzb"].as<double>()*10000)/10000);
    Simulation.SetQzStop(roundf(vm["qze"].as<double>()*10000)/10000);
    Simulation.SetQzStep(roundf(vm["qzs"].as<double>()*100000)/100000);
    Simulation.SetAvgLr(vm["Lr"].as<double>());
    Simulation.SetSigmaR(vm["sr"].as<double>());
    Simulation.SetAvgLz(vm["Lz"].as<double>());
    Simulation.SetSigmaZ(vm["sz"].as<double>());
    strcpy(FileNameBase,vm["output"].as<std::string>().c_str());
  }
  else if (SysModus==e_pccf){
      if (!vm.count("q1")) {
          std::cout << "Not enough arguments. This modus requires to set q1\n";
          return(0);
      }
      std::cout<<"Selected Option: Pre-Calculate Correlationfunction\nInitialize System...\n"<<std::endl;
      Simulation.SetZeta(1);
      Simulation.SetEta(1);
      Simulation.SetQ1(roundf(vm["q1"].as<double>()*100)/100);
      strcpy(FileNameBase,vm["output"].as<std::string>().c_str());
  }

  else if (SysModus==e_convc){
    if (!vm.count("zeta")||!vm.count("eta")||!vm.count("q1")) {
        std::cout << "Not enough arguments. The program requires to set zeta, eta\n";
        return(0);
    }
    std::cout<<"Selected Option: Calculate 2D Scan\nInitialize System...\n"<<std::endl;
    Simulation.SetZeta(roundf(vm["zeta"].as<double>()*10000)/10000);
    Simulation.SetEta(roundf(vm["eta"].as<double>()*10000)/10000);
    Simulation.SetQ1(roundf(vm["q1"].as<double>()*10000)/10000);
  }

  else if (SysModus==e_h){
    std::cout<<"Usage: "<<argv[0]<<" [options]"<<std::endl;
    std::cout << desc << "\n";
  }
  else{
      std::cout<<"Usage: "<<argv[0]<<" [options]"<<std::endl;
      std::cout << desc << "\n";
      return 0;
  }
  Simulation.ConvertUnitsCale();
  Simulation.ListSimParameters();
  PrintLine();
  GetCudaInfo();
  PrintLine();

 if (SysModus==e_pccf){
    PrintProcessInfoHeader();
    Simulation.PreProcessCorrFunc(CNMAX);
    PrintLine();
    sprintf(FileName,"%s.cf",FileNameBase);
    BackupFile(FileName);
    WriteArrayToBinFile(Simulation.m_CorrFuncTable,Simulation.m_Rlength+1,CNMAX+1,FileName);
    PrintLine();
    std::cout<<"Calculation finished successfully! Data are stored in: "<<FileName<<"\n\n";
 }

 else if(SysModus==e_fitd){

      PrintProcessInfoHeader();
       // Simulation.PreProcessCorrFunc(CNMAX);
       Simulation.LoadCorrFunc("/opt/genapp/XDSAnalysis/bin/CorrelationFunctionTable_08112022.cf");
       PrintProcessInfoHeader();
       Simulation.PreProcessHr();
       Simulation.PreProcessHz(CNMAX);
       std::cout<<std::endl;
       std::cout<<std::endl;
       Simulation.PreProcessNSummation(Simulation.GetQzStart());
       Simulation.HankelTransformation(Simulation.GetQzStart());
       Simulation.PostProcessHankel();
       std::cout<<std::endl;
       PrintLine();
       PrintFitTableHeader();
       const gsl_multifit_fdfsolver_type *T;
       gsl_multifit_fdfsolver *s;

       int status;
       size_t i, iter = 0;

       const size_t n = Simulation.GetNumDataLines();
       const size_t p = 2;

       gsl_matrix *covar = gsl_matrix_alloc (p, p);
       gsl_matrix *J = gsl_matrix_alloc (n, p);

       double y[n],y2[n], sigma[n],sigma2[n], qpardata[n];
       gsl_matrix * FitData;

       gsl_matrix_memcpy(FitData, Simulation.m_FitData_comb);
       // std::cout<<"here"<<n<<std::endl;
       // struct data d = { n, y, sigma};
       struct datadiffuse d = { n, FitData, sigma,sigma2,qpardata};

       gsl_multifit_function_fdf f;

       // double x_init[3] = { 1.0, 0.0, 0.0 };
       double x_init[2] = { Simulation.GetEta(), Simulation.GetZeta()};

       gsl_vector_view x = gsl_vector_view_array (x_init, p);



       f.f = &xi_f;
       f.df = &xi_df;
       f.fdf = &xi_fdf;
       f.n = n;
       f.p = p;
       f.params = &d;

       /* This is the data to be fitted */

       for (i = 0; i < n; i++)
         {
           // y[i] = Simulation.m_FitData[i];
           // y2[i] = Simulation.m_FitData2[i];
           qpardata[i] = Simulation.m_FitQr[i];
           sigma[i] = Simulation.m_FitErrData[i];
           sigma2[i] = Simulation.m_FitErrData2[i];
         };

       gsl_vector* StartVals;
       StartVals = gsl_vector_alloc(2);
       // Output = gsl_vector_alloc(n);

       gsl_vector_set(StartVals,0,.1);
       gsl_vector_set(StartVals,1,80);

       // std::cout<<"here"<<std::endl;
       // xi_f(StartVals,&d,Output);


        // for (i = 0; i < n; i++)
        //   {
        //     printf("XiSquare %g \n", gsl_vector_get(Output,i));
        //   };

          // std::cout<<"here end"<<std::endl;
       T = gsl_multifit_fdfsolver_lmsder;
       s = gsl_multifit_fdfsolver_alloc (T, n, p);
       gsl_multifit_fdfsolver_set (s, &f, &x.vector);

       // print_state (iter, s);

       do
         {
           iter++;
           status = gsl_multifit_fdfsolver_iterate (s);
           // std::cout<<"\n\n"<<std::endl;
           // printf ("status = %s\n", gsl_strerror (status));

           // print_state (iter, s);
           PrintFitTableData(iter,gsl_vector_get (s->x, 0),gsl_vector_get (s->x, 1),gsl_blas_dnrm2 (s->f),1);
           Simulation.SetXI2(gsl_blas_dnrm2 (s->f));
           if (status)
             break;

           status = gsl_multifit_test_delta (s->dx, s->x,
                                             5e-4, 5e-4);
         }
       while (status == GSL_CONTINUE && iter < 50);
       gsl_multifit_fdfsolver_jac(s,J);
       gsl_multifit_covar (J, 0.0, covar);
       // for (i = 0; i < n; i++)
       //   {
       //     std::cout<<gsl_matrix_get(J,i,0)<<"\t"<<gsl_matrix_get(J,i,1)<<std::endl;
       //   }
       //   std::cout<<"temp"<<covar->size1<<std::endl;
       //   for (i = 0; i < 2; i++)
       //     {
       //       std::cout<<gsl_matrix_get(covar,i,0)<<"\t"<<gsl_matrix_get(covar,i,1)<<std::endl;
       //     }
       // gsl_matrix_fprintf (stdout, covar, "%g");

     #define FIT(i) gsl_vector_get(s->x, i)
     #define ERR(i) sqrt(gsl_matrix_get(covar,i,i))

     // printf("Eta    = %.5f +/- %.5f\n", FIT(0), ERR(0));
     // printf("Zeta   = %.5f +/- %.5f\n", FIT(1), ERR(1));

     Simulation.SetDEta(ERR(0));
     Simulation.SetDZeta(ERR(1));
     Simulation.ConvertUnitsCale();
     // printf("b      = %.5f +/- %.5f\n", FIT(2), ERR(2));

     // printf ("status = %s\n", gsl_strerror (status));

     gsl_multifit_fdfsolver_free (s);
   // return 0;

    // gsl_vector_free(x);
    // gsl_vector_free(ss);
    // gsl_multimin_fminimizer_free (s);
    double qztemp = Simulation.GetQzStart();
    std::cout<<qztemp<<endl;
    PrintLine();
    Simulation.ListSimParameters();
    Simulation.PreProcessNSummation(qztemp);
    Simulation.HankelTransformation(qztemp);
    Simulation.PostProcessHankel();
    sprintf(FileName,"%s_fitted.fit",FileNameBase);
    BackupFile(FileName);
    Simulation.WriteFitData(FileName);

    qztemp = Simulation.GetQzStart2();
    std::cout<<qztemp<<endl;
    Simulation.ListSimParameters();
    Simulation.PreProcessNSummation(qztemp);
    Simulation.HankelTransformation(qztemp);
    Simulation.PostProcessHankel();
    sprintf(FileName,"%s_fitted2.fit",FileNameBase);
    BackupFile(FileName);
    Simulation.WriteFitData2(FileName);
    sprintf(FileName,"%s_fitted.qr",FileNameBase);
    Simulation.NormalizeQrScan();
    BackupFile(FileName);
    Simulation.WriteQrVector(FileName);
    PrintLine();
    std::cout<<"Calculation finished successful! Data are stored in: "<<FileName<<"\n"<<std::endl;

   }

   else if (SysModus == e_cd){

     PrintProcessInfoHeader();
     Simulation.PreProcessCorrFunc(CNMAX);
     PrintProcessInfoHeader();
     Simulation.PreProcessHr();
     Simulation.PreProcessHz(CNMAX);
     std::cout<<std::endl;
     Simulation.PreProcessNSummation(Simulation.GetQzStart());
     Simulation.HankelTransformation(Simulation.GetQzStart());
     Simulation.PostProcessHankel();
     std::cout<<std::endl;
     PrintLine();
     sprintf(FileName,"%s.qr",FileNameBase);
     BackupFile(FileName);
     Simulation.WriteQrVector(FileName);
     PrintLine();
     std::cout<<"Calculation finished successfully! Data are stored in: "<<FileName<<"\n\n";
   }
 //
   else if (SysModus == e_2Ds){
        Simulation.InitiateQzScan();
        PrintProcessInfoHeader();
        // Simulation.PreProcessCorrFunc(CNMAX);
        Simulation.LoadCorrFunc("CorrelationFunctionTable_08112022.cf");
        PrintProcessInfoHeader();

        Simulation.PreProcessHr();
        Simulation.PreProcessHz(CNMAX);
        std::cout<<std::endl;
        Simulation.QzScan();
        std::cout<<std::endl;
        PrintLine();
        sprintf(FileName,"%s.simg",FileNameBase);
        BackupFile(FileName);
        Simulation.Save2DScan(FileName);
        PrintLine();
        std::cout<<"Calculation finished successfully! Data are stored in: "<<FileName<<"\n\n";
   }

  return 0;
}


double wrapper(const gsl_vector* x, void* Paratemp){
  double value;
  value = Simulation.XiSquareCuda(x,Paratemp);
  return(value);
}


Modus ParseModus(std::string Input){
  if(Input.compare("pccf") == 0){
    // std::cout<<"got it"<<std::endl;
    return(e_pccf);
  }
  else if(Input.compare("cd")   == 0 ){
    return(e_cd);
  }
  else if(Input.compare("fitd")   == 0 ){
    return(e_fitd);
  }
  else if(Input.compare("fitc")   == 0 ){
    return(e_fitc);
  }
  else if(Input.compare("2ds")   == 0 ){
    return(e_2Ds);
  }
  else if(Input.compare("cc")   == 0 ){
    return(e_convc);
  }
  else if(Input.compare("h")  == 0 ){
    return(e_h);
  }
  else{
    return(e_UNKNOWN);
  }
}


//
// int
// print_state (size_t iter, gsl_multifit_fdfsolver * s)
// {
//   printf ("iter: %3u x = % 15.8f % 15.8f "
//           "|f(x)| = %g\n",
//           iter,
//           gsl_vector_get (s->x, 0),
//           gsl_vector_get (s->x, 1),
//           // gsl_vector_get (s->x, 2),
//           gsl_blas_dnrm2 (s->f));
//           return(0);
// }
