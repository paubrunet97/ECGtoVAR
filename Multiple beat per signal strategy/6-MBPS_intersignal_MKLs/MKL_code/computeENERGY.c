/*========================================================*
 * computeENERGY.c
 *
 * The calling syntax from Matlab is:
 *
 *		gap = computeENERGY(K_tot_ALL,betas,A,W);
 *
 * This is a MEX file for MATLAB.
 *========================================================*/

#include "mex.h"
#include "matrix.h"

/* The computational routine */
void computeENERGY(double *K_tot_ALL, double *betas, double *A, double *W, double *Diag, int N, int NA, int m, double output[2], int i1, int i2)
{
    int r,l,i,j,c;

    double gap = 0.;
    double constr = 0.;
    
    double *tmp;
    tmp = (double*)mxMalloc(sizeof(double)*NA);

    double *tmpM;
    tmpM = (double*)mxMalloc(sizeof(double)*N*N);
    
    double tmpV;

    /* precompute K*betas*/
    for(i=0; i<N; i++)
    {
        for(r=0; r<N; r++)
        {
            tmpV = 0.;
            for(c=0; c<m; c++)
            {
                tmpV += K_tot_ALL[i + r*N + c*N*N] * betas[c];                
            }
            tmpM[i + r*N] = tmpV;
        }
    }
    
    for(i=i1; i<i2; i++)
    {
        for(j=(i+1); j<N; j++)
        {        
            double Wij = W[i + j*N];
            
            if(Wij != 0.)
            {
                for(r=0; r<NA; r++)
                {
                    tmpV = 0.;
                    for(l=0; l<N; l++)
                    {
                        tmpV += A[l + r*N] * (tmpM[i + l*N] - tmpM[j + l*N]);
                    }
                    tmp[r] = tmpV;
                }

                double gapij = 0.;
                for(r=0; r<NA; r++)
                {
                    gapij += tmp[r] * tmp[r];
                }

                gap += 2. * gapij * Wij;   
            }
        }
        
        for(r=0; r<NA; r++)
        {
            tmpV = 0.;
            for(l=0; l<N; l++)
            {
                tmpV += A[l + r*N] * tmpM[i + l*N];                 
            }
            tmp[r] = tmpV;
        }
        double constrii = 0.;
        for(r=0; r<NA; r++)
        {
            constrii += tmp[r] * tmp[r];
        }
        constr += constrii * Diag[i];            
    }
    mxFree(tmp);
    mxFree(tmpM);
    
    output[0] = gap;
    output[1] = constr;
}

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if(nrhs != 7)
    {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs", "7 inputs required.");
    }
    if(nlhs != 2)
    {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs", "2 outputs required.");
    }
    
    int nDimNum = mxGetNumberOfDimensions(prhs[0]);
    const mwSize *pDims = mxGetDimensions(prhs[0]);  

    int nDimNumA = mxGetNumberOfDimensions(prhs[2]);
    const mwSize *pDimsA = mxGetDimensions(prhs[2]);  
    int N = pDims[0];
    int NA = pDimsA[1];
    int m = pDims[2];
    
    double *K_tot_ALL = mxGetPr(prhs[0]); 
    double *betas = mxGetPr(prhs[1]); 
    double *A = mxGetPr(prhs[2]); 
    double *W = mxGetPr(prhs[3]); 
    double *Diag = mxGetPr(prhs[4]); 

    double *i1 = mxGetPr(prhs[5]);
    double *i2 = mxGetPr(prhs[6]);

    
    /*mexPrintf("ComputeSWB:Requested range: %d-%d\n",(int)i1[0],(int)i2[0]);*/
  
    double output[2];

    computeENERGY(K_tot_ALL,betas,A,W,Diag,N,NA,m,output,(int)i1[0],(int)i2[0]);
    plhs[0] = mxCreateDoubleScalar(output[0]);
    plhs[1] = mxCreateDoubleScalar(output[1]);
}

