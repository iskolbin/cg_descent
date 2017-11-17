#ifndef CG_DESCENT_H_
#define CG_DESCENT_H_

/*
  cg_descent.h - v6.8.4 - unconstrained nonlinear optimization single header lib

  author: Ilya Kolbin (iskolbin@gmail.com)
  url: github.com/iskolbin/cg_descent

  This is single-header port of CG_DESCENT 6.8 by William W. Hager and
  Hongchao Zhang taken from authors site 
  http://users.clas.ufl.edu/hager/papers/Software/

	Original code is licensed under GPLv2, see details below.

	To use this library you need to define implementation macro once in your code
		#define CG_DESCENT_IMPLEMENTATION
	
	Main deviations from the original library is the ability to pass custom
	argument in the val/grad/valgrad functions. Also you can redefine types
	of using floats and integers (double and size_t by default).

	Also includes some minor fixes to stop compiler from complaining about
	uninitialized and unused variables.
*/

/* =========================================================================
   ============================ CG_DESCENT =================================
   =========================================================================
       ________________________________________________________________
      |      A conjugate gradient method with guaranteed descent       |
      |             C-code Version 1.1  (October 6, 2005)              |
      |                    Version 1.2  (November 14, 2005)            |
      |                    Version 2.0  (September 23, 2007)           |
      |                    Version 3.0  (May 18, 2008)                 |
      |                    Version 4.0  (March 28, 2011)               |
      |                    Version 4.1  (April 8, 2011)                |
      |                    Version 4.2  (April 14, 2011)               |
      |                    Version 5.0  (May 1, 2011)                  |
      |                    Version 5.1  (January 31, 2012)             |
      |                    Version 5.2  (April 17, 2012)               |
      |                    Version 5.3  (May 18, 2012)                 |
      |                    Version 6.0  (November 6, 2012)             |
      |                    Version 6.1  (January 27, 2013)             |
      |                    Version 6.2  (February 2, 2013)             |
      |                    Version 6.3  (April 21, 2013)               |
      |                    Version 6.4  (April 29, 2013)               |
      |                    Version 6.5  (April 30, 2013)               |
      |                    Version 6.6  (May 28, 2013)                 |
      |                    Version 6.7  (April 7, 2014)                |
      |                    Version 6.8  (March 7, 2015)                |
      |                                                                |
      |           William W. Hager    and   Hongchao Zhang             |
      |          hager@math.ufl.edu       hozhang@math.lsu.edu         |
      |                   Department of Mathematics                    |
      |                     University of Florida                      |
      |                 Gainesville, Florida 32611 USA                 |
      |                      352-392-0281 x 244                        |
      |                                                                |
      |                 Copyright by William W. Hager                  |
      |                                                                |
      |          http://www.math.ufl.edu/~hager/papers/CG              |
      |                                                                |
      |  Disclaimer: The views expressed are those of the authors and  |
      |              do not reflect the official policy or position of |
      |              the Department of Defense or the U.S. Government. |
      |                                                                |
      |      Approved for Public Release, Distribution Unlimited       |
      |________________________________________________________________|
       ________________________________________________________________
      |This program is free software; you can redistribute it and/or   |
      |modify it under the terms of the GNU General Public License as  |
      |published by the Free Software Foundation; either version 2 of  |
      |the License, or (at your option) any later version.             |
      |This program is distributed in the hope that it will be useful, |
      |but WITHOUT ANY WARRANTY; without even the implied warranty of  |
      |MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the   |
      |GNU General Public License for more details.                    |
      |                                                                |
      |You should have received a copy of the GNU General Public       |
      |License along with this program; if not, write to the Free      |
      |Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, |
      |MA  02110-1301  USA                                             |
      |________________________________________________________________|

      References:
      1. W. W. Hager and H. Zhang, A new conjugate gradient method
         with guaranteed descent and an efficient line search,
         SIAM Journal on Optimization, 16 (2005), 170-192.
      2. W. W. Hager and H. Zhang, Algorithm 851: CG_DESCENT,
         A conjugate gradient method with guaranteed descent,
         ACM Transactions on Mathematical Software, 32 (2006), 113-137.
      3. W. W. Hager and H. Zhang, A survey of nonlinear conjugate gradient
         methods, Pacific Journal of Optimization, 2 (2006), pp. 35-58.
      4. W. W. Hager and H. Zhang, Limited memory conjugate gradients,
         SIAM Journal on Optimization, 23 (2013), 2150-2168. */

#include <math.h>
#include <limits.h>
#include <float.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef CG_STATIC
#define CG_API static
#else
#define CG_API extern
#endif

#ifndef CG_USE_CUSTOM
#define CG_CUSTOM
#define CG_CUSTOM_ARGUMENT(Com)
#define CG_CUSTOM_STRUCT
#define CG_INIT_CUSTOM(Com)
#endif

typedef enum cg_boolean {
    CG_FALSE = 0,
    CG_TRUE = 1
} cg_boolean;

#ifndef CG_FLOAT
#define CG_FLOAT double
#define CG_FLOAT_INF DBL_MAX
#else
#ifndef CG_FLOAT_INF
#error "CG_FLOAT is redefined, you also need to define CG_FLOAT_INF"
#endif
#endif

#define CG_MAX(a,b) (((a) > (b)) ? (a) : (b))
#define CG_MIN(a,b) (((a) < (b)) ? (a) : (b))

/* if BLAS are used, specify the integer precision */
#define CG_BLAS_INT long int

/* if BLAS are used, comment out the next statement if no
 *    underscore in the subroutine names are needed */
#define CG_BLAS_UNDERSCORE

/* only use ddot when the vector size >= CG_DDOT_START */
#define CG_DDOT_START 100

/* only use dcopy when the vector size >= CG_DCOPY_START */
#define CG_DCOPY_START 100

/* only use ddot when the vector size >= CG_DAXPY_START */
#define CG_DAXPY_START 6000

/* only use dscal when the vector size >= CG_DSCAL_START */
#define CG_DSCAL_START 6000

/* only use idamax when the vector size >= CG_IDACG_MAX_START */
#define CG_IDACG_MAX_START 25

/* only use matrix BLAS for transpose multiplication when number of
   elements in matrix >= CG_MATVEC_START */
#define CG_MATVEC_START 8000

#ifdef CG_BLAS_UNDERSCORE

#define CG_DGEMV dgemv_
#define CG_DTRSV dtrsv_
#define CG_DAXPY daxpy_
#define CG_DDOT ddot_
#define CG_DSCAL dscal_
#define CG_DCOPY dcopy_
#define CG_IDACG_MAX idamax_

#else

#define CG_DGEMV dgemv
#define CG_DTRSV dtrsv
#define CG_DAXPY daxpy
#define CG_DDOT ddot
#define CG_DSCAL dscal
#define CG_DCOPY dcopy
#define CG_IDACG_MAX idamax

#endif
/*============================================================================
   cg_parameter is a structure containing parameters used in cg_descent
   cg_default assigns default values to these parameters */
typedef struct cg_parameter_struct /* user controlled parameters */
{
/*============================================================================
      parameters that the user may wish to modify
  ----------------------------------------------------------------------------*/
    /* T => print final statistics
       F => no printout of statistics */
    cg_boolean PrintFinal ;

    /* Level 0  = no printing), ... , Level 3 = maximum printing */
    size_t PrintLevel ;

    /* T => print parameters values
       F => do not display parmeter values */
    cg_boolean PrintParms ;

    /* T => use LBFGS
       F => only use L-BFGS when memory >= n */
    cg_boolean LBFGS ;

    /* number of vectors stored in memory */
    size_t memory ;

    /* SubCheck and SubSkip control the frequency with which the subspace
       condition is checked. It is checked for SubCheck*mem iterations and
       if not satisfied, then it is skipped for Subskip*mem iterations
       and Subskip is doubled. Whenever the subspace condition is statisfied,
       SubSkip is returned to its original value. */
    size_t SubCheck ;
    size_t SubSkip ;

    /* when relative distance from current gradient to subspace <= eta0,
       enter subspace if subspace dimension = mem */
    CG_FLOAT eta0 ;

    /* when relative distance from current gradient to subspace >= eta1,
       leave subspace */
    CG_FLOAT eta1 ;

    /* when relative distance from current direction to subspace <= eta2,
       always enter subspace (invariant space) */
    CG_FLOAT eta2 ;

    /* T => use approximate Wolfe line search
       F => use ordinary Wolfe line search, switch to approximate Wolfe when
                |f_k+1-f_k| < AWolfeFac*C_k, C_k = average size of cost  */
    cg_boolean  AWolfe ;
    CG_FLOAT AWolfeFac ;

    /* factor in [0, 1] used to compute average cost magnitude C_k as follows:
       Q_k = 1 + (Qdecay)Q_k-1, Q_0 = 0,  C_k = C_k-1 + (|f_k| - C_k-1)/Q_k */
    CG_FLOAT Qdecay ;

    /* terminate after nslow iterations without strict improvement in
       either function value or gradient */
    size_t nslow ;

    /* Stop Rules:
       T => ||proj_grad||_infty <= max(grad_tol,initial ||grad||_infty*StopFact)
       F => ||proj_grad||_infty <= grad_tol*(1 + |f_k|) */
    cg_boolean StopRule ;
    CG_FLOAT    StopFac ;

    /* T => estimated error in function value is eps*Ck,
       F => estimated error in function value is eps */
    cg_boolean PertRule ;
    CG_FLOAT eps ;

    /* factor by which eps grows when line search fails during contraction */
    CG_FLOAT egrow ;

    /* T => attempt quadratic interpolation in line search when
                |f_k+1 - f_k|/f_k <= QuadCutoff
       F => no quadratic interpolation step */
    cg_boolean QuadStep ;
    CG_FLOAT QuadCutOff ;

    /* maximum factor by which a quad step can reduce the step size */
    CG_FLOAT QuadSafe ;

    /* T => when possible, use a cubic step in the line search */
    cg_boolean UseCubic ;

    /* use cubic step when |f_k+1 - f_k|/|f_k| > CubicCutOff */
    CG_FLOAT CubicCutOff ;

    /* |f| < SmallCost*starting cost => skip QuadStep and set PertRule = 0*/
    CG_FLOAT SmallCost ;

    /* T => check that f_k+1 - f_k <= debugtol*C_k
       F => no checking of function values */
    cg_boolean debug ;
    CG_FLOAT debugtol ;

    /* if step is nonzero, it is the initial step of the initial line search */
    CG_FLOAT step ;

    /* abort cg after maxit iterations */
    size_t maxit ;

    /* maximum number of times the bracketing interval grows during expansion */
    size_t ntries ;

    /* maximum factor secant step increases stepsize in expansion phase */
    CG_FLOAT ExpandSafe ;

    /* factor by which secant step is amplified during expansion phase
       where minimizer is bracketed */
    CG_FLOAT SecantAmp ;

    /* factor by which rho grows during expansion phase where minimizer is
       bracketed */
    CG_FLOAT RhoGrow ;

    /* maximum number of times that eps is updated */
    size_t neps ;

    /* maximum number of times the bracketing interval shrinks */
    size_t nshrink ;

   /* maximum number of iterations in line search */
    size_t nline ;

    /* conjugate gradient method restarts after (n*restart_fac) iterations */
    CG_FLOAT restart_fac ;

    /* stop when -alpha*dphi0 (estimated change in function value) <= feps*|f|*/
    CG_FLOAT feps ;

    /* after encountering nan, growth factor when searching for
       a bracketing interval */
    CG_FLOAT nan_rho ;

    /* after encountering nan, decay factor for stepsize */
    CG_FLOAT nan_decay ;

/*============================================================================
       technical parameters which the user probably should not touch
  ----------------------------------------------------------------------------*/
    CG_FLOAT           delta ; /* Wolfe line search parameter */
    CG_FLOAT           sigma ; /* Wolfe line search parameter */
    CG_FLOAT           gamma ; /* decay factor for bracket interval width */
    CG_FLOAT             rho ; /* growth factor when searching for initial
                                  bracketing interval */
    CG_FLOAT            psi0 ; /* factor used in starting guess for iteration 1 */
    CG_FLOAT          psi_lo ; /* in performing a QuadStep, we evaluate at point
                                  betweeen [psi_lo, psi_hi]*psi2*previous step */
    CG_FLOAT          psi_hi ;
    CG_FLOAT            psi1 ; /* for approximate quadratic, use gradient at
                                  psi1*psi2*previous step for initial stepsize */
    CG_FLOAT            psi2 ; /* when starting a new cg iteration, our initial
                                  guess for the line search stepsize is
                                  psi2*previous step */
    cg_boolean  AdaptiveBeta ; /* T => choose beta adaptively, F => use theta */
    CG_FLOAT       BetaLower ; /* lower bound factor for beta */
    CG_FLOAT           theta ; /* parameter describing the cg_descent family */
    CG_FLOAT            qeps ; /* parameter in cost error for quadratic restart
                                  criterion */
    CG_FLOAT           qrule ; /* parameter used to decide if cost is quadratic */
    size_t          qrestart ; /* number of iterations the function should be
                                  nearly quadratic before a restart */
} cg_parameter ;

typedef struct cg_stats_struct /* statistics returned to user */
{
    CG_FLOAT               f ; /*function value at solution */
    CG_FLOAT           gnorm ; /* max abs component of gradient */
    size_t              iter ; /* number of iterations */
    size_t           IterSub ; /* number of subspace iterations */
    size_t            NumSub ; /* total number subspaces */
    size_t             nfunc ; /* number of function evaluations */
    size_t             ngrad ; /* number of gradient evaluations */
} cg_stats ;

/* prototypes */

#ifdef __cplusplus
extern "C" {
#endif

CG_API int cg_descent /*  return status of solution process:
                       0 (convergence tolerance satisfied)
                       1 (change in func <= feps*|f|)
                       2 (total number of iterations exceeded maxit)
                       3 (slope always negative in line search)
                       4 (number of line search iterations exceeds nline)
                       5 (search direction not a descent direction)
                       6 (excessive updating of eps)
                       7 (Wolfe conditions never satisfied)
                       8 (debugger is on and the function value increases)
                       9 (no cost or gradient improvement in
                          2n + Parm->nslow iterations)
                      10 (out of memory)
                      11 (function nan or +-INF and could not be repaired)
                      12 (invalid choice for memory parameter) */
(
    CG_FLOAT          *x, /* input: starting guess, output: the solution */
    size_t             n, /* problem dimension */
    cg_stats      *Stats, /* structure with statistics */
    cg_parameter  *UParm, /* user parameters, NULL = use default parameters */
    CG_FLOAT    grad_tol, /* StopRule = 1: |g|_infty <= max (grad_tol,
                                           StopFac*initial |g|_infty) [default]
                             StopRule = 0: |g|_infty <= grad_tol(1+|f|) */
    CG_FLOAT    (*value) (CG_FLOAT *, size_t CG_CUSTOM),  /* f = value (x, n) */
    void        (*grad) (CG_FLOAT *, CG_FLOAT *, size_t CG_CUSTOM), /* grad (g, x, n) */
    CG_FLOAT    (*valgrad) (CG_FLOAT *, CG_FLOAT *, size_t CG_CUSTOM), /* f = valgrad (g,x,n)*/
    CG_FLOAT    *Work  /* either size 4n work array or NULL */
		CG_CUSTOM
) ;

CG_API void cg_default /* set default parameter values */
(
    cg_parameter   *Parm
) ;

#ifdef __cplusplus
}
#endif

typedef struct cg_com_struct /* common variables */
{
    /* parameters computed by the code */
    size_t             n ; /* problem dimension, saved for reference */
    size_t            nf ; /* number of function evaluations */
    size_t            ng ; /* number of gradient evaluations */
    cg_boolean    QuadOK ; /* T (quadratic step successful) */
    cg_boolean  UseCubic ; /* T (use cubic step) F (use secant step) */
    cg_boolean      neps ; /* number of time eps updated */
    cg_boolean  PertRule ; /* T => estimated error in function value is eps*Ck,
                              F => estimated error in function value is eps */
    cg_boolean     QuadF ; /* T => function appears to be quadratic */
    CG_FLOAT   SmallCost ; /* |f| <= SmallCost => set PertRule = F */
    CG_FLOAT       alpha ; /* stepsize along search direction */
    CG_FLOAT           f ; /* function value for step alpha */
    CG_FLOAT          df ; /* function derivative for step alpha */
    CG_FLOAT       fpert ; /* perturbation is eps*|f| if PertRule is T */
    CG_FLOAT         eps ; /* current value of eps */
    CG_FLOAT         tol ; /* computing tolerance */
    CG_FLOAT          f0 ; /* old function value */
    CG_FLOAT         df0 ; /* old derivative */
    CG_FLOAT          Ck ; /* average cost as given by the rule:
i                            Qk = Qdecay*Qk + 1, Ck += (fabs (f) - Ck)/Qk */
    CG_FLOAT    wolfe_hi ; /* upper bound for slope in Wolfe test */
    CG_FLOAT    wolfe_lo ; /* lower bound for slope in Wolfe test */
    CG_FLOAT   awolfe_hi ; /* upper bound for slope, approximate Wolfe test */
    cg_boolean    AWolfe ; /* F (use Wolfe line search)
                                T (use approximate Wolfe line search)
                                do not change user's AWolfe, this value can be
                                changed based on AWolfeFac */
    cg_boolean     Wolfe ; /* T (means code reached the Wolfe part of cg_line */
    CG_FLOAT         rho ; /* either Parm->rho or Parm->nan_rho */
    CG_FLOAT    alphaold ; /* previous value for stepsize alpha */
    CG_FLOAT          *x ; /* current iterate */
    CG_FLOAT      *xtemp ; /* x + alpha*d */
    CG_FLOAT          *d ; /* current search direction */
    CG_FLOAT          *g ; /* gradient at x */
    CG_FLOAT      *gtemp ; /* gradient at x + alpha*d */
    CG_FLOAT   (*cg_value) (CG_FLOAT *, size_t CG_CUSTOM) ; /* f = cg_value (x, n) */
    void        (*cg_grad) (CG_FLOAT *, CG_FLOAT *, size_t CG_CUSTOM) ; /* cg_grad (g, x, n) */
    CG_FLOAT (*cg_valgrad) (CG_FLOAT *, CG_FLOAT *, size_t CG_CUSTOM) ; /* f = cg_valgrad (g,x,n)*/
    cg_parameter *Parm ; /* user parameters */
		CG_CUSTOM_STRUCT
} cg_com ;

#endif // CG_DESCENT_H_

#ifdef CG_DESCENT_IMPLEMENTATION

/* prototypes */

static int cg_Wolfe
(
    CG_FLOAT alpha, /* stepsize */
    CG_FLOAT     f, /* function value associated with stepsize alpha */
    CG_FLOAT  dphi, /* derivative value associated with stepsize alpha */
    cg_com    *Com  /* cg com */
) ;

static cg_boolean cg_tol
(
    CG_FLOAT gnorm, /* gradient sup-norm */
    cg_com    *Com  /* cg com */
) ;

static int cg_line
(
    cg_com *Com  /* cg com structure */
) ;

static int cg_contract
(
    CG_FLOAT    *A, /* left side of bracketing interval */
    CG_FLOAT   *fA, /* function value at a */
    CG_FLOAT   *dA, /* derivative at a */
    CG_FLOAT    *B, /* right side of bracketing interval */
    CG_FLOAT   *fB, /* function value at b */
    CG_FLOAT   *dB, /* derivative at b */
    cg_com    *Com  /* cg com structure */
) ;

static int cg_evaluate
(
    char    *what, /* fg = evaluate func and grad, g = grad only,f = func only*/
    char     *nan, /* y means check function/derivative values for nan */
    cg_com   *Com
) ;

static CG_FLOAT cg_cubic
(
    CG_FLOAT  a,
    CG_FLOAT fa, /* function value at a */
    CG_FLOAT da, /* derivative at a */
    CG_FLOAT  b,
    CG_FLOAT fb, /* function value at b */
    CG_FLOAT db  /* derivative at b */
) ;

static void cg_matvec
(
    CG_FLOAT  *y, /* product vector */
    CG_FLOAT  *A, /* dense matrix */
    CG_FLOAT  *x, /* input vector */
    size_t     n, /* number of columns of A */
    size_t     m, /* number of rows of A */
    cg_boolean w  /* T => y = A*x, F => y = A'*x */
) ;

static void cg_trisolve
(
    CG_FLOAT  *x, /* right side on input, solution on output */
    CG_FLOAT  *R, /* dense matrix */
    size_t     m, /* leading dimension of R */
    size_t     n, /* dimension of triangular system */
    cg_boolean w  /* T => Rx = y, F => R'x = y */
) ;

static CG_FLOAT cg_inf
(
    CG_FLOAT *x, /* vector */
    size_t    n  /* length of vector */
) ;

static void cg_scale0
(
    CG_FLOAT *y, /* output vector */
    CG_FLOAT *x, /* input vector */
    CG_FLOAT  s, /* scalar */
    size_t    n  /* length of vector */
) ;

static void cg_scale
(
    CG_FLOAT *y, /* output vector */
    CG_FLOAT *x, /* input vector */
    CG_FLOAT  s, /* scalar */
    size_t    n  /* length of vector */
) ;

static void cg_daxpy0
(
    CG_FLOAT     *x, /* input and output vector */
    CG_FLOAT     *d, /* direction */
    CG_FLOAT  alpha, /* stepsize */
    size_t        n  /* length of the vectors */
) ;

static void cg_daxpy
(
    CG_FLOAT     *x, /* input and output vector */
    CG_FLOAT     *d, /* direction */
    CG_FLOAT  alpha, /* stepsize */
    size_t        n  /* length of the vectors */
) ;

static CG_FLOAT cg_dot0
(
    CG_FLOAT *x, /* first vector */
    CG_FLOAT *y, /* second vector */
    size_t    n  /* length of vectors */
) ;

static CG_FLOAT cg_dot
(
    CG_FLOAT *x, /* first vector */
    CG_FLOAT *y, /* second vector */
    size_t    n  /* length of vectors */
) ;

static void cg_copy0
(
    CG_FLOAT *y, /* output of copy */
    CG_FLOAT *x, /* input of copy */
    size_t    n  /* length of vectors */
) ;

static void cg_copy
(
    CG_FLOAT *y, /* output of copy */
    CG_FLOAT *x, /* input of copy */
    size_t    n  /* length of vectors */
) ;

static void cg_step
(
    CG_FLOAT *xtemp, /*output vector */
    CG_FLOAT     *x, /* initial vector */
    CG_FLOAT     *d, /* search direction */
    CG_FLOAT  alpha, /* stepsize */
    size_t        n  /* length of the vectors */
) ;

static void cg_init
(
    CG_FLOAT *x, /* input and output vector */
    CG_FLOAT  s, /* scalar */
    size_t    n  /* length of vector */
) ;

static CG_FLOAT cg_update_2
(
    CG_FLOAT *gold, /* old g */
    CG_FLOAT *gnew, /* new g */
    CG_FLOAT    *d, /* d */
    size_t       n  /* length of vectors */
) ;

static CG_FLOAT cg_update_inf
(
    CG_FLOAT *gold, /* old g */
    CG_FLOAT *gnew, /* new g */
    CG_FLOAT    *d, /* d */
    size_t       n  /* length of vectors */
) ;

static CG_FLOAT cg_update_ykyk
(
    CG_FLOAT *gold, /* old g */
    CG_FLOAT *gnew, /* new g */
    CG_FLOAT *Ykyk,
    CG_FLOAT *Ykgk,
    size_t       n  /* length of vectors */
) ;

static CG_FLOAT cg_update_inf2
(
    CG_FLOAT   *gold, /* old g */
    CG_FLOAT   *gnew, /* new g */
    CG_FLOAT      *d, /* d */
    CG_FLOAT *gnorm2, /* 2-norm of g */
    size_t         n  /* length of vectors */
) ;

static CG_FLOAT cg_update_d
(
    CG_FLOAT      *d,
    CG_FLOAT      *g,
    CG_FLOAT    beta,
    CG_FLOAT *gnorm2, /* 2-norm of g */
    size_t         n  /* length of vectors */
) ;

static void cg_Yk
(
    CG_FLOAT    *y, /*output vector */
    CG_FLOAT *gold, /* initial vector */
    CG_FLOAT *gnew, /* search direction */
    CG_FLOAT  *yty, /* y'y */
    size_t       n  /* length of the vectors */
) ;

static void cg_printParms
(
    cg_parameter  *Parm
) ;

/* If the BLAS are not installed, then the following definitions
   can be ignored. If the BLAS are available, then to use them,
   comment out the the next statement (#define CG_NOBLAS) and make
   any needed adjustments to CG_BLAS_UNDERSCORE and the START parameters.
   cg_descent already does loop unrolling, so there is likely no
   benefit from using unrolled BLAS. There could be a benefit from
   using threaded BLAS if the problems is really big. However,
   performing low dimensional operations with threaded BLAS can be
   less efficient than the cg_descent unrolled loops. Hence,
   START parameters should be specified to determine when to start
   using the BLAS. */


void CG_DGEMV (char *trans, CG_BLAS_INT *m, CG_BLAS_INT *n, CG_FLOAT *alpha, CG_FLOAT *A,
        CG_BLAS_INT *lda, CG_FLOAT *X, CG_BLAS_INT *incx,
        CG_FLOAT *beta, CG_FLOAT *Y, CG_BLAS_INT *incy) ;

void CG_DTRSV (char *uplo, char *trans, char *diag, CG_BLAS_INT *n, CG_FLOAT *A,
        CG_BLAS_INT *lda, CG_FLOAT *X, CG_BLAS_INT *incx) ;

void CG_DAXPY (CG_BLAS_INT *n, CG_FLOAT *DA, CG_FLOAT *DX, CG_BLAS_INT *incx, CG_FLOAT *DY,
        CG_BLAS_INT *incy) ;

CG_FLOAT CG_DDOT (CG_BLAS_INT *n, CG_FLOAT *DX, CG_BLAS_INT *incx, CG_FLOAT *DY,
        CG_BLAS_INT *incy) ;

void CG_DSCAL (CG_BLAS_INT *n, CG_FLOAT *DA, CG_FLOAT *DX, CG_BLAS_INT *incx) ;

void CG_DCOPY (CG_BLAS_INT *n, CG_FLOAT *DX, CG_BLAS_INT *incx, CG_FLOAT *DY,
        CG_BLAS_INT *incy) ;

CG_BLAS_INT CG_IDACG_MAX (CG_BLAS_INT *n, CG_FLOAT *DX, CG_BLAS_INT *incx) ;


/* begin external variables */
CG_FLOAT one [1], zero [1] ;
CG_BLAS_INT blas_one [1] ;
/* end external variables */

int cg_descent /*  return status of solution process:
                       0 (convergence tolerance satisfied)
                       1 (change in func <= feps*|f|)
                       2 (total number of iterations exceeded maxit)
                       3 (slope always negative in line search)
                       4 (number of line search iterations exceeds nline)
                       5 (search direction not a descent direction)
                       6 (excessive updating of eps)
                       7 (Wolfe conditions never satisfied)
                       8 (debugger is on and the function value increases)
                       9 (no cost or gradient improvement in
                          2n + Parm->nslow iterations)
                      10 (out of memory)
                      11 (function nan or +-INF and could not be repaired)
                      12 (invalid choice for memory parameter) */
(
    CG_FLOAT         *x, /* input: starting guess, output: the solution */
		size_t            n, /* problem dimension */
    cg_stats      *Stat, /* structure with statistics (can be NULL) */
    cg_parameter *UParm, /* user parameters, NULL = use default parameters */
    CG_FLOAT   grad_tol, /* StopRule = 1: |g|_infty <= max (grad_tol,
                                           StopFac*initial |g|_infty) [default]
                            StopRule = 0: |g|_infty <= grad_tol(1+|f|) */
    CG_FLOAT   (*value) (CG_FLOAT *, size_t CG_CUSTOM),  /* f = value (x, n) */
    void        (*grad) (CG_FLOAT *, CG_FLOAT *, size_t CG_CUSTOM), /* grad (g, x, n) */
    CG_FLOAT (*valgrad) (CG_FLOAT *, CG_FLOAT *, size_t CG_CUSTOM), /* f = valgrad (g, x, n),
                          NULL = compute value & gradient using value & grad */
    CG_FLOAT *Work  /* NULL => let code allocate memory
                       not NULL => use array Work for required memory
                       The amount of memory needed depends on the value
                       of the parameter memory in the Parm structure.
                       memory > 0 => need (mem+6)*n + (3*mem+9)*mem + 5
                       where mem = CG_MIN(memory, n)
                       memory = 0 => need 4*n */
		CG_CUSTOM
)
{
    size_t i, iter, IterRestart, maxit = 0, nrestart, nrestartsub = 0, qrestart, IterQuad,
			IterSub, IterSubRestart, mem, memk, memsq, memk_begin, nslow, slowlimit,
			l1, l2, j, k, mlast, mlast_sub, nsub, SkFstart = 0, SkFlast = 0,Subspace, NumSub,
			SubSkip, SubCheck = 0, mp, mp_begin = 0, mpp, spp, spp1 ;
    int status, PrintLevel, QuadF, UseMemory, Restart, LBFGS, InvariantSpace, FirstFull, 
      StartSkip = 0, StartCheck, DenseCol1 = 0, NegDiag, memk_is_mem, d0isg = 0 ;
    CG_FLOAT delta2, Qk, Ck, fbest, gbest,
            f, ftemp, gnorm, xnorm, gnorm2, dnorm2, denom,
            t, dphi, dphi0, alpha,
            ykyk, ykgk, dkyk, beta = 0, QuadTrust, tol,
            *d = NULL, *g = NULL, *xtemp = NULL, *gtemp = NULL, *work = NULL ;
    CG_FLOAT  gHg, scale, gsubnorm2 = 0,  ratio, stgkeep = 0,
            alphaold, zeta, yty, ytg, t1, t2, t3, t4,
           *Rk = NULL, *Re = NULL, *Sk = NULL, *SkF = NULL, *stemp = NULL,
           *Yk = NULL, *SkYk = NULL, *dsub = NULL, *gsub = NULL, *gsubtemp = NULL,
           *gkeep = NULL, *tau = NULL, *vsub = NULL, *wsub = NULL ;

    cg_parameter *Parm, ParmStruc ;
    cg_com Com ;

    /* assign values to the external variables */
    one [0] = (CG_FLOAT) 1 ;
    zero [0] = (CG_FLOAT) 0 ;
    blas_one [0] = (CG_BLAS_INT) 1 ;

    /* initialize the parameters */
    if ( UParm == NULL )
    {
        Parm = &ParmStruc ;
        cg_default (Parm) ;
    }
    else Parm = UParm ;
    PrintLevel = Parm->PrintLevel ;
    qrestart = CG_MIN (n, Parm->qrestart) ;
    Com.Parm = Parm ;
    Com.eps = Parm->eps ;
    Com.PertRule = Parm->PertRule ;
    Com.Wolfe = 0 ; /* initially Wolfe line search not performed */
    Com.nf = (size_t) 0 ;  /* number of function evaluations */
    Com.ng = (size_t) 0 ;  /* number of gradient evaluations */
		CG_INIT_CUSTOM(Com)
		iter = (size_t) 0 ;    /* total number of iterations */
    QuadF = 0 ;     /* initially function assumed to be nonquadratic */
    NegDiag = 0 ;   /* no negative diagonal elements in QR factorization */
    mem = Parm->memory ;/* cg_descent corresponds to mem = 0 */

    if ( Parm->PrintParms ) cg_printParms (Parm) ;
    if ( (mem != 0) && (mem < 3) )
    {
        status = 12 ;
        goto Exit ;
    }

    /* allocate work array */
    mem = CG_MIN (mem, n) ;
    if ( Work == NULL )
    {
        if ( mem == 0 ) /* original CG_DESCENT without memory */
        {
            work = (CG_FLOAT *) malloc (4*n*sizeof (CG_FLOAT)) ;
        }
        else if ( Parm->LBFGS || (mem >= n) ) /* use L-BFGS */
        {
            work = (CG_FLOAT *) malloc ((2*mem*(n+1)+4*n)*sizeof (CG_FLOAT)) ;
        }
        else /* limited memory CG_DESCENT */
        {
            i = (mem+6)*n + (3*mem+9)*mem + 5 ;
            work = (CG_FLOAT *) malloc (i*sizeof (CG_FLOAT)) ;
        }
    }
    else work = Work ;
    if ( work == NULL )
    {
        status = 10 ;
        goto Exit ;
    }

    /* set up Com structure */
    Com.x = x ;
    Com.xtemp = xtemp = work ;
    Com.d = d = xtemp+n ;
    Com.g = g = d+n ;
    Com.gtemp = gtemp = g+n ;
    Com.n = n ;          /* problem dimension */
    Com.neps = 0 ;       /* number of times eps updated */
    Com.AWolfe = Parm->AWolfe ; /* do not touch user's AWolfe */
    Com.cg_value = value ;
    Com.cg_grad = grad ;
    Com.cg_valgrad = valgrad ;
    LBFGS = CG_FALSE ;
    UseMemory = 0 ; /* do not use memory */
    Subspace = 0 ;  /* full space, check subspace condition if UseMemory */
    FirstFull = 0 ; /* not first full iteration after leaving subspace */
    memk = 0 ;      /* number of vectors in current memory */

    /* the conjugate gradient algorithm is restarted every nrestart iteration */
    nrestart = (size_t) (((CG_FLOAT) n)*Parm->restart_fac) ;

    /* allocate storage connected with limited memory CG */
    if ( mem > 0 )
    {
        if ( (mem == n) || Parm->LBFGS )
        {
            LBFGS = CG_TRUE ;      /* use L-BFGS */
            mlast = -1 ;
            Sk = gtemp + n ;
            Yk = Sk + mem*n ;
            SkYk = Yk + mem*n ;
            tau = SkYk + mem ;
        }
        else
        {
            UseMemory = 1 ; /* previous search direction will be saved */
            SubSkip = 0 ;      /* number of iterations to skip checking memory*/
            SubCheck = mem*Parm->SubCheck ; /* number of iterations to check */
            StartCheck = 0 ;   /* start checking memory at iteration 0 */
            InvariantSpace = 0 ; /* iterations not in invariant space */
            FirstFull = 1 ;       /* first iteration in full space */
            nsub = 0 ;               /* initial subspace dimension */
            memsq = mem*mem ;
            SkF = gtemp+n ;    /* directions in memory (x_k+1 - x_k) */
            stemp = SkF + mem*n ;/* stores x_k+1 - x_k */
            gkeep = stemp + n ;  /* store gradient when first direction != -g */
            Sk = gkeep + n ;   /* Sk = Rk at start of LBFGS in subspace */
            Rk = Sk + memsq ;  /* upper triangular factor in SkF = Zk*Rk */
            /* zero out Rk to ensure lower triangle is 0 */
            cg_init (Rk, 0, memsq) ;
            Re = Rk + memsq ;  /* end column of Rk, used for new direction */
            Yk = Re + mem+1 ;
            SkYk = Yk + memsq+mem+2 ; /* dot products sk'yk in the subspace */
            tau = SkYk + mem ;       /* stores alpha in Nocedal and Wright */
            dsub = tau + mem ;       /* direction projection in subspace */
            gsub = dsub + mem ;      /* gradient projection in subspace */
            gsubtemp = gsub + mem+1 ;/* new gsub before update */
            wsub = gsubtemp + mem ;  /* mem+1 work array for triangular solve */
            vsub = wsub + mem+1 ;    /* mem work array for triangular solve */
        }
    }

    /* abort when number of iterations reaches maxit */
    maxit = Parm->maxit ;

    f = 0 ;
    fbest = CG_FLOAT_INF ;
    gbest = CG_FLOAT_INF ;
    nslow = 0 ;
    slowlimit = 2*n + Parm->nslow ;

    Ck = 0 ;
    Qk = 0 ;

    /* initial function and gradient evaluations, initial direction */
    Com.alpha = 0 ;
    status = cg_evaluate ("fg", "n", &Com) ;
    f = Com.f ;
    if ( status )
    {
        if ( PrintLevel > 0 ) printf ("Function undefined at starting point\n");
        goto Exit ;
    }
        
    Com.f0 = f + f ;
    Com.SmallCost = fabs (f)*Parm->SmallCost ;
    xnorm = cg_inf (x, n) ;

    /* set d = -g, compute gnorm  = infinity norm of g and
                           gnorm2 = square of 2-norm of g */
    gnorm = cg_update_inf2 (g, g, d, &gnorm2, n) ;
    dnorm2 = gnorm2 ;

    /* check if the starting function value is nan */
    if ( f != f )
    {
        status = 11 ;
        goto Exit ;
    }

    if ( Parm->StopRule ) tol = CG_MAX (gnorm*Parm->StopFac, grad_tol) ;
    else                  tol = grad_tol ;
    Com.tol = tol ;

    if ( PrintLevel >= 1 )
    {
        printf ("iter: %5i f: %13.6e gnorm: %13.6e memk: %zu\n",
        (int) 0, f, gnorm, memk) ;
    }

    if ( cg_tol (gnorm, &Com) )
    {
        iter = 0 ;
        status = 0 ;
        goto Exit ;
    }

    dphi0 = -gnorm2 ;
    delta2 = 2*Parm->delta - 1 ;
    alpha = Parm->step ;
    if ( alpha == 0 )
    {
        if ( xnorm == 0 )
        {
            if ( f != 0 ) alpha = 2.*fabs (f)/gnorm2 ;
            else             alpha = 1 ;
        }
        else    alpha = Parm->psi0*xnorm/gnorm ;
    }

    Com.df0 = -2.0*fabs(f)/alpha ;

    Restart = 0 ;    /* do not restart the algorithm */
    IterRestart = 0 ;    /* counts number of iterations since last restart */
    IterSub = 0 ;        /* counts number of iterations in subspace */
    NumSub =  0 ;        /* total number of subspaces */
    IterQuad = 0 ;       /* counts number of iterations that function change
                            is close to that of a quadratic */
    scale = (CG_FLOAT) 1 ; /* scale is the initial approximation to inverse
                            Hessian in LBFGS; after the initial iteration,
                            scale is estimated by the BB formula */

    /* Start the conjugate gradient iteration.
       alpha starts as old step, ends as final step for current iteration
       f is function value for alpha = 0
       QuadOK = 1 means that a quadratic step was taken */

    for (iter = 1; iter <= maxit; iter++)
    {
        /* save old alpha to simplify formula computing subspace direction */
        alphaold = alpha ;
        Com.QuadOK = 0 ;
        alpha = Parm->psi2*alpha ;
        if ( f != 0 ) t = fabs ((f-Com.f0)/f) ;
        else             t = 1 ;
        Com.UseCubic = 1 ;
        if ( (t < Parm->CubicCutOff) || !Parm->UseCubic ) Com.UseCubic = 0 ;
        if ( Parm->QuadStep )
        {
            /* test if quadratic interpolation step should be tried */
            if ( ((t > Parm->QuadCutOff)&&(fabs(f) >= Com.SmallCost)) || QuadF )
            {
                if ( QuadF )
                {
                    Com.alpha = Parm->psi1*alpha ;
                    status = cg_evaluate ("g", "y", &Com) ;
                    if ( status ) goto Exit ;
                    if ( Com.df > dphi0 )
                    {
                        alpha = -dphi0/((Com.df-dphi0)/Com.alpha) ;
                        Com.QuadOK = 1 ;
                    }
                    else if ( LBFGS )
                    {
                        if ( memk >= n )
                        {
                            alpha = 1 ;
                            Com.QuadOK = 1 ;
                        }
                        else  alpha = 2. ;
                    }
                    else if ( Subspace )
                    {
                        if ( memk >= nsub )
                        {
                            alpha = 1 ;
                            Com.QuadOK = 1 ;
                        }
                        else  alpha = 2. ;
                    }
                }
                else
                {
                    t = CG_MAX (Parm->psi_lo, Com.df0/(dphi0*Parm->psi2)) ;
                    Com.alpha = CG_MIN (t, Parm->psi_hi)*alpha ;
                    status = cg_evaluate ("f", "y", &Com) ;
                    if ( status ) goto Exit ;
                    ftemp = Com.f ;
                    denom = 2.*(((ftemp-f)/Com.alpha)-dphi0) ;
                    if ( denom > 0 )
                    {
                        t = -dphi0*Com.alpha/denom ;
                        /* safeguard */
                        if ( ftemp >= f )
                              alpha = CG_MAX (t, Com.alpha*Parm->QuadSafe) ;
                        else  alpha = t ;
                        Com.QuadOK = 1 ;
                    }
                }
                if ( PrintLevel >= 1 )
                {
                    if ( denom <= 0 )
                    {
                        printf ("Quad step fails (denom = %14.6e)\n", denom);
                    }
                    else if ( Com.QuadOK )
                    {
                        printf ("Quad step %14.6e OK\n", alpha);
                    }
                    else printf ("Quad step %14.6e done, but not OK\n", alpha) ;
                }
            }
            else if ( PrintLevel >= 1 )
            {
                printf ("No quad step (chg: %14.6e, cut: %10.2e)\n",
                         t, Parm->QuadCutOff) ;
            }
        }
        Com.f0 = f ;                          /* f0 saved as prior value */
        Com.df0 = dphi0 ;

        /* parameters in Wolfe and approximate Wolfe conditions, and in update*/

        Qk = Parm->Qdecay*Qk + 1 ;
        Ck = Ck + (fabs (f) - Ck)/Qk ;        /* average cost magnitude */

        if ( Com.PertRule ) Com.fpert = f + Com.eps*fabs (f) ;
        else                Com.fpert = f + Com.eps ;

        Com.wolfe_hi = Parm->delta*dphi0 ;
        Com.wolfe_lo = Parm->sigma*dphi0 ;
        Com.awolfe_hi = delta2*dphi0 ;
        Com.alpha = alpha ;

        /* perform line search */
        status = cg_line (&Com) ;

        /*try approximate Wolfe line search if ordinary Wolfe fails */
        if ( (status > 0) && !Com.AWolfe )
        {
            if ( PrintLevel >= 1 )
            {
                 printf ("\nWOLFE LINE SEARCH FAILS\n") ;
            }
            if ( status != 3 )
            {
                Com.AWolfe = CG_TRUE ;
                status = cg_line (&Com) ;
            }
        }

        alpha = Com.alpha ;
        f = Com.f ;
        dphi = Com.df ;

        if ( status ) goto Exit ;

        /* Test for convergence to within machine epsilon
           [set feps to zero to remove this test] */

        if ( -alpha*dphi0 <= Parm->feps*fabs (f) )
        {
            status = 1 ;
            goto Exit ;
        }

        /* test how close the cost function changes are to that of a quadratic
           QuadTrust = 0 means the function change matches that of a quadratic*/
        t = alpha*(dphi+dphi0) ;
        if ( fabs (t) <= Parm->qeps*CG_MIN (Ck, 1) ) QuadTrust = 0 ;
        else QuadTrust = fabs((2.0*(f-Com.f0)/t)-1) ;
        if ( QuadTrust <= Parm->qrule) IterQuad++ ;
        else                           IterQuad = 0 ;

        if ( IterQuad == qrestart ) QuadF = 1 ;
        IterRestart++ ;
        if ( !Com.AWolfe )
        {
            if ( fabs (f-Com.f0) < Parm->AWolfeFac*Ck )
            {
                Com.AWolfe = CG_TRUE ;
                if ( Com.Wolfe ) Restart = 1 ;
            }
        }

        if ( (mem > 0) && !LBFGS )
        {
            if ( UseMemory )
            {
                if ( (iter > SubCheck + StartCheck) && !Subspace )
                {
                    StartSkip = iter ;
                    UseMemory = 0 ;
                    if ( SubSkip == 0 ) SubSkip = mem*Parm->SubSkip ;
                    else                SubSkip *= 2 ;
                    if ( PrintLevel >= 1 )
                    {
                        printf ("skip subspace %zu iterations\n", SubSkip) ;
                    }
                }
            }
            else
            {
                if ( iter > SubSkip + StartSkip )
                {
                    StartCheck = iter ;
                    UseMemory = 1 ;
                    memk = 0 ;
                }
            }
        }

        if ( !UseMemory )
        {
            if ( !LBFGS )
            {
                if ( (IterRestart >= nrestart) || ((IterQuad == qrestart)
                     && (IterQuad != IterRestart)) ) Restart = 1 ;
            }
        }
        else
        {
            if ( Subspace ) /* the iteration is in the subspace */
            {
                IterSubRestart++ ;

                /* compute project of g into subspace */
                gsubnorm2 = 0 ;
                mp = SkFstart ;
                j = nsub - mp ;

                /* multiply basis vectors by new gradient */
                cg_matvec (wsub, SkF, gtemp, nsub, n, 0) ;

                /* rearrange wsub and store in gsubtemp
                   (elements associated with old vectors should
                    precede elements associated with newer vectors */
                cg_copy0 (gsubtemp, wsub+mp, j) ;
                cg_copy0 (gsubtemp+j, wsub, mp) ;

                /* solve Rk'y = gsubtemp */
                cg_trisolve (gsubtemp, Rk, mem, nsub, 0) ;
                gsubnorm2 = cg_dot0 (gsubtemp, gsubtemp, nsub) ;
                gnorm2 = cg_dot (gtemp, gtemp, n);
                ratio = sqrt(gsubnorm2/gnorm2) ;
                if ( ratio < 1 - Parm->eta1  ) /* Exit Subspace */
                {
                   if ( PrintLevel >= 1 )
                   {
                       printf ("iter: %i exit subspace\n", (int) iter) ;
                   }
                   FirstFull = 1 ; /* first iteration in full space */
                   Subspace = 0 ; /* leave the subspace */
                   InvariantSpace = 0 ;
                   /* check the subspace condition for SubCheck iterations
                      starting from the current iteration (StartCheck) */
                   StartCheck = iter ;
                   if ( IterSubRestart > 1 ) dnorm2 = cg_dot0 (dsub, dsub,nsub);
                }
                else
                {
                   /* Check if a restart should be done in subspace */
                   if ( IterSubRestart == nrestartsub ) Restart = 1 ;
                }
            }
            else  /* in full space */
            {
                if ( (IterRestart == 1) || FirstFull ) memk = 0 ;
                if ( (memk == 1) && InvariantSpace )
                {
                     memk = 0 ;
                     InvariantSpace = 0 ;
                }
                if (memk < mem )
                {
                    memk_is_mem = 0 ;
                    SkFstart = 0 ;
                    /* SkF stores basis vector of the form alpha*d
                       We factor SkF = Zk*Rk where Zk has orthonormal columns
                       and Rk is upper triangular. Zk is not stored; wherever
                       it is needed, we use SkF * inv (Rk) */
                    if (memk == 0)
                    {
                        mlast = 0 ;  /* starting pointer in the memory */
                        memk = 1 ;   /* dimension of current subspace */
 
                        t = sqrt(dnorm2) ;
                        zeta = alpha*t ;
                        Rk [0] = zeta ;
                        cg_scale (SkF, d, alpha, n) ;
                        Yk [0] = (dphi - dphi0)/t ;
                        gsub [0] = dphi/t ;
                        SkYk [0] = alpha*(dphi-dphi0) ;
                        FirstFull = 0 ;
                        if ( IterRestart > 1 )
                        {
                           /* Need to save g for later correction of first
                              column of Yk. Since g does not lie in the
                              subspace and the first column is dense */
                           cg_copy (gkeep, g, n) ;
                           /* Also store dot product of g with the first
                              direction vector -- this saves a later dot
                              product when we fix the first column of Yk */
                           stgkeep = dphi0*alpha ;
                           d0isg = 0 ;
                        }
                        else d0isg = 1 ;
                    }
                    else
                    {
                        mlast = memk ; /* starting pointer in the memory */
                        memk++ ;       /* total number of Rk in the memory */
                        mpp = mlast*n ;
                        spp = mlast*mem ;
                        cg_scale (SkF+mpp, d, alpha, n) ;
 
                        /* check if the alphas are far from 1 */
                        if ((fabs(alpha-5.05)>4.95)||(fabs(alphaold-5.05)>4.95))
                        {
                            /* multiply basis vectors by new direction vector */
                            cg_matvec (Rk+spp, SkF, SkF+mpp, mlast, n, 0) ;

                            /* solve Rk'y = wsub to obtain the components of the
                               new direction vector relative to the orthonormal
                               basis Z in S = ZR, store in next column of Rk */
                            cg_trisolve (Rk+spp, Rk, mem, mlast, 0) ;
                        }
                        else /* alphas are close to 1 */
                        {
                            t1 = -alpha ;
                            t2 = beta*alpha/alphaold ;
                            for (j = 0; j < mlast; j++)
                            {
                                Rk [spp+j] = t1*gsub [j] + t2*Rk [spp-mem+j] ;
                            }
                        }
                        t = alpha*alpha*dnorm2 ;
                        t1 = cg_dot0 (Rk+spp, Rk+spp, mlast) ;
                        if (t <= t1)
                        {
                            zeta = t*1.e-12 ;
                            NegDiag = 1 ;
                        }
                        else zeta = sqrt(t-t1);
   
                        Rk [spp+mlast] = zeta ;
                        t = - zeta/alpha ; /* t = cg_dot0 (Zk+mlast*n, g, n)*/
                        Yk [spp-mem+mlast] = t ;
                        gsub [mlast] = t ;
   
                        /* multiply basis vectors by new gradient */
                        cg_matvec (wsub, SkF, gtemp, mlast, n, 0) ;
                        /* exploit dphi for last multiply */
                        wsub [mlast] = alpha*dphi ;
                        /* solve for new gsub */
                        cg_trisolve (wsub, Rk, mem, memk, 0) ;
                        /* subtract old gsub from new gsub = column of Yk */
                        cg_Yk (Yk+spp, gsub, wsub, NULL, memk) ;
  
                        SkYk [mlast] = alpha*(dphi-dphi0) ;
                    }
                }
                else  /* memk = mem */
                {
                    memk_is_mem = 1 ;
                    mlast = mem-1 ;
                    cg_scale (stemp, d, alpha, n) ;
                    /* compute projection of s_k = alpha_k d_k into subspace
                       check if the alphas are far from 1 */
                    if ((fabs(alpha-5.05)>4.95)||(fabs(alphaold-5.05)>4.95))
                    {
                        mp = SkFstart ;
                        j = mem - mp ;

                        /* multiply basis vectors by sk */
                        cg_matvec (wsub, SkF, stemp, mem, n, 0) ;
                        /* rearrange wsub and store in Re = end col Rk */
                        cg_copy0 (Re, wsub+mp, j) ;
                        cg_copy0 (Re+j, wsub, mp) ;

                        /* solve Rk'y = Re */
                        cg_trisolve (Re, Rk, mem, mem, 0) ;
                    }
                    else /* alphas close to 1 */
                    {
                        t1 = -alpha ;
                        t2 = beta*alpha/alphaold ;
                        for (j = 0; j < mem; j++)
                        {
                            Re [j] = t1*gsub [j] + t2*Re [j-mem] ;
                        }
                    }
 
                    /* t = 2-norm squared of s_k */
                    t = alpha*alpha*dnorm2 ;
                    /* t1 = 2-norm squared of projection */
                    t1 = cg_dot0 (Re, Re, mem) ;
                    if (t <= t1)
                    {
                        zeta = t*1.e-12 ;
                        NegDiag = 1 ;
                    }
                    else zeta = sqrt(t-t1);
 
                    /* dist from new search direction to prior subspace*/
                    Re [mem] = zeta ;

                    /* projection of prior g on new orthogonal
                       subspace vector */
                    t = -zeta/alpha ; /* t = cg_dot(Zk+mpp, g, n)*/
                    gsub [mem] = t ;
                    Yk [memsq] = t ;  /* also store it in Yk */

                    spp = memsq + 1 ;
                    mp = SkFstart ;
                    j = mem - mp ;

                    /* multiply basis vectors by gtemp */
                    cg_matvec (vsub, SkF, gtemp, mem, n, 0) ;

                    /* rearrange and store in wsub */
                    cg_copy0 (wsub, vsub+mp, j) ;
                    cg_copy0 (wsub+j, vsub, mp) ;

                    /* solve Rk'y = wsub */
                    cg_trisolve (wsub, Rk, mem, mem, 0) ;
                    wsub [mem] = (alpha*dphi - cg_dot0 (wsub, Re, mem))/zeta;

                    /* add new column to Yk, store new gsub */
                    cg_Yk (Yk+spp, gsub, wsub, NULL, mem+1) ;
 
                    /* store sk (stemp) at SkF+SkFstart */
                    cg_copy (SkF+SkFstart*n, stemp, n) ;
                    SkFstart++ ;
                    if ( SkFstart == mem ) SkFstart = 0 ;
 
                    mp = SkFstart ;
                    for (k = 0; k < mem; k++)
                    {
                        spp = (k+1)*mem + k ;
                        t1 = Rk [spp] ;
                        t2 = Rk [spp+1] ;
                        t = sqrt(t1*t1 + t2*t2) ;
                        t1 = t1/t ;
                        t2 = t2/t ;
 
                        /* update Rk */
                        Rk [k*mem+k] = t ;
                        for (j = (k+2); j <= mem; j++)
                        {
                            spp1 = spp ;
                            spp = j*mem + k ;
                            t3 = Rk [spp] ;
                            t4 = Rk [spp+1] ;
                            Rk [spp1] = t1*t3 + t2*t4 ;
                            Rk [spp+1] = t1*t4 - t2*t3 ;
                        }
                        /* update Yk */
                        if ( k < 2 ) /* mem should be greater than 2 */
                        {
                            /* first 2 rows are dense */
                            spp = k ;
                            for (j = 1; j < mem; j++)
                            {
                                spp1 = spp ;
                                spp = j*mem + k ;
                                t3 = Yk [spp] ;
                                t4 = Yk [spp+1] ;
                                Yk [spp1] = t1*t3 + t2*t4 ;
                                Yk [spp+1] = t1*t4 -t2*t3 ;
                            }
                            spp1 = spp ;
                            spp = mem*mem + 1 + k ;
                            t3 = Yk [spp] ;
                            t4 = Yk [spp+1] ;
                            Yk [spp1] = t1*t3 + t2*t4 ;
                            Yk [spp+1] = t1*t4 -t2*t3 ;
                        }
                        else if ( (k == 2) && (2 < mem-1))
                        {
                            spp = k ;

                            /* col 1 dense since the oldest direction
                               vector has been dropped */
                            j = 1 ;
                            spp1 = spp ;
                            spp = j*mem + k ;
                            /* single nonzero percolates down the column */
                            t3 = Yk [spp] ;  /* t4 = 0. */
                            Yk [spp1] = t1*t3 ;
                            Yk [spp+1] = -t2*t3 ;
                            /* process rows in Hessenberg part of matrix */
                            for (j = 2; j < mem; j++)
                            {
                                spp1 = spp ;
                                spp = j*mem + k ;
                                t3 = Yk [spp] ;
                                t4 = Yk [spp+1] ;
                                Yk [spp1] = t1*t3 + t2*t4 ;
                                Yk [spp+1] = t1*t4 -t2*t3 ;
                            }
                            spp1 = spp ;
                            spp = mem*mem + 1 + k ;
                            t3 = Yk [spp] ;
                            t4 = Yk [spp+1] ;
                            Yk [spp1] = t1*t3 + t2*t4 ;
                            Yk [spp+1] = t1*t4 -t2*t3 ;
                        }
                        else if ( k < (mem-1) )
                        {
                            spp = k ;

                            /* process first column */
                            j = 1 ;
                            spp1 = spp ;
                            spp = j*mem + k ;
                            t3 = Yk [spp] ;  /* t4 = 0. */
                            Yk [spp1] = t1*t3 ;
                            Yk [spp+1] = -t2*t3 ;

                            /* process rows in Hessenberg part of matrix */
                            j = k-1 ;
                            spp = (j-1)*mem+k ;
                            spp1 = spp ;
                            spp = j*mem + k ;
                            t3 = Yk [spp] ;
                            Yk [spp1] = t1*t3 ; /* t4 = 0. */
                            /* Yk [spp+1] = -t2*t3 ;*/
                            /* Theoretically this element is zero */
                            for (j = k; j < mem; j++)
                            {
                                spp1 = spp ;
                                spp = j*mem + k ;
                                t3 = Yk [spp] ;
                                t4 = Yk [spp+1] ;
                                Yk [spp1] = t1*t3 + t2*t4 ;
                                Yk [spp+1] = t1*t4 -t2*t3 ;
                            }
                            spp1 = spp ;
                            spp = mem*mem + 1 + k ;
                            t3 = Yk [spp] ;
                            t4 = Yk [spp+1] ;
                            Yk [spp1] = t1*t3 + t2*t4 ;
                            Yk [spp+1] = t1*t4 -t2*t3 ;
                        }
                        else /* k = mem-1 */
                        {
                            spp = k ;

                            /* process first column */
                            j = 1 ;
                            spp1 = spp ;
                            spp = j*mem + k ;
                            t3 = Yk [spp] ; /* t4 = 0. */
                            Yk [spp1] = t1*t3 ;

                            /* process rows in Hessenberg part of matrix */
                            j = k-1 ;
                            spp = (j-1)*mem+k ;
                            spp1 = spp ;
                            spp = j*mem + k ;
                            t3 = Yk [spp] ; /* t4 = 0. */
                            Yk [spp1] = t1*t3 ;

                            j = k ;
                            spp1 = spp ;
                            spp = j*mem + k ; /* j=mem-1 */
                            t3 = Yk [spp] ;
                            t4 = Yk [spp+1] ;
                            Yk [spp1] = t1*t3 + t2*t4 ;

                            spp1 = spp ;
                            spp = mem*mem + 1 + k ; /* j=mem */
                            t3 = Yk [spp] ;
                            t4 = Yk [spp+1] ;
                            Yk [spp1] = t1*t3 + t2*t4 ;
                        }
                        /* update g in subspace */
                        if ( k < (mem-1) )
                        {
                            t3 = gsub [k] ;
                            t4 = gsub [k+1] ;
                            gsub [k] = t1*t3 + t2*t4 ;
                            gsub [k+1] = t1*t4 -t2*t3 ;
                        }
                        else /* k = mem-1 */
                        {
                            t3 = gsub [k] ;
                            t4 = gsub [k+1] ;
                            gsub [k] = t1*t3 + t2*t4 ;
                        }
                    }
 
                    /* update SkYk */
                    for (k = 0; k < mlast; k++) SkYk [k] = SkYk [k+1] ;
                    SkYk [mlast] = alpha*(dphi-dphi0) ;
                }
 
                /* calculate t = ||gsub|| / ||gtemp||  */
                gsubnorm2 = cg_dot0 (gsub, gsub, memk) ;
                gnorm2 = cg_dot (gtemp, gtemp, n) ;
                ratio = sqrt (gsubnorm2/gnorm2) ;
                if ( ratio > 1-Parm->eta2) InvariantSpace = 1 ;

                /* check to see whether to enter subspace */
                if ( ((memk > 1) && InvariantSpace) ||
                     ((memk == mem) && (ratio > 1-Parm->eta0)) )
                {
                    NumSub++ ;
                    if ( PrintLevel >= 1 )
                    {
                        if ( InvariantSpace )
                        {
                            printf ("iter: %i invariant space, "
                                    "enter subspace\n", (int) iter) ;
                        }
                        else
                        {
                            printf ("iter: %i enter subspace\n", (int) iter) ;
                        }
                    }
                    /* if the first column is dense, we need to correct it
                       now since we do not know the entries until the basis
                       is determined */
                    if ( !d0isg && !memk_is_mem )
                    {
                        wsub [0] = stgkeep ;
                        /* mlast = memk -1 */
                        cg_matvec (wsub+1, SkF+n, gkeep, mlast, n, 0) ;
                        /* solve Rk'y = wsub */
                        cg_trisolve (wsub, Rk, mem, memk, 0) ;
                        /* corrected first column of Yk */
                        Yk [1] -= wsub [1] ;
                        cg_scale0 (Yk+2, wsub+2, -1, memk-2) ;
                    }
                    if ( d0isg && !memk_is_mem ) DenseCol1 = 0 ;
                    else                         DenseCol1 = 1 ;

                    Subspace = 1 ;
                    /* reset subspace skipping to 0, need to test invariance */
                    SubSkip = 0 ;
                    IterSubRestart = 0 ;
                    nsub = memk ; /* dimension of subspace */
                    nrestartsub = (int) (((CG_FLOAT) nsub)*Parm->restart_fac) ;
                    mp_begin = mlast ;
                    memk_begin = nsub ;
                    SkFlast = (SkFstart+nsub-1) % mem ;
                    cg_copy0 (gsubtemp, gsub, nsub) ;
                    /* Rk contains the sk for subspace, initialize Sk = Rk */
                    cg_copy (Sk, Rk, (int) mem*nsub) ;
                }
                else
                {
                   if ( (IterRestart == nrestart) ||
                       ((IterQuad == qrestart) && (IterQuad != IterRestart)) )
                   {
                       Restart = 1 ;
                   }
                }
            } /* done checking the full space */
        } /* done using the memory */

        /* compute search direction */
        if ( LBFGS )
        {
            gnorm = cg_inf (gtemp, n) ;
            if ( cg_tol (gnorm, &Com) )
            {
                status = 0 ;
                cg_copy (x, xtemp, n) ;
                goto Exit ;
            }

            if ( IterRestart == nrestart ) /* restart the l-bfgs method */
            {
                IterRestart = 0 ;
                IterQuad = 0 ;
                mlast = -1 ;
                memk = 0 ;
                scale = (CG_FLOAT) 1 ;

                /* copy xtemp to x */
                cg_copy (x, xtemp, n) ;

                /* set g = gtemp, d = -g, compute 2-norm of g */
                gnorm2 = cg_update_2 (g, gtemp, d, n) ;

                dnorm2 = gnorm2 ;
                dphi0 = -gnorm2 ;
            }
            else
            {
                mlast = (mlast+1) % mem ;
                spp = mlast*n ;
                cg_step (Sk+spp, xtemp, x, -1, n) ;
                cg_step (Yk+spp, gtemp, g, -1, n) ;
                SkYk [mlast] = alpha*(dphi-dphi0) ;
                if (memk < mem) memk++ ;

                /* copy xtemp to x */
                cg_copy (x, xtemp, n) ;

                /* copy gtemp to g and compute 2-norm of g */
                gnorm2 = cg_update_2 (g, gtemp, NULL, n) ;

                /* calculate Hg = H g, saved in gtemp */
                mp = mlast ;  /* memk is the number of vectors in the memory */
                for (j = 0; j < memk; j++)
                {
                    mpp = mp*n ;
                    t = cg_dot (Sk+mpp, gtemp, n)/SkYk[mp] ;
                    tau [mp] = t ;
                    cg_daxpy (gtemp, Yk+mpp, -t, n) ;
                    if ( mp == 0 ) mp = mem-1 ; else mp--;
                }
                /* scale = (alpha*dnorm2)/(dphi-dphi0) ; */
                t = cg_dot (Yk+mlast*n, Yk+mlast*n, n) ;
                if ( t > 0 )
                {
                    scale = SkYk[mlast]/t ;
                }

                cg_scale (gtemp, gtemp, scale, n) ;

                for (j = 0; j < memk; j++)
                {
                    mp +=  1 ;
                    if ( mp == mem ) mp = 0 ;
                    mpp = mp*n ;
                    t = cg_dot (Yk+mpp, gtemp, n)/SkYk[mp] ;
                    cg_daxpy (gtemp, Sk+mpp, tau [mp]-t, n) ;
                }

                /* set d = -gtemp, compute 2-norm of gtemp */
                dnorm2 = cg_update_2 (NULL, gtemp, d, n) ;
                dphi0 = -cg_dot (g, gtemp, n) ;
            }
        } /* end of LBFGS */

        else if ( Subspace ) /* compute search direction in subspace */
        {
            IterSub++ ;

            /* set x = xtemp */
            cg_copy (x, xtemp, n) ;
            /* set g = gtemp and compute infinity norm of g */
            gnorm = cg_update_inf (g, gtemp, NULL, n) ;

            if ( cg_tol (gnorm, &Com) )
            {
                status = 0 ;
                goto Exit ;
            }

            if ( Restart ) /*restart in subspace*/
            {
                scale = (CG_FLOAT) 1 ;
                Restart = 0 ;
                IterRestart = 0 ;
                IterSubRestart = 0 ;
                IterQuad = 0 ;
                mp_begin = -1 ;
                memk_begin = 0 ;
                memk = 0 ;

                if ( PrintLevel >= 1 ) printf ("RESTART Sub-CG\n") ;

                /* search direction d = -Zk gsub, gsub = Zk' g, dsub = -gsub
                                 => d =  Zk dsub = SkF (Rk)^{-1} dsub */
                cg_scale0 (dsub, gsubtemp, -1, nsub) ;
                cg_copy0 (gsub, gsubtemp, nsub) ;
                cg_copy0 (vsub, dsub, nsub) ;
                cg_trisolve (vsub, Rk, mem, nsub, 1) ;
                /* rearrange and store in wsub */
                mp = SkFlast ;
                j = nsub - (mp+1) ;
                cg_copy0 (wsub, vsub+j, mp+1) ;
                cg_copy0 (wsub+(mp+1), vsub, j) ;
                cg_matvec (d, SkF, wsub, nsub, n, 1) ;

                dphi0 = -gsubnorm2 ; /* gsubnorm2 was calculated before */
                dnorm2 = gsubnorm2 ;
            }
            else  /* continue in subspace without restart */
            {
                mlast_sub = (mp_begin + IterSubRestart) % mem ;

                if (IterSubRestart > 0 ) /* not first iteration in subspace  */
                {
                    /* add new column to Yk memory,
                       calculate yty, Sk, Yk and SkYk */
                    spp = mlast_sub*mem ;
                    cg_scale0 (Sk+spp, dsub, alpha, nsub) ;
                    /* yty = (gsubtemp-gsub)'(gsubtemp-gsub),
                       set gsub = gsubtemp */
                    cg_Yk (Yk+spp, gsub, gsubtemp, &yty, nsub) ;
                    SkYk [mlast_sub] = alpha*(dphi - dphi0) ;
                    if ( yty > 0 )
                    {
                        scale = SkYk [mlast_sub]/yty ;
                    }
                }
                else
                {
                    yty = cg_dot0 (Yk+mlast_sub*mem, Yk+mlast_sub*mem, nsub) ;
                    if ( yty > 0 )
                    {
                        scale = SkYk [mlast_sub]/yty ;
                    }
                }

                /* calculate gsubtemp = H gsub */
                mp = mlast_sub ;
                /* memk = size of the L-BFGS memory in subspace */
                memk = CG_MIN (memk_begin + IterSubRestart, mem) ;
                l1 = CG_MIN (IterSubRestart, memk) ;
                /* l2 = number of triangular columns in Yk with a zero */
                l2 = memk - l1 ;
                /* l1 = number of dense column in Yk (excluding first) */
                l1++ ;
                l1 = CG_MIN (l1, memk) ;

                /* process dense columns */
                for (j = 0; j < l1; j++)
                {
                    mpp = mp*mem ;
                    t = cg_dot0 (Sk+mpp, gsubtemp, nsub)/SkYk[mp] ;
                    tau [mp] = t ;
                    /* update gsubtemp -= t*Yk+mpp */
                    cg_daxpy0 (gsubtemp, Yk+mpp, -t, nsub) ;
                    if ( mp == 0 ) mp = mem-1 ; else mp--;
                }

                /* process columns from triangular (Hessenberg) matrix */
                for (j = 1; j < l2; j++)
                {
                    mpp = mp*mem ;
                    t = cg_dot0 (Sk+mpp, gsubtemp, mp+1)/SkYk[mp] ;
                    tau [mp] = t ;
                    /* update gsubtemp -= t*Yk+mpp */
                    if ( mp == 0 && DenseCol1 )
                    {
                        cg_daxpy0 (gsubtemp, Yk+mpp, -t, nsub) ;
                    }
                    else
                    {
                        cg_daxpy0 (gsubtemp, Yk+mpp, -t, CG_MIN(mp+2,nsub)) ;
                    }
                    if ( mp == 0 ) mp = mem-1 ; else mp--;
                }
                cg_scale0 (gsubtemp, gsubtemp, scale, nsub) ;

                /* process columns from triangular (Hessenberg) matrix */
                for (j = 1; j < l2; j++)
                {
                    mp++ ;
                    if ( mp == mem ) mp = 0 ;
                    mpp = mp*mem ;
                    if ( mp == 0 && DenseCol1 )
                    {
                        t = cg_dot0 (Yk+mpp, gsubtemp, nsub)/SkYk[mp] ;
                    }
                    else
                    {
                        t = cg_dot0 (Yk+mpp, gsubtemp, CG_MIN(mp+2,nsub))/SkYk[mp];
                    }
                    /* update gsubtemp += (tau[mp]-t)*Sk+mpp */
                    cg_daxpy0 (gsubtemp, Sk+mpp, tau [mp] - t, mp+1) ;
                }

                /* process dense columns */
                for (j = 0; j < l1; j++)
                {
                    mp++ ;
                    if ( mp == mem ) mp = 0 ;
                    mpp = mp*mem ;
                    t = cg_dot0 (Yk+mpp, gsubtemp, nsub)/SkYk [mp] ;
                    /* update gsubtemp += (tau[mp]-t)*Sk+mpp */
                    cg_daxpy0 (gsubtemp, Sk+mpp, tau [mp] - t, nsub) ;
                } /* done computing H gsubtemp */

                /* compute d = Zk dsub = SkF (Rk)^{-1} dsub */
                cg_scale0 (dsub, gsubtemp, -1, nsub) ;
                cg_copy0 (vsub, dsub, nsub) ;
                cg_trisolve (vsub, Rk, mem, nsub, 1) ;
                /* rearrange and store in wsub */
                mp = SkFlast ;
                j = nsub - (mp+1) ;
                cg_copy0 (wsub, vsub+j, mp+1) ;
                cg_copy0 (wsub+(mp+1), vsub, j) ;

                cg_matvec (d, SkF, wsub, nsub, n, 1) ;
                dphi0 = -cg_dot0  (gsubtemp, gsub, nsub) ;
            }
        } /* end of subspace search direction */
        else  /* compute the search direction in the full space */
        {
            if ( Restart ) /*restart in fullspace*/
            {
                Restart = 0 ;
                IterRestart = 0 ;
                IterQuad = 0 ;
                if ( PrintLevel >= 1 ) printf ("RESTART CG\n") ;

                /* set x = xtemp */
                cg_copy (x, xtemp, n) ;

                if ( UseMemory )
                {
                   /* set g = gtemp, d = -g, compute infinity norm of g,
                      gnorm2 was already computed above */
                   gnorm = cg_update_inf (g, gtemp, d, n) ;
                }
                else
                {
                    /* set g = gtemp, d = -g, compute infinity and 2-norm of g*/
                    gnorm = cg_update_inf2 (g, gtemp, d, &gnorm2, n) ;
                }

                if ( cg_tol (gnorm, &Com) )
                {
                   status = 0 ;
                   goto Exit ;
                }
                dphi0 = -gnorm2 ;
                dnorm2 = gnorm2 ;
                beta = 0 ;
            }
            else if ( !FirstFull ) /* normal fullspace step*/
            {
                /* set x = xtemp */
                cg_copy (x, xtemp, n) ;

                /* set g = gtemp, compute gnorm = infinity norm of g,
                   ykyk = ||gtemp-g||_2^2, and ykgk = (gtemp-g) dot gnew */
                gnorm = cg_update_ykyk (g, gtemp, &ykyk, &ykgk, n) ;

                if ( cg_tol (gnorm, &Com) )
                {
                   status = 0 ;
                   goto Exit ;
                }

                dkyk = dphi - dphi0 ;
                if ( Parm->AdaptiveBeta ) t = 2. - 1/(0.1*QuadTrust + 1) ;
                else                      t = Parm->theta ;
                beta = (ykgk - t*dphi*ykyk/dkyk)/dkyk ;

                /* faster: initialize dnorm2 = gnorm2 at start, then
                           dnorm2 = gnorm2 + beta**2*dnorm2 - 2.*beta*dphi
                           gnorm2 = ||g_{k+1}||^2
                           dnorm2 = ||d_{k+1}||^2
                           dpi = g_{k+1}' d_k */

                /* lower bound for beta is BetaLower*d_k'g_k/ ||d_k||^2 */
                beta = CG_MAX (beta, Parm->BetaLower*dphi0/dnorm2) ;

                /* update search direction d = -g + beta*dold */
                if ( UseMemory )
                {
                    /* update search direction d = -g + beta*dold, and
                       compute 2-norm of d, 2-norm of g computed above */
                    dnorm2 = cg_update_d (d, g, beta, NULL, n) ;
                }
                else
                {
                    /* update search direction d = -g + beta*dold, and
                       compute 2-norms of d and g */
                    dnorm2 = cg_update_d (d, g, beta, &gnorm2, n) ;
                }

                dphi0 = -gnorm2 + beta*dphi ;
                if ( Parm->debug ) /* Check that dphi0 = d'g */
                {
                    t = 0 ;
                    for (i = 0; i < n; i++)  t = t + d [i]*g [i] ;
                    if ( fabs(t-dphi0) > Parm->debugtol*fabs(dphi0) )
                    {
                        printf("Warning, dphi0 != d'g!\n");
                        printf("dphi0:%13.6e, d'g:%13.6e\n",dphi0, t) ;
                    }
                }
            }
            else /* FirstFull = 1, precondition after leaving subspace */
            {
                /* set x = xtemp */
                cg_copy (x, xtemp, n) ;

                /* set g = gtemp, compute gnorm = infinity norm of g,
                   ykyk = ||gtemp-g||_2^2, and ykgk = (gtemp-g) dot gnew */
                gnorm = cg_update_ykyk (g, gtemp, &ykyk, &ykgk, n) ;

                if ( cg_tol (gnorm, &Com) )
                {
                   status = 0 ;
                   goto Exit ;
                }

                mlast_sub = (mp_begin + IterSubRestart) % mem ;
                /* save Sk */
                spp = mlast_sub*mem ;
                cg_scale0 (Sk+spp, dsub, alpha, nsub) ;
                /* calculate yty, save Yk, set gsub = gsubtemp */
                cg_Yk (Yk+spp, gsub, gsubtemp, &yty, nsub) ;
                ytg = cg_dot0  (Yk+spp, gsub, nsub) ;
                t = alpha*(dphi - dphi0) ;
                SkYk [mlast_sub] = t ;

                /* scale = t/ykyk ; */
                if ( yty > 0 )
                {
                    scale = t/yty ;
                }

                /* calculate gsubtemp = H gsub */
                mp = mlast_sub ;
                /* memk = size of the L-BFGS memory in subspace */
                memk = CG_MIN (memk_begin + IterSubRestart, mem) ;
                l1 = CG_MIN (IterSubRestart, memk) ;
                /* l2 = number of triangular columns in Yk with a zero */
                l2 = memk - l1 ;
                /* l1 = number of dense column in Yk (excluding first) */
                l1++ ;
                l1 = CG_MIN (l1, memk) ;

                /* process dense columns */
                for (j = 0; j < l1; j++)
                {
                    mpp = mp*mem ;
                    t = cg_dot0 (Sk+mpp, gsubtemp, nsub)/SkYk[mp] ;
                    tau [mp] = t ;
                    /* update gsubtemp -= t*Yk+mpp */
                    cg_daxpy0 (gsubtemp, Yk+mpp, -t, nsub) ;
                    if ( mp == 0 ) mp = mem-1 ; else mp--;
                }

                /* process columns from triangular (Hessenberg) matrix */
                for (j = 1; j < l2; j++)
                {
                    mpp = mp*mem ;
                    t = cg_dot0 (Sk+mpp, gsubtemp, mp+1)/SkYk[mp] ;
                    tau [mp] = t ;
                    /* update gsubtemp -= t*Yk+mpp */
                    if ( mp == 0 && DenseCol1 )
                    {
                        cg_daxpy0 (gsubtemp, Yk+mpp, -t, nsub) ;
                    }
                    else
                    {
                        cg_daxpy0 (gsubtemp, Yk+mpp, -t, CG_MIN(mp+2,nsub)) ;
                    }
                    if ( mp == 0 ) mp = mem-1 ; else mp--;
                }
                cg_scale0 (gsubtemp, gsubtemp, scale, nsub) ;

                /* process columns from triangular (Hessenberg) matrix */
                for (j = 1; j < l2; j++)
                {
                    mp++ ;
                    if ( mp == mem ) mp = 0 ;
                    mpp = mp*mem ;
                    if ( mp == 0 && DenseCol1 )
                    {
                        t = cg_dot0 (Yk+mpp, gsubtemp, nsub)/SkYk[mp] ;
                    }
                    else
                    {
                        t = cg_dot0 (Yk+mpp, gsubtemp, CG_MIN(mp+2,nsub))/SkYk[mp];
                    }
                    /* update gsubtemp += (tau[mp]-t)*Sk+mpp */
                    cg_daxpy0 (gsubtemp, Sk+mpp, tau [mp] - t, mp+1) ;
                }

                /* process dense columns */
                for (j = 0; j < l1; j++)
                {
                    mp++ ;
                    if ( mp == mem ) mp = 0 ;
                    mpp = mp*mem ;
                    t = cg_dot0 (Yk+mpp, gsubtemp, nsub)/SkYk [mp] ;
                    /* update gsubtemp += (tau[mp]-t)*Sk+mpp */
                    cg_daxpy0 (gsubtemp, Sk+mpp, tau [mp] - t, nsub) ;
                } /* done computing H gsubtemp */

                /* compute beta */
                dkyk = dphi - dphi0 ;
                if ( Parm->AdaptiveBeta ) t = 2. - 1/(0.1*QuadTrust + 1) ;
                else                      t = Parm->theta ;
                t1 = CG_MAX(ykyk-yty, 0) ; /* Theoretically t1 = ykyk-yty */
                if ( ykyk > 0 )
                {
                    scale = (alpha*dkyk)/ykyk ; /* = sigma */
                }
                beta = scale*((ykgk - ytg) - t*dphi*t1/dkyk)/dkyk ;
             /* beta = CG_MAX (beta, Parm->BetaLower*dphi0/dnorm2) ; */
                beta = CG_MAX (beta, Parm->BetaLower*(dphi0*alpha)/dkyk) ;

                /* compute search direction
                   d = -Zk (H - sigma)ghat - sigma g + beta d

                   Note: d currently contains last 2 terms so only need
                         to add the Zk term. Above gsubtemp = H ghat */

                /* form vsub = sigma ghat - H ghat = sigma ghat - gsubtemp */
                cg_scale0 (vsub, gsubtemp, -1, nsub) ;
                cg_daxpy0 (vsub, gsub, scale, nsub) ;
                cg_trisolve (vsub, Rk, mem, nsub, 1) ;

                /* rearrange vsub and store in wsub */
                mp = SkFlast ;
                j = nsub - (mp+1) ;
                cg_copy0 (wsub, vsub+j, mp+1) ;
                cg_copy0 (wsub+(mp+1), vsub, j) ;


                /* save old direction d in gtemp */
                cg_copy (gtemp, d, n) ;

                /* d = Zk (sigma - H)ghat */
                cg_matvec (d, SkF, wsub, nsub, n, 1) ;

                /* incorporate the new g and old d terms in new d */
                cg_daxpy (d, g, -scale, n) ;
                cg_daxpy (d, gtemp, beta, n) ;

                gHg = cg_dot0  (gsubtemp, gsub, nsub) ;
                t1 = CG_MAX(gnorm2 -gsubnorm2, 0) ;
                dphi0 = -gHg - scale*t1 + beta*dphi ;
                /* dphi0 = cg_dot (d, g, n) could be inaccurate */
                dnorm2 = cg_dot (d, d, n) ;
            }  /* end of preconditioned step */
        }  /* search direction has been computed */

        /* test for slow convergence */
        if ( (f < fbest) || (gnorm2 < gbest) )
        {
            nslow = 0 ;
            if ( f < fbest ) fbest = f ;
            if ( gnorm2 < gbest ) gbest = gnorm2 ;
        }
        else nslow++ ;
        if ( nslow > slowlimit )
        {
            status = 9 ;
            goto Exit ;
        }

        if ( PrintLevel >= 1 )
        {
            printf ("\niter: %5i f = %13.6e gnorm = %13.6e memk: %zu "
                    "Subspace: %zu\n", (int) iter, f, gnorm, memk, Subspace) ;
        }

        if ( Parm->debug )
        {
            if ( f > Com.f0 + Parm->debugtol*Ck )
            {
                status = 8 ;
                goto Exit ;
            }
        }

        if ( dphi0 > 0 )
        {
           status = 5 ;
           goto Exit ;
        }
    }
    status = 2 ;
Exit:
    if ( status == 11 ) gnorm = CG_FLOAT_INF ; /* function is undefined */
    if ( Stat != NULL )
    {
        Stat->nfunc = Com.nf ;
        Stat->ngrad = Com.ng ;
        Stat->iter = iter ;
        Stat->NumSub = NumSub ;
        Stat->IterSub = IterSub ;
        if ( status < 10 ) /* function was evaluated */
        {
            Stat->f = f ;
            Stat->gnorm = gnorm ;
        }
    }
    /* If there was an error, the function was evaluated, and its value
       is defined, then copy the most recent x value to the returned x
       array and evaluate the norm of the gradient at this point */
    if ( (status > 0) && (status < 10) && gtemp != NULL )
    {
        cg_copy (x, xtemp, n) ;
        gnorm = 0 ;
        for (i = 0; i < n; i++)
        {
            g [i] = gtemp [i] ;
            t = fabs (g [i]) ;
            gnorm = CG_MAX (gnorm, t) ;
        }
        if ( Stat != NULL ) Stat->gnorm = gnorm ;
    }
    if ( Parm->PrintFinal || PrintLevel >= 1 )
    {
        const char mess1 [] = "Possible causes of this error message:" ;
        const char mess2 [] = "   - your tolerance may be too strict: "
                              "grad_tol = " ;
        const char mess3 [] = "Line search fails" ;
        const char mess4 [] = "   - your gradient routine has an error" ;
        const char mess5 [] = "   - the parameter epsilon is too small" ;

        printf ("\nTermination status: %i\n", status) ;

        if ( status && NegDiag )
        {
            printf ("Parameter eta2 may be too small\n") ;
        }

        if ( status == 0 )
        {
            printf ("Convergence tolerance for gradient satisfied\n\n") ;
        }
        else if ( status == 1 )
        {
            printf ("Terminating since change in function value "
                    "<= feps*|f|\n\n") ;
        }
        else if ( status == 2 )
        {
            printf ("Number of iterations exceed specified limit\n") ;
            printf ("Iterations: %10.0f maxit: %10.0f\n",
                    (CG_FLOAT) iter, (CG_FLOAT) maxit) ;
            printf ("%s\n", mess1) ;
            printf ("%s %e\n\n", mess2, grad_tol) ;
        }
        else if ( status == 3 )
        {
            printf ("Slope always negative in line search\n") ;
            printf ("%s\n", mess1) ;
            printf ("   - your cost function has an error\n") ;
            printf ("%s\n\n", mess4) ;
        }
        else if ( status == 4 )
        {
            printf ("Line search fails, too many iterations\n") ;
            printf ("%s\n", mess1) ;
            printf ("%s %e\n\n", mess2, grad_tol) ;
        }
        else if ( status == 5 )
        {
            printf ("Search direction not a descent direction\n\n") ;
        }
        else if ( status == 6 ) /* line search fails, excessive eps updating */
        {
            printf ("%s due to excessive updating of eps\n", mess3) ;
            printf ("%s\n", mess1) ;
            printf ("%s %e\n", mess2, grad_tol) ;
            printf ("%s\n\n", mess4) ;
        }
        else if ( status == 7 ) /* line search fails */
        {
            printf ("%s\n%s\n", mess3, mess1) ;
            printf ("%s %e\n", mess2, grad_tol) ;
            printf ("%s\n%s\n\n", mess4, mess5) ;
        }
        else if ( status == 8 )
        {
            printf ("Debugger is on, function value does not improve\n") ;
            printf ("new value: %25.16e old value: %25.16e\n\n", f, Com.f0) ;
        }
        else if ( status == 9 )
        {
            printf ("%zu iterations without strict improvement in cost "
                    "or gradient\n\n", nslow) ;
        }
        else if ( status == 10 )
        {
            printf ("Insufficient memory for specified problem dimension %zu"
                    " in cg_descent\n", n ) ;
        }
        else if ( status == 11 )
        {
            printf ("Function nan and could not be repaired\n\n") ;
        }
        else if ( status == 12 )
        {
            printf ("memory = %zu is an invalid choice for parameter memory\n",
                     Parm->memory) ;
            printf ("memory should be either 0 or greater than 2\n\n") ;
        }

        printf ("maximum norm for gradient: %13.6e\n", gnorm) ;
        printf ("function value:            %13.6e\n\n", f) ;
        printf ("iterations:              %zu\n", iter) ;
        printf ("function evaluations:    %zu\n", Com.nf) ;
        printf ("gradient evaluations:    %zu\n", Com.ng) ;
        if ( IterSub > 0 )
        {
            printf ("subspace iterations:     %zu\n", IterSub) ;
            printf ("number of subspaces:     %zu\n", NumSub) ;
        }
        printf ("===================================\n\n") ;
    }
    if ( Work == NULL && work != NULL ) free (work) ;
    return (status) ;
}

/* =========================================================================
   ==== cg_Wolfe ===========================================================
   =========================================================================
   Check whether the Wolfe or the approximate Wolfe conditions are satisfied
   ========================================================================= */
static int cg_Wolfe
(
    CG_FLOAT   alpha, /* stepsize */
    CG_FLOAT       f, /* function value associated with stepsize alpha */
    CG_FLOAT    dphi, /* derivative value associated with stepsize alpha */
    cg_com    *Com  /* cg com */
)
{
    if ( dphi >= Com->wolfe_lo )
    {
        /* test original Wolfe conditions */
        if ( f - Com->f0 <= alpha*Com->wolfe_hi )
        {
            if ( Com->Parm->PrintLevel >= 2 )
            {
                printf ("Wolfe conditions hold\n") ;
/*              printf ("wolfe f: %25.15e f0: %25.15e df: %25.15e\n",
                         f, Com->f0, dphi) ;*/
            }
            return (1) ;
        }
        /* test approximate Wolfe conditions */
        else if ( Com->AWolfe )
        {
/*          if ( Com->Parm->PrintLevel >= 2 )
            {
                printf ("f:    %e fpert:    %e ", f, Com->fpert) ;
                if ( f > Com->fpert ) printf ("(fail)\n") ;
                else                  printf ("(OK)\n") ;
                printf ("dphi: %e hi bound: %e ", dphi, Com->awolfe_hi) ;
                if ( dphi > Com->awolfe_hi ) printf ("(fail)\n") ;
                else                         printf ("(OK)\n") ;
            }*/
            if ( (f <= Com->fpert) && (dphi <= Com->awolfe_hi) )
            {
                if ( Com->Parm->PrintLevel >= 2 )
                {
                    printf ("Approximate Wolfe conditions hold\n") ;
/*                  printf ("f: %25.15e fpert: %25.15e dphi: %25.15e awolf_hi: "
                           "%25.15e\n", f, Com->fpert, dphi, Com->awolfe_hi) ;*/
                }
                return (1) ;
            }
        }
    }
/*  else if ( Com->Parm->PrintLevel >= 2 )
    {
        printf ("dphi: %e lo bound: %e (fail)\n", dphi, Com->wolfe_lo) ;
    }*/
    return (0) ;
}

/* =========================================================================
   ==== cg_tol =============================================================
   =========================================================================
   Check for convergence
   ========================================================================= */
static cg_boolean cg_tol
(
    CG_FLOAT gnorm, /* gradient sup-norm */
    cg_com    *Com  /* cg com */
)
{
    /* StopRule = T => |grad|_infty <= max (tol, |grad|_infty*StopFact)
                  F => |grad|_infty <= tol*(1+|f|)) */
    if ( Com->Parm->StopRule )
    {
        if ( gnorm <= Com->tol ) return CG_TRUE ;
    }
    else if ( gnorm <= Com->tol*(1.0 + fabs (Com->f)) ) return CG_TRUE ;
    return CG_FALSE ;
}

/* =========================================================================
   ==== cg_line ============================================================
   =========================================================================
   Approximate Wolfe line search routine
   Return:
      -2 (function nan)
       0 (Wolfe or approximate Wolfe conditions satisfied)
       3 (slope always negative in line search)
       4 (number line search iterations exceed nline)
       6 (excessive updating of eps)
       7 (Wolfe conditions never satisfied)
   ========================================================================= */
static int cg_line
(
    cg_com   *Com /* cg com structure */
)
{
    cg_boolean AWolfe;
    int PrintLevel, qb, qb0, status, toggle ;
    CG_FLOAT alpha, a, a1, a2, b, bmin, B, da, db, d0, d1, d2, dB, df, f, fa, fb,
           fB, a0, b0, da0, db0, fa0, fb0, width, rho ;
    char *s1, *s2, *fmt1, *fmt2 ;
    size_t iter, ngrow ;
		cg_parameter *Parm ;

    AWolfe = Com->AWolfe ;
    Parm = Com->Parm ;
    PrintLevel = Parm->PrintLevel ;
    if ( PrintLevel >= 1 )
    {
        if ( AWolfe )
        {
            printf ("Approximate Wolfe line search\n") ;
            printf ("=============================\n") ;
        }
        else
        {
            printf ("Wolfe line search\n") ;
            printf ("=================\n") ;
        }
    }

    /* evaluate function or gradient at Com->alpha (starting guess) */
    if ( Com->QuadOK )
    {
        status = cg_evaluate ("fg", "y", Com) ;
        fb = Com->f ;
        if ( !AWolfe ) fb -= Com->alpha*Com->wolfe_hi ;
        qb = 1 ; /* function value at b known */
    }
    else
    {
        status = cg_evaluate ("g", "y", Com) ;
        qb = 0 ;
    }
    if ( status ) return (status) ; /* function is undefined */
    b = Com->alpha ;

    if ( AWolfe )
    {
        db = Com->df ;
        d0 = da = Com->df0 ;
    }
    else
    {
        db = Com->df - Com->wolfe_hi ;
        d0 = da = Com->df0 - Com->wolfe_hi ;
    }
    a = 0 ;
    a1 = 0 ;
    d1 = d0 ;
    fa = Com->f0 ;
    if ( PrintLevel >= 1 )
    {
        fmt1 = "%9s %2s a: %13.6e b: %13.6e fa: %13.6e fb: %13.6e "
               "da: %13.6e db: %13.6e\n" ;
        fmt2 = "%9s %2s a: %13.6e b: %13.6e fa: %13.6e fb:  x.xxxxxxxxxx "
               "da: %13.6e db: %13.6e\n" ;
        if ( Com->QuadOK ) s2 = "OK" ;
        else               s2 = "" ;
        if ( qb ) printf (fmt1, "start    ", s2, a, b, fa, fb, da, db);
        else      printf (fmt2, "start    ", s2, a, b, fa, da, db) ;
    }

    /* if a quadratic interpolation step performed, check Wolfe conditions */
    if ( (Com->QuadOK) && (Com->f <= Com->f0) )
    {
        if ( cg_Wolfe (b, Com->f, Com->df, Com) ) return (0) ;
    }

    /* if a Wolfe line search and the Wolfe conditions have not been satisfied*/
    if ( !AWolfe ) Com->Wolfe = CG_TRUE ;

    /*Find initial interval [a,b] such that
      da <= 0, db >= 0, fa <= fpert = [(f0 + eps*fabs (f0)) or (f0 + eps)] */
    rho = Com->rho ;
    ngrow = 1 ;
    while ( db < 0 )
    {
        if ( !qb )
        {
            status = cg_evaluate ("f", "n", Com) ;
            if ( status ) return (status) ;
            if ( AWolfe ) fb = Com->f ;
            else          fb = Com->f - b*Com->wolfe_hi ;
            qb = 1 ;
        }
        if ( fb > Com->fpert ) /* contract interval [a, b] */
        {
            status = cg_contract (&a, &fa, &da, &b, &fb, &db, Com) ;
            if ( status == 0 ) return (0) ;   /* Wolfe conditions hold */
            if ( status == -2 ) goto Line ; /* db >= 0 */
            if ( Com->neps > Parm->neps ) return (6) ;
        }

        /* expansion phase */
        ngrow++ ;
        if ( ngrow > Parm->ntries ) return (3) ;
        /* update interval (a replaced by b) */
        a = b ;
        fa = fb ;
        da = db ;
        /* store old values of a and corresponding derivative */
        d2 = d1 ;
        d1 = da ;
        a2 = a1 ;
        a1 = a ;

        bmin = rho*b ;
        if ( (ngrow == 2) || (ngrow == 3) || (ngrow == 6) )
        {
            if ( d1 > d2 )
            {
                if ( ngrow == 2 )
                {
                    b = a1 - (a1-a2)*(d1/(d1-d2)) ;
                }
                else
                {
                    if ( (d1-d2)/(a1-a2) >= (d2-d0)/a2 )
                    {
                        /* convex derivative, secant overestimates minimizer */
                        b = a1 - (a1-a2)*(d1/(d1-d2)) ;
                    }
                    else
                    {
                        /* concave derivative, secant underestimates minimizer*/
                        b = a1 - Parm->SecantAmp*(a1-a2)*(d1/(d1-d2)) ;
                    }
                }
                /* safeguard growth */
                b = CG_MIN (b, Parm->ExpandSafe*a1) ;
            }
            else rho *= Parm->RhoGrow ;
        }
        else rho *= Parm->RhoGrow ;
        b = CG_MAX (bmin, b) ;
        Com->alphaold = Com->alpha ;
        Com->alpha = b ;
        status = cg_evaluate ("g", "p", Com) ;
        if ( status ) return (status) ;
        b = Com->alpha ;
        qb = 0 ;
        if ( AWolfe ) db = Com->df ;
        else          db = Com->df - Com->wolfe_hi ;
        if ( PrintLevel >= 2 )
        {
            if ( Com->QuadOK ) s2 = "OK" ;
            else               s2 = "" ;
            printf (fmt2, "expand   ", s2, a, b, fa, da, db) ;
        }
    }

    /* we now have fa <= fpert, da >= 0, db <= 0 */
Line:
    toggle = 0 ;
    width = b - a ;
    qb0 = 0 ;
    for (iter = 0; iter < Parm->nline; iter++)
    {
        /* determine the next iterate */
        if ( (toggle == 0) || ((toggle == 2) && ((b-a) <= width)) )
        {
            Com->QuadOK = 1 ;
            if ( Com->UseCubic && qb )
            {
                s1 = "cubic    " ;
                alpha = cg_cubic (a, fa, da, b, fb, db) ;
                if ( alpha < 0 ) /* use secant method */
                {
                    s1 = "secant   " ;
                    if      ( -da < db ) alpha = a - (a-b)*(da/(da-db)) ;
                    else if ( da != db ) alpha = b - (a-b)*(db/(da-db)) ;
                    else                 alpha = -1. ;
                }
            }
            else
            {
                s1 = "secant   " ;
                if      ( -da < db ) alpha = a - (a-b)*(da/(da-db)) ;
                else if ( da != db ) alpha = b - (a-b)*(db/(da-db)) ;
                else                 alpha = -1. ;
            }
            width = Parm->gamma*(b - a) ;
        }
        else if ( toggle == 1 ) /* iteration based on smallest value*/
        {
            Com->QuadOK = 1 ;
            if ( Com->UseCubic )
            {
                s1 = "cubic    " ;
                if ( Com->alpha == a ) /* a is most recent iterate */
                {
                    alpha = cg_cubic (a0, fa0, da0, a, fa, da) ;
                }
                else if ( qb0 )        /* b is most recent iterate */
                {
                    alpha = cg_cubic (b, fb, db, b0, fb0, db0) ;
                }
                else alpha = -1. ;

                /* if alpha no good, use cubic between a and b */
                if ( (alpha <= a) || (alpha >= b) )
                {
                    if ( qb ) alpha = cg_cubic (a, fa, da, b, fb, db) ;
                    else alpha = -1. ;
                }

                /* if alpha still no good, use secant method */
                if ( alpha < 0 )
                {
                    s1 = "secant   " ;
                    if      ( -da < db ) alpha = a - (a-b)*(da/(da-db)) ;
                    else if ( da != db ) alpha = b - (a-b)*(db/(da-db)) ;
                    else                 alpha = -1. ;
                }
            }
            else /* ( use secant ) */
            {
                s1 = "secant   " ;
                if ( (Com->alpha == a) && (da > da0) ) /* use a0 if possible */
                {
                    alpha = a - (a-a0)*(da/(da-da0)) ;
                }
                else if ( db < db0 )                   /* use b0 if possible */
                {
                    alpha = b - (b-b0)*(db/(db-db0)) ;
                }
                else /* secant based on a and b */
                {
                    if      ( -da < db ) alpha = a - (a-b)*(da/(da-db)) ;
                    else if ( da != db ) alpha = b - (a-b)*(db/(da-db)) ;
                    else                 alpha = -1. ;
                }

                if ( (alpha <= a) || (alpha >= b) )
                {
                    if      ( -da < db ) alpha = a - (a-b)*(da/(da-db)) ;
                    else if ( da != db ) alpha = b - (a-b)*(db/(da-db)) ;
                    else                 alpha = -1. ;
                }
            }
        }
        else
        {
            alpha = .5*(a+b) ; /* use bisection if b-a decays slowly */
            s1 = "bisection" ;
            Com->QuadOK = 0 ;
        }

        if ( (alpha <= a) || (alpha >= b) )
        {
            alpha = .5*(a+b) ;
            s1 = "bisection" ;
            if ( (alpha == a) || (alpha == b) ) return (7) ;
            Com->QuadOK = 0 ; /* bisection was used */
        }

        if ( toggle == 0 ) /* save values for next iteration */
        {
            a0 = a ;
            b0 = b ;
            da0 = da ;
            db0 = db ;
            fa0 = fa ;
            if ( qb )
            {
                fb0 = fb ;
                qb0 = 1 ;
            }
        }

        toggle++ ;
        if ( toggle > 2 ) toggle = 0 ;

        Com->alpha = alpha ;
        status = cg_evaluate ("fg", "n", Com) ;
        if ( status ) return (status) ;
        Com->alpha = alpha ;
        f = Com->f ;
        df = Com->df ;
        if ( Com->QuadOK )
        {
            if ( cg_Wolfe (alpha, f, df, Com) )
            {
                if ( PrintLevel >= 2 )
                {
                    printf ("             a: %13.6e f: %13.6e df: %13.6e %1s\n",
                             alpha, f, df, s1) ;
                }
                return (0) ;
            }
        }
        if ( !AWolfe )
        {
            f -= alpha*Com->wolfe_hi ;
            df -= Com->wolfe_hi ;
        }
        if ( df >= 0 )
        {
            b = alpha ;
            fb = f ;
            db = df ;
            qb = 1 ;
        }
        else if ( f <= Com->fpert )
        {
            a = alpha ;
            da = df ;
            fa = f ;
        }
        else
        {
            B = b ;
            if ( qb ) fB = fb ;
            dB = db ;
            b = alpha ;
            fb = f ;
            db = df ;
            /* contract interval [a, alpha] */
            status = cg_contract (&a, &fa, &da, &b, &fb, &db, Com) ;
            if ( status == 0 ) return (0) ;
            if ( status == -1 ) /* eps reduced, use [a, b] = [alpha, b] */
            {
                if ( Com->neps > Parm->neps ) return (6) ;
                a = b ;
                fa = fb ;
                da = db ;
                b = B ;
                if ( qb ) fb = fB ;
                db = dB ;
            }
            else qb = 1 ;
        }
        if ( PrintLevel >= 2 )
        {
            if ( Com->QuadOK ) s2 = "OK" ;
            else               s2 = "" ;
            if ( !qb ) printf (fmt2, s1, s2, a, b, fa, da, db) ;
            else       printf (fmt1, s1, s2, a, b, fa, fb, da, db) ;
        }
    }
    return (4) ;
}

/* =========================================================================
   ==== cg_contract ========================================================
   =========================================================================
   The input for this routine is an interval [a, b] with the property that
   fa <= fpert, da >= 0, db >= 0, and fb >= fpert. The returned status is

  11  function or derivative not defined
   0  if the Wolfe conditions are satisfied
  -1  if a new value for eps is generated with the property that for the
      corresponding fpert, we have fb <= fpert
  -2  if a subinterval, also denoted [a, b], is generated with the property
      that fa <= fpert, da >= 0, and db <= 0

   NOTE: The input arguments are unchanged when status = -1
   ========================================================================= */
static int cg_contract
(
    CG_FLOAT    *A, /* left side of bracketing interval */
    CG_FLOAT   *fA, /* function value at a */
    CG_FLOAT   *dA, /* derivative at a */
    CG_FLOAT    *B, /* right side of bracketing interval */
    CG_FLOAT   *fB, /* function value at b */
    CG_FLOAT   *dB, /* derivative at b */
    cg_com  *Com  /* cg com structure */
)
{
    cg_boolean AWolfe;
    int PrintLevel, toggle, status ;
    size_t iter;
    CG_FLOAT a, alpha, b, old, da, db, df, dold, f, fa, fb, f1, fold, t, width ;
    char *s ;
    cg_parameter *Parm ;

    AWolfe = Com->AWolfe ;
    Parm = Com->Parm ;
    PrintLevel = Parm->PrintLevel ;
    a = *A ;
    fa = *fA ;
    da = *dA ;
    b = *B ;
    fb = *fB ;
    db = *dB ;
    f1 = fb ;
    toggle = 0 ;
    width = 0 ;
    for (iter = 0; iter < Parm->nshrink; iter++)
    {
        if ( (toggle == 0) || ((toggle == 2) && ((b-a) <= width)) )
        {
            /* cubic based on bracketing interval */
            alpha = cg_cubic (a, fa, da, b, fb, db) ;
            toggle = 0 ;
            width = Parm->gamma*(b-a) ;
            if ( iter ) Com->QuadOK = 1 ; /* at least 2 cubic iterations */
        }
        else if ( toggle == 1 )
        {
            Com->QuadOK = 1 ;
            /* cubic based on most recent iterate and smallest value */
            if ( old < a ) /* a is most recent iterate */
            {
                alpha = cg_cubic (a, fa, da, old, fold, dold) ;
            }
            else           /* b is most recent iterate */
            {
                alpha = cg_cubic (a, fa, da, b, fb, db) ;
            }
        }
        else
        {
            alpha = .5*(a+b) ; /* use bisection if b-a decays slowly */
            Com->QuadOK = 0 ;
        }

        if ( (alpha <= a) || (alpha >= b) )
        {
            alpha = .5*(a+b) ;
            Com->QuadOK = 0 ; /* bisection was used */
        }

        toggle++ ;
        if ( toggle > 2 ) toggle = 0 ;

        Com->alpha = alpha ;
        status = cg_evaluate ("fg", "n", Com) ;
        if ( status ) return (status) ;
        f = Com->f ;
        df = Com->df ;

        if ( Com->QuadOK )
        {
            if ( cg_Wolfe (alpha, f, df, Com) ) return (0) ;
        }
        if ( !AWolfe )
        {
            f -= alpha*Com->wolfe_hi ;
            df -= Com->wolfe_hi ;
        }
        if ( df >= 0 )
        {
            *B = alpha ;
            *fB = f ;
            *dB = df ;
            *A = a ;
            *fA = fa ;
            *dA = da ;
            return (-2) ;
        }
        if ( f <= Com->fpert ) /* update a using alpha */
        {
            old = a ;
            a = alpha ;
            fold = fa ;
            fa = f ;
            dold = da ;
            da = df ;
        }
        else                     /* update b using alpha */
        {
            old = b ;
            b = alpha ;
            fb = f ;
            db = df ;
        }
        if ( PrintLevel >= 2 )
        {
            if ( Com->QuadOK ) s = "OK" ;
            else               s = "" ;
            printf ("contract  %2s a: %13.6e b: %13.6e fa: %13.6e fb: "
                    "%13.6e da: %13.6e db: %13.6e\n", s, a, b, fa, fb, da, db) ;
        }
    }

    /* see if the cost is small enough to change the PertRule */
    if ( fabs (fb) <= Com->SmallCost ) Com->PertRule = 0 ;

    /* increase eps if slope is negative after Parm->nshrink iterations */
    t = Com->f0 ;
    if ( Com->PertRule )
    {
        if ( t != 0 )
        {
            Com->eps = Parm->egrow*(f1-t)/fabs (t) ;
            Com->fpert = t + fabs (t)*Com->eps ;
        }
        else Com->fpert = 2.*f1 ;
    }
    else
    {
        Com->eps = Parm->egrow*(f1-t) ;
        Com->fpert = t + Com->eps ;
    }
    if ( PrintLevel >= 1 )
    {
        printf ("--increase eps: %e fpert: %e\n", Com->eps, Com->fpert) ;
    }
    Com->neps++ ;
    return (-1) ;
}

/* =========================================================================
   ==== cg_fg_evaluate =====================================================
   Evaluate the function and/or gradient.  Also, possibly check if either is nan
   and if so, then reduce the stepsize. Only used at the start of an iteration.
   Return:
      11 (function nan)
       0 (successful evaluation)
   =========================================================================*/

static int cg_evaluate
(
    char    *what, /* fg = evaluate func and grad, g = grad only,f = func only*/
    char     *nan, /* y means check function/derivative values for nan */
    cg_com   *Com
)
{
    size_t n, i ;
    CG_FLOAT alpha, *d, *gtemp, *x, *xtemp ;
    cg_parameter *Parm ;
    Parm = Com->Parm ;
    n = Com->n ;
    x = Com->x ;
    d = Com->d ;
    xtemp = Com->xtemp ;
    gtemp = Com->gtemp ;
    alpha = Com->alpha ;
    /* check to see if values are nan */
    if ( !strcmp (nan, "y") || !strcmp (nan, "p") )
    {
        if ( !strcmp (what, "f") ) /* compute function */
        {
            cg_step (xtemp, x, d, alpha, n) ;
            /* provisional function value */
            Com->f = Com->cg_value (xtemp, n CG_CUSTOM_ARGUMENT(Com)) ;
            Com->nf++ ;

            /* reduce stepsize if function value is nan */
            if ( (Com->f != Com->f) || (Com->f >= CG_FLOAT_INF) || (Com->f <= -CG_FLOAT_INF) )
            {
                for (i = 0; i < Parm->ntries; i++)
                {
                    if ( !strcmp (nan, "p") ) /* contract from good alpha */
                    {
                        alpha = Com->alphaold + .8*(alpha - Com->alphaold) ;
                    }
                    else                      /* multiply by nan_decay */
                    {
                        alpha *= Parm->nan_decay ;
                    }
                    cg_step (xtemp, x, d, alpha, n) ;
                    Com->f = Com->cg_value (xtemp, n CG_CUSTOM_ARGUMENT(Com)) ;
                    Com->nf++ ;
                    if ( (Com->f == Com->f) && (Com->f < CG_FLOAT_INF) &&
                         (Com->f > -CG_FLOAT_INF) ) break ;
                }
                if ( i == Parm->ntries ) return (11) ;
            }
            Com->alpha = alpha ;
        }
        else if ( !strcmp (what, "g") ) /* compute gradient */
        {
            cg_step (xtemp, x, d, alpha, n) ;
            Com->cg_grad (gtemp, xtemp, n CG_CUSTOM_ARGUMENT(Com)) ;
            Com->ng++ ;
            Com->df = cg_dot (gtemp, d, n) ;
            /* reduce stepsize if derivative is nan */
            if ( (Com->df != Com->df) || (Com->df >= CG_FLOAT_INF) || (Com->df <= -CG_FLOAT_INF) )
            {
                for (i = 0; i < Parm->ntries; i++)
                {
                    if ( !strcmp (nan, "p") ) /* contract from good alpha */
                    {
                        alpha = Com->alphaold + .8*(alpha - Com->alphaold) ;
                    }
                    else                      /* multiply by nan_decay */
                    {
                        alpha *= Parm->nan_decay ;
                    }
                    cg_step (xtemp, x, d, alpha, n) ;
                    Com->cg_grad (gtemp, xtemp, n CG_CUSTOM_ARGUMENT(Com)) ;
                    Com->ng++ ;
                    Com->df = cg_dot (gtemp, d, n) ;
                    if ( (Com->df == Com->df) && (Com->df < CG_FLOAT_INF) &&
                         (Com->df > -CG_FLOAT_INF) ) break ;
                }
                if ( i == Parm->ntries ) return (11) ;
                Com->rho = Parm->nan_rho ;
            }
            else Com->rho = Parm->rho ;
            Com->alpha = alpha ;
        }
        else                            /* compute function and gradient */
        {
            cg_step (xtemp, x, d, alpha, n) ;
            if ( Com->cg_valgrad != NULL )
            {
                Com->f = Com->cg_valgrad (gtemp, xtemp, n CG_CUSTOM_ARGUMENT(Com)) ;
            }
            else
            {
                Com->cg_grad (gtemp, xtemp, n CG_CUSTOM_ARGUMENT(Com)) ;
                Com->f = Com->cg_value (xtemp, n CG_CUSTOM_ARGUMENT(Com)) ;
            }
            Com->df = cg_dot (gtemp, d, n) ;
            Com->nf++ ;
            Com->ng++ ;
            /* reduce stepsize if function value or derivative is nan */
            if ( (Com->df !=  Com->df) || (Com->f != Com->f) ||
                 (Com->df >=  CG_FLOAT_INF)     || (Com->f >= CG_FLOAT_INF)    ||
                 (Com->df <= -CG_FLOAT_INF)     || (Com->f <= -CG_FLOAT_INF))
            {
                for (i = 0; i < Parm->ntries; i++)
                {
                    if ( !strcmp (nan, "p") ) /* contract from good alpha */
                    {
                        alpha = Com->alphaold + .8*(alpha - Com->alphaold) ;
                    }
                    else                      /* multiply by nan_decay */
                    {
                        alpha *= Parm->nan_decay ;
                    }
                    cg_step (xtemp, x, d, alpha, n) ;
                    if ( Com->cg_valgrad != NULL )
                    {
                        Com->f = Com->cg_valgrad (gtemp, xtemp, n CG_CUSTOM_ARGUMENT(Com)) ;
                    }
                    else
                    {
                        Com->cg_grad (gtemp, xtemp, n CG_CUSTOM_ARGUMENT(Com)) ;
                        Com->f = Com->cg_value (xtemp, n CG_CUSTOM_ARGUMENT(Com)) ;
                    }
                    Com->df = cg_dot (gtemp, d, n) ;
                    Com->nf++ ;
                    Com->ng++ ;
                    if ( (Com->df == Com->df) && (Com->f == Com->f) &&
                         (Com->df <  CG_FLOAT_INF)     && (Com->f <  CG_FLOAT_INF)    &&
                         (Com->df > -CG_FLOAT_INF)     && (Com->f > -CG_FLOAT_INF) ) break ;
                }
                if ( i == Parm->ntries ) return (11) ;
                Com->rho = Parm->nan_rho ;
            }
            else Com->rho = Parm->rho ;
            Com->alpha = alpha ;
        }
    }
    else                                /* evaluate without nan checking */
    {
        if ( !strcmp (what, "fg") )     /* compute function and gradient */
        {
            if ( alpha == 0 )        /* evaluate at x */
            {
                /* the following copy is not needed except when the code
                   is run using the MATLAB mex interface */
                cg_copy (xtemp, x, n) ;
                if ( Com->cg_valgrad != NULL )
                {
                    Com->f = Com->cg_valgrad (Com->g, xtemp, n CG_CUSTOM_ARGUMENT(Com)) ;
                }
                else
                {
                    Com->cg_grad (Com->g, xtemp, n CG_CUSTOM_ARGUMENT(Com)) ;
                    Com->f = Com->cg_value (xtemp, n CG_CUSTOM_ARGUMENT(Com)) ;
                }
            }
            else
            {
                cg_step (xtemp, x, d, alpha, n) ;
                if ( Com->cg_valgrad != NULL )
                {
                    Com->f = Com->cg_valgrad (gtemp, xtemp, n CG_CUSTOM_ARGUMENT(Com)) ;
                }
                else
                {
                    Com->cg_grad (gtemp, xtemp, n CG_CUSTOM_ARGUMENT(Com)) ;
                    Com->f = Com->cg_value (xtemp, n CG_CUSTOM_ARGUMENT(Com)) ;
                }
                Com->df = cg_dot (gtemp, d, n) ;
            }
            Com->nf++ ;
            Com->ng++ ;
            if ( (Com->df != Com->df) || (Com->f != Com->f) ||
                 (Com->df == CG_FLOAT_INF)     || (Com->f == CG_FLOAT_INF)    ||
                 (Com->df ==-CG_FLOAT_INF)     || (Com->f ==-CG_FLOAT_INF) ) return (11) ;
        }
        else if ( !strcmp (what, "f") ) /* compute function */
        {
            cg_step (xtemp, x, d, alpha, n) ;
            Com->f = Com->cg_value (xtemp, n CG_CUSTOM_ARGUMENT(Com)) ;
            Com->nf++ ;
            if ( (Com->f != Com->f) || (Com->f == CG_FLOAT_INF) || (Com->f ==-CG_FLOAT_INF) )
                return (11) ;
        }
        else
        {
            cg_step (xtemp, x, d, alpha, n) ;
            Com->cg_grad (gtemp, xtemp, n CG_CUSTOM_ARGUMENT(Com)) ;
            Com->df = cg_dot (gtemp, d, n) ;
            Com->ng++ ;
            if ( (Com->df != Com->df) || (Com->df == CG_FLOAT_INF) || (Com->df ==-CG_FLOAT_INF) )
                return (11) ;
        }
    }
    return (0) ;
}

/* =========================================================================
   ==== cg_cubic ===========================================================
   =========================================================================
   Compute the minimizer of a Hermite cubic. If the computed minimizer
   outside [a, b], return -1 (it is assumed that a >= 0).
   ========================================================================= */
static CG_FLOAT cg_cubic
(
    CG_FLOAT  a,
    CG_FLOAT fa, /* function value at a */
    CG_FLOAT da, /* derivative at a */
    CG_FLOAT  b,
    CG_FLOAT fb, /* function value at b */
    CG_FLOAT db  /* derivative at b */
)
{
    CG_FLOAT c, d1, d2, delta, t, v, w ;
    delta = b - a ;
    if ( delta == 0 ) return (a) ;
    v = da + db - 3.*(fb-fa)/delta ;
    t = v*v - da*db ;
    if ( t < 0 ) /* complex roots, use secant method */
    {
         if ( fabs (da) < fabs (db) ) c = a - (a-b)*(da/(da-db)) ;
         else if ( da != db )         c = b - (a-b)*(db/(da-db)) ;
         else                         c = -1 ;
         return (c) ;
    }

    if ( delta > 0 ) w = sqrt(t) ;
    else                w =-sqrt(t) ;
    d1 = da + v - w ;
    d2 = db + v + w ;
    if ( (d1 == 0) && (d2 == 0) ) return (-1.) ;
    if ( fabs (d1) >= fabs (d2) ) c = a + delta*da/d1 ;
    else                          c = b - delta*db/d2 ;
    return (c) ;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       Start of routines that could use the BLAS
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

/* =========================================================================
   ==== cg_matvec ==========================================================
   =========================================================================
   Compute y = A*x or A'*x where A is a dense rectangular matrix
   ========================================================================= */
static void cg_matvec
(
    CG_FLOAT  *y, /* product vector */
    CG_FLOAT  *A, /* dense matrix */
    CG_FLOAT  *x, /* input vector */
    size_t     n, /* number of columns of A */
    size_t     m, /* number of rows of A */
    cg_boolean w  /* T => y = A*x, F => y = A'*x */
)
{
/* if the blas have not been installed, then hand code the produce */
#ifdef CG_NOBLAS
    size_t j, l ;
    l = 0 ;
    if ( w )
    {
        cg_scale0 (y, A, x [0], (int) m) ;
        for (j = 1; j < n; j++)
        {
            l += m ;
            cg_daxpy0 (y, A+l, x [j], (int) m) ;
        }
    }
    else
    {
        for (j = 0; j < n; j++)
        {
            y [j] = cg_dot0 (A+l, x, (int) m) ;
            l += m ;
        }
    }
#endif

/* if the blas have been installed, then possibly call gdemv */
#ifndef CG_NOBLAS
    size_t j, l ;
    CG_BLAS_INT M, N ;
    if ( w || (!w && (m*n < CG_MATVEC_START)) )
    {
        l = 0 ;
        if ( w )
        {
            cg_scale (y, A, x [0], m) ;
            for (j = 1; j < n; j++)
            {
                l += m ;
                cg_daxpy (y, A+l, x [j], m) ;
            }
        }
        else
        {
            for (j = 0; j < n; j++)
            {
                y [j] = cg_dot0 (A+l, x, (int) m) ;
                l += m ;
            }
        }
    }
    else
    {
        M = (CG_BLAS_INT) m ;
        N = (CG_BLAS_INT) n ;
        /* only use transpose mult with blas
        CG_DGEMV ("n", &M, &N, one, A, &M, x, blas_one, zero, y, blas_one) ;*/
        CG_DGEMV ("t", &M, &N, one, A, &M, x, blas_one, zero, y, blas_one) ;
    }
#endif

    return ;
}

/* =========================================================================
   ==== cg_trisolve ========================================================
   =========================================================================
   Solve Rx = y or R'x = y where R is a dense upper triangular matrix
   ========================================================================= */
static void cg_trisolve
(
    CG_FLOAT  *x, /* right side on input, solution on output */
    CG_FLOAT  *R, /* dense matrix */
    size_t     m, /* leading dimension of R */
    size_t     n, /* dimension of triangular system */
    cg_boolean w  /* T => Rx = y, F => R'x = y */
)
{
    size_t i, l ;
    if ( w )
    {
        l = m*n ;
        for (i = n; i > 0; )
        {
            i-- ;
            l -= (m-i) ;
            x [i] /= R [l] ;
            l -= i ;
            cg_daxpy0 (x, R+l, -x [i], i) ;
        }
    }
    else
    {
        l = 0 ;
        for (i = 0; i < n; i++)
        {
            x [i] = (x [i] - cg_dot0 (x, R+l, i))/R [l+i] ;
            l += m ;
        }
    }

/* equivalent to:
    CG_BLAS_INT M, N ;
    M = (CG_BLAS_INT) m ;
    N = (CG_BLAS_INT) n ;
    if ( w ) CG_DTRSV ("u", "n", "n", &N, R, &M, x, blas_one) ;
    else     CG_DTRSV ("u", "t", "n", &N, R, &M, x, blas_one) ; */

    return ;
}

/* =========================================================================
   ==== cg_inf =============================================================
   =========================================================================
   Compute infinity norm of vector
   ========================================================================= */
static CG_FLOAT cg_inf
(
    CG_FLOAT *x, /* vector */
    size_t    n /* length of vector */
)
{
#ifdef CG_NOBLAS
    size_t i, n5 ;
    CG_FLOAT t ;
    t = 0 ;
    n5 = n % 5 ;

    for (i = 0; i < n5; i++) if ( t < fabs (x [i]) ) t = fabs (x [i]) ;
    for (; i < n; i += 5)
    {
        if ( t < fabs (x [i]  ) ) t = fabs (x [i]  ) ;
        if ( t < fabs (x [i+1]) ) t = fabs (x [i+1]) ;
        if ( t < fabs (x [i+2]) ) t = fabs (x [i+2]) ;
        if ( t < fabs (x [i+3]) ) t = fabs (x [i+3]) ;
        if ( t < fabs (x [i+4]) ) t = fabs (x [i+4]) ;
    }
    return (t) ;
#endif

#ifndef CG_NOBLAS
    size_t i, n5 ;
    CG_FLOAT t ;
    CG_BLAS_INT N ;
    if ( n < CG_IDACG_MAX_START )
    {
        t = 0 ;
        n5 = n % 5 ;

        for (i = 0; i < n5; i++) if ( t < fabs (x [i]) ) t = fabs (x [i]) ;
        for (; i < n; i += 5)
        {
            if ( t < fabs (x [i]  ) ) t = fabs (x [i]  ) ;
            if ( t < fabs (x [i+1]) ) t = fabs (x [i+1]) ;
            if ( t < fabs (x [i+2]) ) t = fabs (x [i+2]) ;
            if ( t < fabs (x [i+3]) ) t = fabs (x [i+3]) ;
            if ( t < fabs (x [i+4]) ) t = fabs (x [i+4]) ;
        }
        return (t) ;
    }
    else
    {
        N = (CG_BLAS_INT) n ;
        i = (size_t) CG_IDACG_MAX (&N, x, blas_one) ;
        return (fabs (x [i-1])) ; /* adjust for fortran indexing */
    }
#endif
}

/* =========================================================================
   ==== cg_scale0 ===========================================================
   =========================================================================
   compute y = s*x where s is a scalar
   ========================================================================= */
static void cg_scale0
(
    CG_FLOAT *y, /* output vector */
    CG_FLOAT *x, /* input vector */
    CG_FLOAT  s, /* scalar */
    size_t    n  /* length of vector */
)
{
    size_t i, n5 ;
    n5 = n % 5 ;
    if ( s == -1)
    {
       for (i = 0; i < n5; i++) y [i] = -x [i] ;
       for (; i < n;)
       {
           y [i] = -x [i] ;
           i++ ;
           y [i] = -x [i] ;
           i++ ;
           y [i] = -x [i] ;
           i++ ;
           y [i] = -x [i] ;
           i++ ;
           y [i] = -x [i] ;
           i++ ;
       }
    }
    else
    {
        for (i = 0; i < n5; i++) y [i] = s*x [i] ;
        for (; i < n;)
        {
            y [i] = s*x [i] ;
            i++ ;
            y [i] = s*x [i] ;
            i++ ;
            y [i] = s*x [i] ;
            i++ ;
            y [i] = s*x [i] ;
            i++ ;
            y [i] = s*x [i] ;
            i++ ;
        }
    }
    return ;
}

/* =========================================================================
   ==== cg_scale ===========================================================
   =========================================================================
   compute y = s*x where s is a scalar
   ========================================================================= */
static void cg_scale
(
    CG_FLOAT *y, /* output vector */
    CG_FLOAT *x, /* input vector */
    CG_FLOAT  s, /* scalar */
    size_t    n /* length of vector */
)
{
    size_t i, n5 ;
    n5 = n % 5 ;
    if ( y == x)
    {
#ifdef CG_NOBLAS
        for (i = 0; i < n5; i++) y [i] *= s ;
        for (; i < n;)
        {
            y [i] *= s ;
            i++ ;
            y [i] *= s ;
            i++ ;
            y [i] *= s ;
            i++ ;
            y [i] *= s ;
            i++ ;
            y [i] *= s ;
            i++ ;
        }
#endif
#ifndef CG_NOBLAS
        if ( n < CG_DSCAL_START )
        {
            for (i = 0; i < n5; i++) y [i] *= s ;
            for (; i < n;)
            {
                y [i] *= s ;
                i++ ;
                y [i] *= s ;
                i++ ;
                y [i] *= s ;
                i++ ;
                y [i] *= s ;
                i++ ;
                y [i] *= s ;
                i++ ;
            }
        }
        else
        {
            CG_BLAS_INT N ;
            N = (CG_BLAS_INT) n ;
            CG_DSCAL (&N, &s, x, blas_one) ;
        }
#endif
    }
    else
    {
        for (i = 0; i < n5; i++) y [i] = s*x [i] ;
        for (; i < n;)
        {
            y [i] = s*x [i] ;
            i++ ;
            y [i] = s*x [i] ;
            i++ ;
            y [i] = s*x [i] ;
            i++ ;
            y [i] = s*x [i] ;
            i++ ;
            y [i] = s*x [i] ;
            i++ ;
        }
    }
    return ;
}

/* =========================================================================
   ==== cg_daxpy0 ==========================================================
   =========================================================================
   Compute x = x + alpha d
   ========================================================================= */
static void cg_daxpy0
(
    CG_FLOAT     *x, /* input and output vector */
    CG_FLOAT     *d, /* direction */
    CG_FLOAT  alpha, /* stepsize */
    size_t        n  /* length of the vectors */
)
{
    size_t i, n5 ;
    n5 = n % 5 ;
    if (alpha == -1)
    {
        for (i = 0; i < n5; i++) x [i] -= d[i] ;
        for (; i < n; i += 5)
        {
            x [i]   -= d [i] ;
            x [i+1] -= d [i+1] ;
            x [i+2] -= d [i+2] ;
            x [i+3] -= d [i+3] ;
            x [i+4] -= d [i+4] ;
        }
    }
    else
    {
        for (i = 0; i < n5; i++) x [i] += alpha*d[i] ;
        for (; i < n; i += 5)
        {
            x [i]   += alpha*d [i] ;
            x [i+1] += alpha*d [i+1] ;
            x [i+2] += alpha*d [i+2] ;
            x [i+3] += alpha*d [i+3] ;
            x [i+4] += alpha*d [i+4] ;
        }
    }
    return ;
}

/* =========================================================================
   ==== cg_daxpy ===========================================================
   =========================================================================
   Compute x = x + alpha d
   ========================================================================= */
static void cg_daxpy
(
    CG_FLOAT     *x, /* input and output vector */
    CG_FLOAT     *d, /* direction */
    CG_FLOAT  alpha, /* stepsize */
    size_t         n  /* length of the vectors */
)
{
#ifdef CG_NOBLAS
    size_t i, n5 ;
    n5 = n % 5 ;
    if (alpha == -1)
    {
        for (i = 0; i < n5; i++) x [i] -= d[i] ;
        for (; i < n; i += 5)
        {
            x [i]   -= d [i] ;
            x [i+1] -= d [i+1] ;
            x [i+2] -= d [i+2] ;
            x [i+3] -= d [i+3] ;
            x [i+4] -= d [i+4] ;
        }
    }
    else
    {
        for (i = 0; i < n5; i++) x [i] += alpha*d[i] ;
        for (; i < n; i += 5)
        {
            x [i]   += alpha*d [i] ;
            x [i+1] += alpha*d [i+1] ;
            x [i+2] += alpha*d [i+2] ;
            x [i+3] += alpha*d [i+3] ;
            x [i+4] += alpha*d [i+4] ;
        }
    }
#endif

#ifndef CG_NOBLAS
    size_t i, n5 ;
    CG_BLAS_INT N ;
    if ( n < CG_DAXPY_START )
    {
        n5 = n % 5 ;
        if (alpha == -1)
        {
            for (i = 0; i < n5; i++) x [i] -= d[i] ;
            for (; i < n; i += 5)
            {
                x [i]   -= d [i] ;
                x [i+1] -= d [i+1] ;
                x [i+2] -= d [i+2] ;
                x [i+3] -= d [i+3] ;
                x [i+4] -= d [i+4] ;
            }
        }
        else
        {
            for (i = 0; i < n5; i++) x [i] += alpha*d[i] ;
            for (; i < n; i += 5)
            {
                x [i]   += alpha*d [i] ;
                x [i+1] += alpha*d [i+1] ;
                x [i+2] += alpha*d [i+2] ;
                x [i+3] += alpha*d [i+3] ;
                x [i+4] += alpha*d [i+4] ;
            }
        }
    }
    else
    {
        N = (CG_BLAS_INT) n ;
        CG_DAXPY (&N, &alpha, d, blas_one, x, blas_one) ;
    }
#endif

    return ;
}

/* =========================================================================
   ==== cg_dot0 ============================================================
   =========================================================================
   Compute dot product of x and y, vectors of length n
   ========================================================================= */
static CG_FLOAT cg_dot0
(
    CG_FLOAT *x, /* first vector */
    CG_FLOAT *y, /* second vector */
    size_t    n  /* length of vectors */
)
{
    size_t i, n5 ;
    CG_FLOAT t ;
    t = 0 ;
    if ( n <= 0 ) return (t) ;
    n5 = n % 5 ;
    for (i = 0; i < n5; i++) t += x [i]*y [i] ;
    for (; i < n; i += 5)
    {
        t += x [i]*y[i] + x [i+1]*y [i+1] + x [i+2]*y [i+2]
                        + x [i+3]*y [i+3] + x [i+4]*y [i+4] ;
    }
    return (t) ;
}

/* =========================================================================
   ==== cg_dot =============================================================
   =========================================================================
   Compute dot product of x and y, vectors of length n
   ========================================================================= */
static CG_FLOAT cg_dot
(
    CG_FLOAT *x, /* first vector */
    CG_FLOAT *y, /* second vector */
    size_t    n  /* length of vectors */
)
{
#ifdef CG_NOBLAS
    size_t i, n5 ;
    CG_FLOAT t ;
    t = 0 ;
    if ( n <= 0 ) return (t) ;
    n5 = n % 5 ;
    for (i = 0; i < n5; i++) t += x [i]*y [i] ;
    for (; i < n; i += 5)
    {
        t += x [i]*y[i] + x [i+1]*y [i+1] + x [i+2]*y [i+2]
                        + x [i+3]*y [i+3] + x [i+4]*y [i+4] ;
    }
    return (t) ;
#endif

#ifndef CG_NOBLAS
    size_t i, n5 ;
    CG_FLOAT t ;
    CG_BLAS_INT N ;
    if ( n < CG_DDOT_START )
    {
        t = 0 ;
        if ( n <= 0 ) return (t) ;
        n5 = n % 5 ;
        for (i = 0; i < n5; i++) t += x [i]*y [i] ;
        for (; i < n; i += 5)
        {
            t += x [i]*y[i] + x [i+1]*y [i+1] + x [i+2]*y [i+2]
                            + x [i+3]*y [i+3] + x [i+4]*y [i+4] ;
        }
        return (t) ;
    }
    else
    {
        N = (CG_BLAS_INT) n ;
        return (CG_DDOT (&N, x, blas_one, y, blas_one)) ;
    }
#endif
}

/* =========================================================================
   === cg_copy0 ============================================================
   =========================================================================
   Copy vector x into vector y
   ========================================================================= */
static void cg_copy0
(
    CG_FLOAT *y, /* output of copy */
    CG_FLOAT *x, /* input of copy */
    size_t    n  /* length of vectors */
)
{
    size_t i, n5 ;
    n5 = n % 5 ;
    for (i = 0; i < n5; i++) y [i] = x [i] ;
    for (; i < n; )
    {
        y [i] = x [i] ;
        i++ ;
        y [i] = x [i] ;
        i++ ;
        y [i] = x [i] ;
        i++ ;
        y [i] = x [i] ;
        i++ ;
        y [i] = x [i] ;
        i++ ;
    }
    return ;
}

/* =========================================================================
   === cg_copy =============================================================
   =========================================================================
   Copy vector x into vector y
   ========================================================================= */
static void cg_copy
(
    CG_FLOAT *y, /* output of copy */
    CG_FLOAT *x, /* input of copy */
    size_t    n  /* length of vectors */
)
{
#ifdef CG_NOBLAS
    size_t i, n5 ;
    n5 = n % 5 ;
    for (i = 0; i < n5; i++) y [i] = x [i] ;
    for (; i < n; )
    {
        y [i] = x [i] ;
        i++ ;
        y [i] = x [i] ;
        i++ ;
        y [i] = x [i] ;
        i++ ;
        y [i] = x [i] ;
        i++ ;
        y [i] = x [i] ;
        i++ ;
    }
#endif

#ifndef CG_NOBLAS
    size_t i, n5 ;
    CG_BLAS_INT N ;
    if ( n < CG_DCOPY_START )
    {
        n5 = n % 5 ;
        for (i = 0; i < n5; i++) y [i] = x [i] ;
        for (; i < n; )
        {
            y [i] = x [i] ;
            i++ ;
            y [i] = x [i] ;
            i++ ;
            y [i] = x [i] ;
            i++ ;
            y [i] = x [i] ;
            i++ ;
            y [i] = x [i] ;
            i++ ;
        }
    }
    else
    {
        N = (CG_BLAS_INT) n ;
        CG_DCOPY (&N, x, blas_one, y, blas_one) ;
    }
#endif

    return ;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       End of routines that could use the BLAS
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

/* =========================================================================
   ==== cg_step ============================================================
   =========================================================================
   Compute xtemp = x + alpha d
   ========================================================================= */
static void cg_step
(
    CG_FLOAT *xtemp, /*output vector */
    CG_FLOAT     *x, /* initial vector */
    CG_FLOAT     *d, /* search direction */
    CG_FLOAT  alpha, /* stepsize */
    size_t        n  /* length of the vectors */
)
{
    size_t n5, i ;
    n5 = n % 5 ;
    if (alpha == -1)
    {
        for (i = 0; i < n5; i++) xtemp [i] = x[i] - d[i] ;
        for (; i < n; i += 5)
        {
            xtemp [i]   = x [i]   - d [i] ;
            xtemp [i+1] = x [i+1] - d [i+1] ;
            xtemp [i+2] = x [i+2] - d [i+2] ;
            xtemp [i+3] = x [i+3] - d [i+3] ;
            xtemp [i+4] = x [i+4] - d [i+4] ;
        }
    }
    else
    {
        for (i = 0; i < n5; i++) xtemp [i] = x[i] + alpha*d[i] ;
        for (; i < n; i += 5)
        {
            xtemp [i]   = x [i]   + alpha*d [i] ;
            xtemp [i+1] = x [i+1] + alpha*d [i+1] ;
            xtemp [i+2] = x [i+2] + alpha*d [i+2] ;
            xtemp [i+3] = x [i+3] + alpha*d [i+3] ;
            xtemp [i+4] = x [i+4] + alpha*d [i+4] ;
        }
    }
    return ;
}

/* =========================================================================
   ==== cg_init ============================================================
   =========================================================================
   initialize x to a given scalar value
   ========================================================================= */
static void cg_init
(
    CG_FLOAT *x, /* input and output vector */
    CG_FLOAT  s, /* scalar */
    size_t    n /* length of vector */
)
{
    size_t i, n5 ;
    n5 = n % 5 ;
    for (i = 0; i < n5; i++) x [i] = s ;
    for (; i < n;)
    {
        x [i] = s ;
        i++ ;
        x [i] = s ;
        i++ ;
        x [i] = s ;
        i++ ;
        x [i] = s ;
        i++ ;
        x [i] = s ;
        i++ ;
    }
    return ;
}

/* =========================================================================
   ==== cg_update_2 ========================================================
   =========================================================================
   Set gold = gnew (if not equal), compute 2-norm^2 of gnew, and optionally
      set d = -gnew
   ========================================================================= */
static CG_FLOAT cg_update_2
(
    CG_FLOAT *gold, /* old g */
    CG_FLOAT *gnew, /* new g */
    CG_FLOAT    *d, /* d */
    size_t       n /* length of vectors */
)
{
    size_t i, n5 ;
    CG_FLOAT s, t ;
    t = 0 ;
    n5 = n % 5 ;

    if ( d == NULL )
    {
        for (i = 0; i < n5; i++)
        {
            s = gnew [i] ;
            t += s*s ;
            gold [i] = s ;
        }
        for (; i < n; )
        {
            s = gnew [i] ;
            t += s*s ;
            gold [i] = s ;
            i++ ;

            s = gnew [i] ;
            t += s*s ;
            gold [i] = s ;
            i++ ;

            s = gnew [i] ;
            t += s*s ;
            gold [i] = s ;
            i++ ;

            s = gnew [i] ;
            t += s*s ;
            gold [i] = s ;
            i++ ;

            s = gnew [i] ;
            t += s*s ;
            gold [i] = s ;
            i++ ;
        }
    }
    else if ( gold != NULL )
    {
        for (i = 0; i < n5; i++)
        {
            s = gnew [i] ;
            t += s*s ;
            gold [i] = s ;
            d [i] = -s ;
        }
        for (; i < n; )
        {
            s = gnew [i] ;
            t += s*s ;
            gold [i] = s ;
            d [i] = -s ;
            i++ ;

            s = gnew [i] ;
            t += s*s ;
            gold [i] = s ;
            d [i] = -s ;
            i++ ;

            s = gnew [i] ;
            t += s*s ;
            gold [i] = s ;
            d [i] = -s ;
            i++ ;

            s = gnew [i] ;
            t += s*s ;
            gold [i] = s ;
            d [i] = -s ;
            i++ ;

            s = gnew [i] ;
            t += s*s ;
            gold [i] = s ;
            d [i] = -s ;
            i++ ;
        }
    }
    else
    {
        for (i = 0; i < n5; i++)
        {
            s = gnew [i] ;
            t += s*s ;
            d [i] = -s ;
        }
        for (; i < n; )
        {
            s = gnew [i] ;
            t += s*s ;
            d [i] = -s ;
            i++ ;

            s = gnew [i] ;
            t += s*s ;
            d [i] = -s ;
            i++ ;

            s = gnew [i] ;
            t += s*s ;
            d [i] = -s ;
            i++ ;

            s = gnew [i] ;
            t += s*s ;
            d [i] = -s ;
            i++ ;

            s = gnew [i] ;
            t += s*s ;
            d [i] = -s ;
            i++ ;
        }
    }
    return (t) ;
}

/* =========================================================================
   ==== cg_update_inf ======================================================
   =========================================================================
   Set gold = gnew, compute inf-norm of gnew, and optionally set d = -gnew
   ========================================================================= */
static CG_FLOAT cg_update_inf
(
    CG_FLOAT *gold, /* old g */
    CG_FLOAT *gnew, /* new g */
    CG_FLOAT    *d, /* d */
    size_t       n /* length of vectors */
)
{
    size_t i, n5 ;
    CG_FLOAT s, t ;
    t = 0 ;
    n5 = n % 5 ;

    if ( d == NULL )
    {
        for (i = 0; i < n5; i++)
        {
            s = gnew [i] ;
            gold [i] = s ;
            if ( t < fabs (s) ) t = fabs (s) ;
        }
        for (; i < n; )
        {
            s = gnew [i] ;
            gold [i] = s ;
            if ( t < fabs (s) ) t = fabs (s) ;
            i++ ;

            s = gnew [i] ;
            gold [i] = s ;
            if ( t < fabs (s) ) t = fabs (s) ;
            i++ ;

            s = gnew [i] ;
            gold [i] = s ;
            if ( t < fabs (s) ) t = fabs (s) ;
            i++ ;

            s = gnew [i] ;
            gold [i] = s ;
            if ( t < fabs (s) ) t = fabs (s) ;
            i++ ;

            s = gnew [i] ;
            gold [i] = s ;
            if ( t < fabs (s) ) t = fabs (s) ;
            i++ ;
        }
    }
    else
    {
        for (i = 0; i < n5; i++)
        {
            s = gnew [i] ;
            gold [i] = s ;
            d [i] = -s ;
            if ( t < fabs (s) ) t = fabs (s) ;
        }
        for (; i < n; )
        {
            s = gnew [i] ;
            gold [i] = s ;
            d [i] = -s ;
            if ( t < fabs (s) ) t = fabs (s) ;
            i++ ;

            s = gnew [i] ;
            gold [i] = s ;
            d [i] = -s ;
            if ( t < fabs (s) ) t = fabs (s) ;
            i++ ;

            s = gnew [i] ;
            gold [i] = s ;
            d [i] = -s ;
            if ( t < fabs (s) ) t = fabs (s) ;
            i++ ;

            s = gnew [i] ;
            gold [i] = s ;
            d [i] = -s ;
            if ( t < fabs (s) ) t = fabs (s) ;
            i++ ;

            s = gnew [i] ;
            gold [i] = s ;
            d [i] = -s ;
            if ( t < fabs (s) ) t = fabs (s) ;
            i++ ;
        }
    }
    return (t) ;
}
/* =========================================================================
   ==== cg_update_ykyk =====================================================
   =========================================================================
   Set gold = gnew, compute inf-norm of gnew
                            ykyk = 2-norm(gnew-gold)^2
                            ykgk = (gnew-gold) dot gnew
   ========================================================================= */
static CG_FLOAT cg_update_ykyk
(
    CG_FLOAT *gold, /* old g */
    CG_FLOAT *gnew, /* new g */
    CG_FLOAT *Ykyk,
    CG_FLOAT *Ykgk,
    size_t       n /* length of vectors */
)
{
    size_t i, n5 ;
    CG_FLOAT t, gnorm, yk, ykyk, ykgk ;
    gnorm = 0 ;
    ykyk = 0 ;
    ykgk = 0 ;
    n5 = n % 5 ;

    for (i = 0; i < n5; i++)
    {
        t = gnew [i] ;
        if ( gnorm < fabs (t) ) gnorm = fabs (t) ;
        yk = t - gold [i] ;
        gold [i] = t ;
        ykgk += yk*t ;
        ykyk += yk*yk ;
    }
    for (; i < n; )
    {
        t = gnew [i] ;
        if ( gnorm < fabs (t) ) gnorm = fabs (t) ;
        yk = t - gold [i] ;
        gold [i] = t ;
        ykgk += yk*t ;
        ykyk += yk*yk ;
        i++ ;

        t = gnew [i] ;
        if ( gnorm < fabs (t) ) gnorm = fabs (t) ;
        yk = t - gold [i] ;
        gold [i] = t ;
        ykgk += yk*t ;
        ykyk += yk*yk ;
        i++ ;

        t = gnew [i] ;
        if ( gnorm < fabs (t) ) gnorm = fabs (t) ;
        yk = t - gold [i] ;
        gold [i] = t ;
        ykgk += yk*t ;
        ykyk += yk*yk ;
        i++ ;

        t = gnew [i] ;
        if ( gnorm < fabs (t) ) gnorm = fabs (t) ;
        yk = t - gold [i] ;
        gold [i] = t ;
        ykgk += yk*t ;
        ykyk += yk*yk ;
        i++ ;

        t = gnew [i] ;
        if ( gnorm < fabs (t) ) gnorm = fabs (t) ;
        yk = t - gold [i] ;
        gold [i] = t ;
        ykgk += yk*t ;
        ykyk += yk*yk ;
        i++ ;
    }
    *Ykyk = ykyk ;
    *Ykgk = ykgk ;
    return (gnorm) ;
}

/* =========================================================================
   ==== cg_update_inf2 =====================================================
   =========================================================================
   Set gold = gnew, compute inf-norm of gnew & 2-norm of gnew, set d = -gnew
   ========================================================================= */
static CG_FLOAT cg_update_inf2
(
    CG_FLOAT   *gold, /* old g */
    CG_FLOAT   *gnew, /* new g */
    CG_FLOAT      *d, /* d */
    CG_FLOAT *gnorm2, /* 2-norm of g */
    size_t         n /* length of vectors */
)
{
    size_t i, n5 ;
    CG_FLOAT gnorm, s, t ;
    gnorm = 0 ;
    s = 0 ;
    n5 = n % 5 ;

    for (i = 0; i < n5; i++)
    {
        t = gnew [i] ;
        if ( gnorm < fabs (t) ) gnorm = fabs (t) ;
        s += t*t ;
        gold [i] = t ;
        d [i] = -t ;
    }
    for (; i < n; )
    {
        t = gnew [i] ;
        if ( gnorm < fabs (t) ) gnorm = fabs (t) ;
        s += t*t ;
        gold [i] = t ;
        d [i] = -t ;
        i++ ;

        t = gnew [i] ;
        if ( gnorm < fabs (t) ) gnorm = fabs (t) ;
        s += t*t ;
        gold [i] = t ;
        d [i] = -t ;
        i++ ;

        t = gnew [i] ;
        if ( gnorm < fabs (t) ) gnorm = fabs (t) ;
        s += t*t ;
        gold [i] = t ;
        d [i] = -t ;
        i++ ;

        t = gnew [i] ;
        if ( gnorm < fabs (t) ) gnorm = fabs (t) ;
        s += t*t ;
        gold [i] = t ;
        d [i] = -t ;
        i++ ;

        t = gnew [i] ;
        if ( gnorm < fabs (t) ) gnorm = fabs (t) ;
        s += t*t ;
        gold [i] = t ;
        d [i] = -t ;
        i++ ;
    }
    *gnorm2 = s ;
    return (gnorm) ;
}

/* =========================================================================
   ==== cg_update_d ========================================================
   =========================================================================
   Set d = -g + beta*d, compute 2-norm of d, and optionally the 2-norm of g
   ========================================================================= */
static CG_FLOAT cg_update_d
(
    CG_FLOAT      *d,
    CG_FLOAT      *g,
    CG_FLOAT    beta,
    CG_FLOAT *gnorm2, /* 2-norm of g */
    size_t         n /* length of vectors */
)
{
    size_t i, n5 ;
    CG_FLOAT dnorm2, s, t ;
    s = 0 ;
    dnorm2 = 0 ;
    n5 = n % 5 ;
    if ( gnorm2 == NULL )
    {
        for (i = 0; i < n5; i++)
        {
            t = g [i] ;
            t = -t + beta*d [i] ;
            d [i] = t ;
            dnorm2 += t*t ;
        }
        for (; i < n; )
        {
            t = g [i] ;
            t = -t + beta*d [i] ;
            d [i] = t ;
            dnorm2 += t*t ;
            i++ ;

            t = g [i] ;
            t = -t + beta*d [i] ;
            d [i] = t ;
            dnorm2 += t*t ;
            i++ ;

            t = g [i] ;
            t = -t + beta*d [i] ;
            d [i] = t ;
            dnorm2 += t*t ;
            i++ ;

            t = g [i] ;
            t = -t + beta*d [i] ;
            d [i] = t ;
            dnorm2 += t*t ;
            i++ ;

            t = g [i] ;
            t = -t + beta*d [i] ;
            d [i] = t ;
            dnorm2 += t*t ;
            i++ ;
        }
    }
    else
    {
        s = 0 ;
        for (i = 0; i < n5; i++)
        {
            t = g [i] ;
            s += t*t ;
            t = -t + beta*d [i] ;
            d [i] = t ;
            dnorm2 += t*t ;
        }
        for (; i < n; )
        {
            t = g [i] ;
            s += t*t ;
            t = -t + beta*d [i] ;
            d [i] = t ;
            dnorm2 += t*t ;
            i++ ;

            t = g [i] ;
            s += t*t ;
            t = -t + beta*d [i] ;
            d [i] = t ;
            dnorm2 += t*t ;
            i++ ;

            t = g [i] ;
            s += t*t ;
            t = -t + beta*d [i] ;
            d [i] = t ;
            dnorm2 += t*t ;
            i++ ;

            t = g [i] ;
            s += t*t ;
            t = -t + beta*d [i] ;
            d [i] = t ;
            dnorm2 += t*t ;
            i++ ;

            t = g [i] ;
            s += t*t ;
            t = -t + beta*d [i] ;
            d [i] = t ;
            dnorm2 += t*t ;
            i++ ;
        }
        *gnorm2 = s ;
    }

    return (dnorm2) ;
}

/* =========================================================================
   ==== cg_Yk ==============================================================
   =========================================================================
   Compute y = gnew - gold, set gold = gnew, compute y'y
   ========================================================================= */
static void cg_Yk
(
    CG_FLOAT    *y, /*output vector */
    CG_FLOAT *gold, /* initial vector */
    CG_FLOAT *gnew, /* search direction */
    CG_FLOAT  *yty, /* y'y */
    size_t       n  /* length of the vectors */
)
{
    size_t n5, i ;
    CG_FLOAT s, t ;
    n5 = n % 5 ;
    if ( (y != NULL) && (yty == NULL) )
    {
        for (i = 0; i < n5; i++)
        {
            y [i] = gnew [i] - gold [i] ;
            gold [i] = gnew [i] ;
        }
        for (; i < n; )
        {
            y [i] = gnew [i] - gold [i] ;
            gold [i] = gnew [i] ;
            i++ ;
    
            y [i] = gnew [i] - gold [i] ;
            gold [i] = gnew [i] ;
            i++ ;
    
            y [i] = gnew [i] - gold [i] ;
            gold [i] = gnew [i] ;
            i++ ;
    
            y [i] = gnew [i] - gold [i] ;
            gold [i] = gnew [i] ;
            i++ ;
    
            y [i] = gnew [i] - gold [i] ;
            gold [i] = gnew [i] ;
            i++ ;
        }
    }
    else if ( (y == NULL) && (yty != NULL) )
    {
        s = 0 ;
        for (i = 0; i < n5; i++)
        {
            t = gnew [i] - gold [i] ;
            gold [i] = gnew [i] ;
            s += t*t ;
        }
        for (; i < n; )
        {
            t = gnew [i] - gold [i] ;
            gold [i] = gnew [i] ;
            s += t*t ;
            i++ ;
    
            t = gnew [i] - gold [i] ;
            gold [i] = gnew [i] ;
            s += t*t ;
            i++ ;
    
            t = gnew [i] - gold [i] ;
            gold [i] = gnew [i] ;
            s += t*t ;
            i++ ;
    
            t = gnew [i] - gold [i] ;
            gold [i] = gnew [i] ;
            s += t*t ;
            i++ ;
    
            t = gnew [i] - gold [i] ;
            gold [i] = gnew [i] ;
            s += t*t ;
            i++ ;
        }
        *yty = s ;
    }
    else
    {
        s = 0 ;
        for (i = 0; i < n5; i++)
        {
            t = gnew [i] - gold [i] ;
            gold [i] = gnew [i] ;
            y [i] = t ;
            s += t*t ;
        }
        for (; i < n; )
        {
            t = gnew [i] - gold [i] ;
            gold [i] = gnew [i] ;
            y [i] = t ;
            s += t*t ;
            i++ ;
    
            t = gnew [i] - gold [i] ;
            gold [i] = gnew [i] ;
            y [i] = t ;
            s += t*t ;
            i++ ;
    
            t = gnew [i] - gold [i] ;
            gold [i] = gnew [i] ;
            y [i] = t ;
            s += t*t ;
            i++ ;
    
            t = gnew [i] - gold [i] ;
            gold [i] = gnew [i] ;
            y [i] = t ;
            s += t*t ;
            i++ ;
    
            t = gnew [i] - gold [i] ;
            gold [i] = gnew [i] ;
            y [i] = t ;
            s += t*t ;
            i++ ;
        }
        *yty = s ;
    }

    return ;
}

/* =========================================================================
   === cg_default ==========================================================
   =========================================================================
   Set default conjugate gradient parameter values. If the parameter argument
   of cg_descent is NULL, this routine is called by cg_descent automatically.
   If the user wishes to set parameter values, then the cg_parameter structure
   should be allocated in the main program. The user could call cg_default
   to initialize the structure, and then individual elements in the structure
   could be changed, before passing the structure to cg_descent.
   =========================================================================*/
void cg_default
(
    cg_parameter   *Parm
)
{
    /* T => print final function value
       F => no printout of final function value */
    Parm->PrintFinal = CG_FALSE ;

    /* Level 0 = no printing, ... , Level 3 = maximum printing */
    Parm->PrintLevel = 0 ;

    /* T => print parameters values
       F => do not display parameter values */
    Parm->PrintParms = CG_FALSE ;

    /* T => use LBFGS
       F => only use L-BFGS when memory >= n */
    Parm->LBFGS = CG_FALSE ;

    /* number of vectors stored in memory (code breaks in the Yk update if
       memory = 1 or 2) */
    Parm->memory = 11 ;

    /* SubCheck and SubSkip control the frequency with which the subspace
       condition is checked. It it checked for SubCheck*mem iterations and
       if it is not activated, then it is skipped for Subskip*mem iterations
       and Subskip is doubled. Whenever the subspace condition is satisfied,
       SubSkip is returned to its original value. */
    Parm->SubCheck = 8 ;
    Parm->SubSkip = 4 ;

    /* when relative distance from current gradient to subspace <= eta0,
       enter subspace if subspace dimension = mem (eta0 = 0 means gradient
       inside subspace) */
    Parm->eta0 = 0.001 ; /* corresponds to eta0*eta0 in the paper */

    /* when relative distance from current gradient to subspace >= eta1,
       leave subspace (eta1 = 1 means gradient orthogonal to subspace) */
    Parm->eta1 = 0.900 ; /* corresponds to eta1*eta1 in the paper */

    /* when relative distance from current gradient to subspace <= eta2,
       always enter subspace (invariant space) */
    Parm->eta2 = 1.e-10 ;

    /* T => use approximate Wolfe line search
       F => use ordinary Wolfe line search, switch to approximate Wolfe when
                |f_k+1-f_k| < AWolfeFac*C_k, C_k = average size of cost */
    Parm->AWolfe = CG_FALSE ;
    Parm->AWolfeFac = 1.e-3 ;

    /* factor in [0, 1] used to compute average cost magnitude C_k as follows:
       Q_k = 1 + (Qdecay)Q_k-1, Q_0 = 0,  C_k = C_k-1 + (|f_k| - C_k-1)/Q_k */
    Parm->Qdecay = .7 ;

    /* terminate after 2*n + nslow iterations without strict improvement in
       either function value or gradient */
    Parm->nslow = 1000 ;

    /* Stop Rules:
       T => ||grad||_infty <= max(grad_tol, initial |grad|_infty*StopFact)
       F => ||grad||_infty <= grad_tol*(1 + |f_k|) */
    Parm->StopRule = CG_TRUE ;
    Parm->StopFac = 0.e-12 ;

    /* T => estimated error in function value is eps*Ck,
       F => estimated error in function value is eps */
    Parm->PertRule = CG_TRUE ;
    Parm->eps = 1.e-6 ;

    /* factor by which eps grows when line search fails during contraction */
    Parm->egrow = 10. ;

    /* T => attempt quadratic interpolation in line search when
                |f_k+1 - f_k|/|f_k| > QuadCutOff
       F => no quadratic interpolation step */
    Parm->QuadStep = CG_TRUE ;
    Parm->QuadCutOff = 1.e-12 ;

    /* maximum factor by which a quad step can reduce the step size */
    Parm->QuadSafe = 1.e-10 ;

    /* T => when possible, use a cubic step in the line search */
    Parm->UseCubic = CG_TRUE ;

    /* use cubic step when |f_k+1 - f_k|/|f_k| > CubicCutOff */
    Parm->CubicCutOff = 1.e-12 ;

    /* |f| < SmallCost*starting cost => skip QuadStep and set PertRule = 0*/
    Parm->SmallCost = 1.e-30 ;

    /* T => check that f_k+1 - f_k <= debugtol*C_k
       F => no checking of function values */
    Parm->debug = 0 ;
    Parm->debugtol = 1.e-10 ;

    /* if step is nonzero, it is the initial step of the initial line search */
    Parm->step = 0 ;

    /* abort cg after maxit iterations */
    Parm->maxit = SIZE_MAX ;

    /* maximum number of times the bracketing interval grows during expansion */
    Parm->ntries = 50 ;

    /* maximum factor secant step increases stepsize in expansion phase */
    Parm->ExpandSafe = 200. ;

    /* factor by which secant step is amplified during expansion phase
       where minimizer is bracketed */
    Parm->SecantAmp = 1.05 ;

    /* factor by which rho grows during expansion phase where minimizer is
       bracketed */
    Parm->RhoGrow = 2.0 ;

    /* maximum number of times that eps is updated */
    Parm->neps = 5 ;

    /* maximum number of times the bracketing interval shrinks */
    Parm->nshrink = 10 ;

    /* maximum number of secant iterations in line search is nline */
    Parm->nline = 50 ;

    /* conjugate gradient method restarts after (n*restart_fac) iterations */
    Parm->restart_fac = 6.0 ;

    /* stop when -alpha*dphi0 (estimated change in function value) <= feps*|f|*/
    Parm->feps = 0 ;

    /* after encountering nan, growth factor when searching for
       a bracketing interval */
    Parm->nan_rho = 1.3 ;

    /* after encountering nan, decay factor for stepsize */
    Parm->nan_decay = 0.1 ;

    /* Wolfe line search parameter, range [0, .5]
       phi (a) - phi(0) <= delta phi'(0) */
    Parm->delta = .1 ;

    /* Wolfe line search parameter, range [delta, 1]
       phi' (a) >= sigma phi'(0) */
    Parm->sigma = .9 ;

    /* decay factor for bracket interval width in line search, range (0, 1) */
    Parm->gamma = .66 ;

    /* growth factor in search for initial bracket interval */
    Parm->rho = 5. ;

    /* starting guess for line search =
         psi0 ||x_0||_infty over ||g_0||_infty if x_0 != 0
         psi0 |f(x_0)|/||g_0||_2               otherwise */
    Parm->psi0 = .01 ;      /* factor used in starting guess for iteration 1 */

    /* for a QuadStep, function evaluated on interval
       [psi_lo, phi_hi]*psi2*previous step */
    Parm->psi_lo = 0.1 ;
    Parm->psi_hi = 10. ;

    /* when the function is approximately quadratic, use gradient at
       psi1*psi2*previous step for estimating initial stepsize */
    Parm->psi1 = 1.0 ;

    /* when starting a new cg iteration, our initial guess for the line
       search stepsize is psi2*previous step */
    Parm->psi2 = 2. ;

    /* choose theta adaptively if AdaptiveBeta = T */
    Parm->AdaptiveBeta = 0 ;

    /* lower bound for beta is BetaLower*d_k'g_k/ ||d_k||^2 */
    Parm->BetaLower = 0.4 ;

    /* value of the parameter theta in the cg_descent update formula:
       W. W. Hager and H. Zhang, A survey of nonlinear conjugate gradient
       methods, Pacific Journal of Optimization, 2 (2006), pp. 35-58. */
    Parm->theta = 1.0 ;

    /* parameter used in cost error estimate for quadratic restart criterion */
    Parm->qeps = 1.e-12 ;

    /* number of iterations the function is nearly quadratic before a restart */
    Parm->qrestart = 6 ;

    /* treat cost as quadratic if
       |1 - (cost change)/(quadratic cost change)| <= qrule */
    Parm->qrule = 1.e-8 ;
}

/* =========================================================================
   ==== cg_printParms ======================================================
   =========================================================================
   Print the contents of the cg_parameter structure
   ========================================================================= */
static void cg_printParms
(
    cg_parameter  *Parm
)
{
    printf ("PARAMETERS:\n") ;
    printf ("\n") ;
    printf ("Wolfe line search parameter ..................... delta: %e\n",
             Parm->delta) ;
    printf ("Wolfe line search parameter ..................... sigma: %e\n",
             Parm->sigma) ;
    printf ("decay factor for bracketing interval ............ gamma: %e\n",
             Parm->gamma) ;
    printf ("growth factor for bracket interval ................ rho: %e\n",
             Parm->rho) ;
    printf ("growth factor for bracket interval after nan .. nan_rho: %e\n",
             Parm->nan_rho) ;
    printf ("decay factor for stepsize after nan ......... nan_decay: %e\n",
             Parm->nan_decay) ;
    printf ("parameter in lower bound for beta ........... BetaLower: %e\n",
             Parm->BetaLower) ;
    printf ("parameter describing cg_descent family .......... theta: %e\n",
             Parm->theta) ;
    printf ("perturbation parameter for function value ......... eps: %e\n",
             Parm->eps) ;
    printf ("factor by which eps grows if necessary .......... egrow: %e\n",
             Parm->egrow) ;
    printf ("factor for computing average cost .............. Qdecay: %e\n",
             Parm->Qdecay) ;
    printf ("relative change in cost to stop quadstep ... QuadCutOff: %e\n",
             Parm->QuadCutOff) ;
    printf ("maximum factor quadstep reduces stepsize ..... QuadSafe: %e\n",
             Parm->QuadSafe) ;
    printf ("skip quadstep if |f| <= SmallCost*start cost  SmallCost: %e\n",
             Parm->SmallCost) ;
    printf ("relative change in cost to stop cubic step  CubicCutOff: %e\n",
             Parm->CubicCutOff) ;
    printf ("terminate if no improvement over nslow iter ..... nslow: %zu\n",
             Parm->nslow) ;
    printf ("factor multiplying gradient in stop condition . StopFac: %e\n",
             Parm->StopFac) ;
    printf ("cost change factor, approx Wolfe transition . AWolfeFac: %e\n",
             Parm->AWolfeFac) ;
    printf ("restart cg every restart_fac*n iterations . restart_fac: %e\n",
             Parm->restart_fac) ;
    printf ("cost error in quadratic restart is qeps*cost ..... qeps: %e\n",
             Parm->qeps) ;
    printf ("number of quadratic iterations before restart  qrestart: %zu\n",
             Parm->qrestart) ;
    printf ("parameter used to decide if cost is quadratic ... qrule: %e\n",
             Parm->qrule) ;
    printf ("stop when cost change <= feps*|f| ................ feps: %e\n",
             Parm->feps) ;
    printf ("starting guess parameter in first iteration ...... psi0: %e\n",
             Parm->psi0) ;
    printf ("starting step in first iteration if nonzero ...... step: %e\n",
             Parm->step) ;
    printf ("lower bound factor in quad step ................ psi_lo: %e\n",
             Parm->psi_lo) ;
    printf ("upper bound factor in quad step ................ psi_hi: %e\n",
             Parm->psi_hi) ;
    printf ("initial guess factor for quadratic functions ..... psi1: %e\n",
             Parm->psi1) ;
    printf ("initial guess factor for general iteration ....... psi2: %e\n",
             Parm->psi2) ;
    printf ("max iterations .................................. maxit: %zu\n",
             Parm->maxit) ;
    printf ("max number of contracts in the line search .... nshrink: %zu\n",
             Parm->nshrink) ;
    printf ("max expansions in line search .................. ntries: %zu\n",
             Parm->ntries) ;
    printf ("maximum growth of secant step in expansion . ExpandSafe: %e\n",
             Parm->ExpandSafe) ;
    printf ("growth factor for secant step during expand . SecantAmp: %e\n",
             Parm->SecantAmp) ;
    printf ("growth factor for rho during expansion phase .. RhoGrow: %e\n",
             Parm->RhoGrow) ;
    printf ("distance threshhold for entering subspace ........ eta0: %e\n",
             Parm->eta0) ;
    printf ("distance threshhold for leaving subspace ......... eta1: %e\n",
             Parm->eta1) ;
    printf ("distance threshhold for invariant space .......... eta2: %e\n",
             Parm->eta2) ;
    printf ("number of vectors stored in memory ............. memory: %zu\n",
             Parm->memory) ;
    printf ("check subspace condition mem*SubCheck its .... SubCheck: %zu\n",
             Parm->SubCheck) ;
    printf ("skip subspace checking for mem*SubSkip its .... SubSkip: %zu\n",
             Parm->SubSkip) ;
    printf ("max number of times that eps is updated .......... neps: %zu\n",
             Parm->neps) ;
    printf ("max number of iterations in line search ......... nline: %zu\n",
             Parm->nline) ;
    printf ("print level (0 = none, 3 = maximum) ........ PrintLevel: %zu\n",
             Parm->PrintLevel) ;
    printf ("Logical parameters:\n") ;
    if ( Parm->PertRule )
        printf ("    Error estimate for function value is eps*Ck\n") ;
    else
        printf ("    Error estimate for function value is eps\n") ;
    if ( Parm->QuadStep )
        printf ("    Use quadratic interpolation step\n") ;
    else
        printf ("    No quadratic interpolation step\n") ;
    if ( Parm->UseCubic)
        printf ("    Use cubic interpolation step when possible\n") ;
    else
        printf ("    Avoid cubic interpolation steps\n") ;
    if ( Parm->AdaptiveBeta )
        printf ("    Adaptively adjust direction update parameter beta\n") ;
    else
        printf ("    Use fixed parameter theta in direction update\n") ;
    if ( Parm->PrintFinal )
        printf ("    Print final cost and statistics\n") ;
    else
        printf ("    Do not print final cost and statistics\n") ;
    if ( Parm->PrintParms )
        printf ("    Print the parameter structure\n") ;
    else
        printf ("    Do not print parameter structure\n") ;
    if ( Parm->AWolfe)
        printf ("    Approximate Wolfe line search\n") ;
    else {
        printf ("    Wolfe line search") ;
        if ( Parm->AWolfeFac > 0. )
            printf (" ... switching to approximate Wolfe\n") ;
        else
            printf ("\n") ;
		}
    if ( Parm->StopRule )
        printf ("    Stopping condition uses initial grad tolerance\n") ;
    else
        printf ("    Stopping condition weighted by absolute cost\n") ;
    if ( Parm->debug)
        printf ("    Check for decay of cost, debugger is on\n") ;
    else
        printf ("    Do not check for decay of cost, debugger is off\n") ;
}

/*
Version 1.2 Change:
  1. The variable dpsi needs to be included in the argument list for
     subroutine cg_updateW (update of a Wolfe line search)

Version 2.0 Changes:
     The user interface was redesigned. The parameters no longer need to
     be read from a file. For compatibility with earlier versions of the
     code, we include the routine cg_readParms to read parameters.
     In the simplest case, the user can use NULL for the
     parameter argument of cg_descent, and the code sets the default
     parameter values. If the user wishes to modify the parameters, call
     cg_default in the main program to initialize a cg_parameter
     structure. Individual elements of the structure could be modified.
     The header file cg_user.h contains the structures and prototypes
     that the user may need to reference or modify, while cg_descent.h
     contains header elements that only cg_descent will access.  Note
     that the arguments of cg_descent have changed.

Version 3.0 Changes:
     Major overhaul

Version 4.0 Changes:
  Modifications 1-3 were made based on results obtained by Yu-Hong Dai and
  Cai-Xia Kou in the paper "A nonlinear conjugate gradient algorithm with an
     optimal property and an improved Wolfe line search"
  1. Set theta = 1.0 by default in the cg_descent rule for beta_k (Dai and
     Kou showed both theoretical and practical advantages in this choice).
  2. Increase the default value of restart_fac to 6 (a value larger than 1 is
     more efficient when the problem dimension is small)
  3. Restart the CG iteration if the objective function is nearly quadratic
     for several iterations (qrestart).
  4. New lower bound for beta: BetaLower*d_k'g_k/ ||d_k||^2. This lower
     bound guarantees convergence and it seems to provide better
     performance than the original lower bound for beta in cg_descent;
     it also seems to give slightly better performance than the
     lower bound BetaLower*d_k'g_k+1/ ||d_k||^2 suggested by Dai and Kou.
  5. Evaluation of the objective function and gradient is now handled by
     the routine cg_evaluate.

Version 4.1 Changes:
  1. Change cg_tol to be consistent with corresponding routine in asa_cg
  2. Compute dnorm2 when d is evaluated and make loops consistent with asa_cg

Version 4.2 Changes:
  1. Modify the line search so that when there are too many contractions,
     the code will increase eps and switch to expansion of the search interval.
     This fixes some cases where the code terminates when eps is too small.
     When the estimated error in the cost function is too small, the algorithm
     could fail in cases where the slope is negative at both ends of the
     search interval and the objective function value on the right side of the
     interval is larger than the value at the left side (because the true
     objective function value on the right is not greater than the value on
     the left).
  2. Fix bug in cg_lineW

Version 5.0 Changes:
     Revise the line search routines to exploit steps based on the
     minimizer of a Hermite interpolating cubic. Combine the approximate
     and the ordinary Wolfe line search into a single routine.
     Include safeguarded extrapolation during the expansion phase
     of the line search. Employ a quadratic interpolation step even
     when the requirement ftemp < f for a quadstep is not satisfied.

Version 5.1 Changes:
  1. Shintaro Kaneko pointed out spelling error in line 738, change
     "stict" to "strict"
  2. Add MATLAB interface

Version 5.2 Changes:
  1. Make QuadOK always 1 in the quadratic interpolation step.
  2. Change QuadSafe to 1.e-10 instead of 1.e-3 (less safe guarding).
  3. Change psi0 to 2 in the routine for computing the stepsize
     in the first iteration when x = 0.
  4. In the quadratic step routine, the stepsize at the trial point
     is the maximum of {psi_lo*psi2, prior df/(current df)}*previous step
  5. In quadratic step routine, the function is evaluated on the safe-guarded
     interval [psi_lo, phi_hi]*psi2*previous step.
  6. Allow more rapid expansion in the line search routine.

Version 5.3 Changes:
  1. Make changes so that cg_descent works with version R2012a of MATLAB.
     This required allocating memory for the work array inside the MATLAB
     mex routine and rearranging memory so that the xtemp pointer and the
     work array pointer are the same.

Version 6.0 Changes:
  Major revision of the code to implement the limited memory conjugate
  gradient algorithm documented in reference [4] at the top of this file.
  The code has doubled in length. The line search remains the same except
  for small adjustments in the nan detection which allows it to
  solve more problems that generate nan's. The direction routine, however,
  is completely new. Version 5.3 of the code is obtained by setting
  the memory parameter to be 0. The L-BFGS algorithm is obtained by setting
  the LBFGS parameter to 1. Otherwise, for memory > 0 and LBFGS = 0,
  the search directions are obtained by a limited memory version of the
  original cg_descent algorithm. The memory is used to detect when the
  gradients lose orthogonality locally. When orthogonality is lost,
  the algorithm solves a subspace problem until orthogonality is restored.
  It is now possible to utilize the BLAS, if they are available, by
  commenting out a line in the file cg_blas.h. See the README file for
  the details.

Version 6.1 Changes:
  Fixed problems connected with memory handling in the MATLAB version
  of the code. These errors only arise when using some versions of MATLAB.
  Replaced "malloc" in the cg_descent mex routine with "mxMalloc".
  Thanks to Stephen Vavasis for reporting this error that occurred when
  using MATLAB version R2012a, 7.14.

Version 6.2 Change:
  When using cg_descent in MATLAB, the input starting guess is no longer
  overwritten by the final solution. This makes the cg_descent mex function
  compliant with MATLAB's convention for the treatment of input arguments.
  Thanks to Dan Scholnik, Naval Research Laboratory, for pointing out this
  inconsistency with MATLAB convention in earlier versions of cg_descent.

Version 6.3 Change:
  For problems of dimension <= Parm->memory (default 11), the final
  solution was not copied to the user's solution argument x. Instead x
  stored the the next-to-final iterate. This final copy is now inserted
  on line 969.  Thanks to Arnold Neumaier of the University of Vienna
  for pointing out this bug.

Version 6.4 Change:
  In order to prevent a segmentation fault connected with MATLAB's memory
  handling, inserted a copy statement inside cg_evaluate. This copy is
  only needed when solving certain problems with MATLAB.  Many thanks
  to Arnold Neumaier for pointing out this problem

Version 6.5 Change:
  When using the code in MATLAB, changed the x vector that is passed to
  the user's function and gradient routine to be column vectors instead
  of row vectors.

Version 6.6 Change:
  Expand the statistics structure to include the number of subspaces (NumSub)
  and the number of subspace iterations (IterSub).

Version 6.7 Change:
  Add interface to CUTEst

Version 6.8 Change:
  When the denominator of the variable "scale" vanishes, retain the
  previous value of scale. This correct an error pointed out by
  Zachary Blunden-Codd.
*/

#endif // CG_DESCENT_IMPLEMENTATION
