// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Jonathan Zimmerman (Sandia)
------------------------------------------------------------------------- */

#include "pair_lj_bump_smooth_linear.h"

#include <cmath>
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"
#include "math_const.h"

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairLJBumpSmoothLinear::PairLJBumpSmoothLinear(LAMMPS *lmp) : Pair(lmp) {
	single_hessian_enable = 1;
	centroidstressflag = 1;
  born_matrix_enable = 0;
}

/* ---------------------------------------------------------------------- */

PairLJBumpSmoothLinear::~PairLJBumpSmoothLinear()
{
  if (copymode) return;
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(ljcut);
    memory->destroy(dljcut);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);

   // forbump
    memory->destroy(start_bump);
    memory->destroy(end_bump);
    memory->destroy(energy_bump);

  }
}

/* ---------------------------------------------------------------------- */

void PairLJBumpSmoothLinear::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,r2inv,r6inv,forcelj,factor_lj;
  double r,rinv;
  int *ilist,*jlist,*numneigh,**firstneigh;

  // forbump
  double rtmp, btmp;
  //for smooth
//  double r, rinv;

  evdwl = 0.0;
  ev_init(eflag,vflag);
 // if (eflag || vflag) ev_setup(eflag,vflag);
 // else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        rtmp = sqrt(rsq);
        rinv = 1.0 / rtmp;
        r2inv = 1.0 / rsq;
        r6inv = r2inv * r2inv *  r2inv;
        forcelj = r6inv*(lj1[itype][jtype]*r6inv-lj2[itype][jtype]);
        forcelj = rinv*forcelj - dljcut[itype][jtype];

        fpair = factor_lj*forcelj*rinv;

//for bump
        if(rtmp >= start_bump[itype][jtype] && rtmp <= end_bump[itype][jtype]) {
          fpair += -energy_bump[itype][jtype]*MY_PI*sin(MY_PI*(end_bump[itype][jtype]+start_bump[itype][jtype]-rtmp-rtmp)/(end_bump[itype][jtype]-start_bump[itype][jtype]))/(end_bump[itype][jtype]-start_bump[itype][jtype])/rtmp;
        }

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (eflag) {
          evdwl = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]);
          evdwl = evdwl - ljcut[itype][jtype]
                          + (rtmp-cut[itype][jtype])*dljcut[itype][jtype];
           //bump
          if(rtmp >= start_bump[itype][jtype] && rtmp <= end_bump[itype][jtype]) {
            btmp = sin(MY_PI*(end_bump[itype][jtype]-rtmp)/(end_bump[itype][jtype]-start_bump[itype][jtype]));
            evdwl += energy_bump[itype][jtype]*btmp*btmp;
          }
	//binghui has the following	
//          evdwl *= factor_lj;

       }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,fpair,delx,dely,delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairLJBumpSmoothLinear::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(epsilon,n+1,n+1,"pair:epsilon");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(ljcut,n+1,n+1,"pair:ljcut");
  memory->create(dljcut,n+1,n+1,"pair:dljcut");
  memory->create(lj1,n+1,n+1,"pair:lj1");
  memory->create(lj2,n+1,n+1,"pair:lj2");
  memory->create(lj3,n+1,n+1,"pair:lj3");
  memory->create(lj4,n+1,n+1,"pair:lj4");

  // forbump
  memory->create(start_bump,n+1,n+1,"pair:startbump");
  memory->create(end_bump,n+1,n+1,"pair:endbump");
  memory->create(energy_bump,n+1,n+1,"pair:energybump");

}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairLJBumpSmoothLinear::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR,"Illegal pair_style command");

  cut_global = utils::numeric(FLERR,arg[0],false,lmp);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j])
          cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairLJBumpSmoothLinear::coeff(int narg, char **arg)
{
  //originally there should be 4 or 5 arguments
  // now should be 7 or 8 arguments, re_start, re_end, re_energy for bump
 if (narg != 7 && narg != 8)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);

  double epsilon_one = utils::numeric(FLERR,arg[2],false,lmp);
  double sigma_one = utils::numeric(FLERR,arg[3],false,lmp);

  //for bump
  double re_start = atof(arg[4]);
  double re_end = atof(arg[5]);
  double re_energy = atof(arg[6]);

  double cut_one = cut_global;
  if (narg == 8) {
    cut_one = utils::numeric(FLERR,arg[7],false,lmp);
  }

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      sigma[i][j] = sigma_one;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;

      //bump
      start_bump[i][j]  = re_start  * sigma[i][j];
      end_bump[i][j]    = re_end    * cut[i][j];
      energy_bump[i][j] = re_energy * epsilon[i][j];
      start_bump[j][i]  = start_bump[i][j];
      end_bump[j][i]    = end_bump[i][j];
      energy_bump[j][i] = energy_bump[i][j];
      printf("bump [%d %d] start,end,energy, %f %f %f\n", i, j, start_bump[i][j], end_bump[i][j], energy_bump[i][j]);

      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairLJBumpSmoothLinear::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    epsilon[i][j] = mix_energy(epsilon[i][i],epsilon[j][j],
                               sigma[i][i],sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i],sigma[j][j]);
    cut[i][j] = mix_distance(cut[i][i],cut[j][j]);
  }

  lj1[i][j] = 48.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj2[i][j] = 24.0 * epsilon[i][j] * pow(sigma[i][j],6.0);
  lj3[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj4[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],6.0);
  double cut6inv = pow(cut[i][j],-6.0);
  double cutinv  = 1.0/cut[i][j];
  ljcut[i][j]  = cut6inv*(lj3[i][j]*cut6inv-lj4[i][j]);
  dljcut[i][j] = cutinv*cut6inv*(lj1[i][j]*cut6inv-lj2[i][j]);

  cut[j][i] = cut[i][j];
  lj1[j][i] = lj1[i][j];
  lj2[j][i] = lj2[i][j];
  lj3[j][i] = lj3[i][j];
  lj4[j][i] = lj4[i][j];
  cut[j][i] = cut[i][j];
  ljcut[j][i] = ljcut[i][j];
  dljcut[j][i] = dljcut[i][j];

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJBumpSmoothLinear::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&epsilon[i][j],sizeof(double),1,fp);
        fwrite(&sigma[i][j],sizeof(double),1,fp);
        fwrite(&cut[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJBumpSmoothLinear::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR,&setflag[i][j],sizeof(int),1,fp,nullptr,error);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          utils::sfread(FLERR,&epsilon[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&sigma[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&cut[i][j],sizeof(double),1,fp,nullptr,error);
        }
        MPI_Bcast(&epsilon[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJBumpSmoothLinear::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJBumpSmoothLinear::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
    utils::sfread(FLERR,&cut_global,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&mix_flag,sizeof(int),1,fp,nullptr,error);
  }
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}

/* ---------------------------------------------------------------------- */

double PairLJBumpSmoothLinear::single(int /*i*/, int /*j*/, int itype, int jtype,
                                  double rsq,
                                  double /*factor_coul*/, double factor_lj,
                                  double &fforce)
{
  double r2inv,r6inv,forcelj,philj,r,rinv,rtmp,btmp;

  r2inv = 1.0/rsq;
  r6inv = r2inv*r2inv*r2inv;
  rinv  = sqrt(r2inv);
  r     = sqrt(rsq);
  forcelj = r6inv*(lj1[itype][jtype]*r6inv-lj2[itype][jtype]);
  forcelj = rinv*forcelj - dljcut[itype][jtype];
  fforce = factor_lj*forcelj*rinv;
  rtmp = sqrt(rsq);
        if(rtmp >= start_bump[itype][jtype] && rtmp <= end_bump[itype][jtype]) {
          fforce += -energy_bump[itype][jtype]*MY_PI*sin(MY_PI*(end_bump[itype][jtype]+start_bump[itype][jtype]-rtmp-rtmp)/(end_bump[itype][jtype]-start_bump[itype][jtype]))/(end_bump[itype][jtype]-start_bump[itype][jtype])/rtmp;
        }
  philj = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]);
  philj = philj - ljcut[itype][jtype]
                + (r-cut[itype][jtype])*dljcut[itype][jtype];

  //printf("pritom debug : %f %f %f\n",start_bump[itype][jtype],end_bump[itype][jtype],energy_bump[itype][jtype]);

         rtmp = sqrt(rsq);
            if(rtmp >= start_bump[itype][jtype] && rtmp <= end_bump[itype][jtype]) {
		                btmp = sin(MY_PI*(end_bump[itype][jtype]-rtmp)/(end_bump[itype][jtype]-start_bump[itype][jtype]));
				printf("pritom debug1 : %f %f %f %f %f\n",philj,btmp, start_bump[itype][jtype],end_bump[itype][jtype],energy_bump[itype][jtype]);            
				
				philj += energy_bump[itype][jtype]*btmp*btmp;          
				printf("pritom debug2 : %f %f %f %f %f\n",philj,btmp, start_bump[itype][jtype],end_bump[itype][jtype],energy_bump[itype][jtype]);}
  //double evdwl;
  //evdwl = 0.0;
    //      evdwl = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]);
//	            evdwl = evdwl - ljcut[itype][jtype]
//			                              + (r-cut[itype][jtype])*dljcut[itype][jtype];


           //bump
//	             if(rtmp >= start_bump[itype][jtype] && rtmp <= end_bump[itype][jtype]) {
//	                         btmp = sin(MY_PI*(end_bump[itype][jtype]-rtmp)/(end_bump[itype][jtype]-start_bump[itype][jtype]));
//	                                     evdwl += energy_bump[itype][jtype]*btmp*btmp;
//	                                               }


  return factor_lj*philj;
}

double PairLJBumpSmoothLinear::single_hessian(int /*i*/, int /*j*/, int itype, int jtype, double rsq,
                         double delr[3], double /*factor_coul*/, double factor_lj,
                         double &fforce, double d2u[6])
{
  double r2inv,r6inv,forcelj,philj,r,rinv,rtmp,btmp;

  r2inv = 1.0/rsq;
  r6inv = r2inv*r2inv*r2inv;
  rinv  = sqrt(r2inv);
  r     = sqrt(rsq);
  forcelj = r6inv*(lj1[itype][jtype]*r6inv-lj2[itype][jtype]);
												  
  forcelj = rinv*forcelj - dljcut[itype][jtype];
  fforce = factor_lj*forcelj*rinv;

  rtmp = sqrt(rsq);
        if(rtmp >= start_bump[itype][jtype] && rtmp <= end_bump[itype][jtype]) {
          fforce += -energy_bump[itype][jtype]*MY_PI*sin(MY_PI*(end_bump[itype][jtype]+start_bump[itype][jtype]-rtmp-rtmp)/(end_bump[itype][jtype]-start_bump[itype][jtype]))/(end_bump[itype][jtype]-start_bump[itype][jtype])/rtmp;
        }

  philj = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]);
  philj = philj - ljcut[itype][jtype]
                + (r-cut[itype][jtype])*dljcut[itype][jtype];
  
  
         rtmp = sqrt(rsq);
            if(rtmp >= start_bump[itype][jtype] && rtmp <= end_bump[itype][jtype]) {
		                btmp = sin(MY_PI*(end_bump[itype][jtype]-rtmp)/(end_bump[itype][jtype]-start_bump[itype][jtype]));
				            philj += energy_bump[itype][jtype]*btmp*btmp;          }
	
  
  double d2r = factor_lj * r6inv * (13.0*lj1[itype][jtype]*r6inv - 7.0*lj2[itype][jtype])/rsq;
  hessian_twobody(fforce, -(fforce + d2r) / rsq, delr, d2u);
																															  
																					 
												   


  return factor_lj*philj;
}


void PairLJBumpSmoothLinear::born_matrix(int /*i*/, int /*j*/, int itype, int jtype, double rsq,
                            double /*factor_coul*/, double factor_lj, double &dupair,
                            double &du2pair)
{
  double rinv, r2inv, r6inv, du, du2;

  r2inv = 1.0 / rsq;
  rinv = sqrt(r2inv);
  r6inv = r2inv * r2inv * r2inv;

  // Reminder: lj1 = 48*e*s^12, lj2 = 24*e*s^6
  // so dupair = -forcelj/r = -fforce*r (forcelj from single method)

  du = r6inv * rinv * (lj2[itype][jtype] - lj1[itype][jtype] * r6inv);
  du2 = r6inv * r2inv * (13 * lj1[itype][jtype] * r6inv - 7 * lj2[itype][jtype]);

  dupair = factor_lj * du;
  du2pair = factor_lj * du2;
}
