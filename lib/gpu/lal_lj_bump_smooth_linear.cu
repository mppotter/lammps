// **************************************************************************
//                           lj_bump_smooth_linear.cu
//                             -------------------
//                              Mark Potter (RPI)
//
//  Device code for acceleration of the lj/bump/smooth/linear pair style
//
// __________________________________________________________________________
//    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
// __________________________________________________________________________
//
//    begin                :
//    email                : pottem3@rpi.edu
// ***************************************************************************


#if defined(NV_KERNEL) || defined(USE_HIP)
#include "lal_aux_fun1.h"
#include <math.h>
#ifndef _DOUBLE_DOUBLE
_texture( pos_tex,float4);
#else
_texture_2d( pos_tex,int4);
#endif
#else
#define pos_tex x_
#endif

__kernel void k_lj_bump_smooth_linear(const __global numtyp4 *restrict x_,
                        const __global numtyp4 *restrict lj1,
                        const __global numtyp4 *restrict lj3,
                        const int lj_types,
                        const __global numtyp *restrict sp_lj,
                        const __global numtyp4 *restrict bump_data,
                        const __global numtyp4 *restrict smooth_linear_data,
                        const __global int * dev_nbor,
                        const __global int * dev_packed,
                        __global acctyp4 *restrict ans,
                        __global acctyp *restrict engv,
                        const int eflag, const int vflag, const int inum,
                        const int nbor_pitch, const int t_per_atom) {
  int tid, ii, offset;
  atom_info(t_per_atom,ii,tid,offset);

  int n_stride;
  local_allocate_store_pair();

  // for bump
  numtyp r, btmp;
  numtyp rinv;

  acctyp4 f;
  f.x=(acctyp)0; f.y=(acctyp)0; f.z=(acctyp)0;
  acctyp energy, virial[6];
  if (EVFLAG) {
    energy=(acctyp)0;
    for (int i=0; i<6; i++) virial[i]=(acctyp)0;
  }

  if (ii<inum) {
    int i, numj, nbor, nbor_end;
    nbor_info(dev_nbor,dev_packed,nbor_pitch,t_per_atom,ii,offset,i,numj,
              n_stride,nbor_end,nbor);

    numtyp4 ix; fetch4(ix,i,pos_tex); //x_[i];
    int itype=ix.w;

    numtyp factor_lj;
    for ( ; nbor<nbor_end; nbor+=n_stride) {

      int j=dev_packed[nbor];
      factor_lj = sp_lj[sbmask(j)];
      j &= NEIGHMASK;

      numtyp4 jx; fetch4(jx,j,pos_tex); //x_[j];
      int jtype=jx.w;

      // Compute r12
      numtyp delx = ix.x-jx.x;
      numtyp dely = ix.y-jx.y;
      numtyp delz = ix.z-jx.z;
      numtyp r2inv = delx*delx+dely*dely+delz*delz;

      int mtype=itype*lj_types+jtype;
      if (r2inv<lj1[mtype].z) {
        r = ucl_sqrt(r2inv);
        r2inv = ucl_recip(r2inv); // This makes r2inv the actual r^2 inverse
        rinv = ucl_recip(r);
        numtyp r6inv = r2inv*r2inv*r2inv;
        numtyp force = r6inv*(lj1[mtype].x*r6inv-lj1[mtype].y);
        force = rinv*(rinv*force - smooth_linear_data[mtype].y);
        force*=factor_lj;

        // for bump
        if(r >= bump_data[mtype].x && r <= bump_data[mtype].y) {
            force += -bump_data[mtype].z*M_PI*sinpi((bump_data[mtype].y+bump_data[mtype].x-r-r)/(bump_data[mtype].y-bump_data[mtype].x))/(bump_data[mtype].y-bump_data[mtype].x)/r;
        }

        f.x+=delx*force;
        f.y+=dely*force;
        f.z+=delz*force;

        if (EVFLAG && eflag) {
          numtyp e=r6inv*(lj3[mtype].x*r6inv-lj3[mtype].y);
          e-=smooth_linear_data[mtype].x
             -(r-smooth_linear_data[mtype].z)*smooth_linear_data[mtype].y;
          //bump
          if(r >= bump_data[mtype].x && r <= bump_data[mtype].y) {
            btmp = sinpi((bump_data[mtype].y-r)/(bump_data[mtype].y-bump_data[mtype].x));
            e += bump_data[mtype].z*btmp*btmp;
          }
          energy+=e; //factor_lj*(e-lj3[mtype].z);
        }
        if (EVFLAG && vflag) {
          virial[0] += delx*delx*force;
          virial[1] += dely*dely*force;
          virial[2] += delz*delz*force;
          virial[3] += delx*dely*force;
          virial[4] += delx*delz*force;
          virial[5] += dely*delz*force;
        }
      }

    } // for nbor
  } // if ii
  store_answers(f,energy,virial,ii,inum,tid,t_per_atom,offset,eflag,vflag,
                ans,engv);
}

__kernel void k_lj_bump_smooth_linear_fast(const __global numtyp4 *restrict x_,
                        const __global numtyp4 *restrict lj1_in,
                        const __global numtyp4 *restrict lj3_in,
                        const __global numtyp *restrict sp_lj_in,
                        const __global numtyp4 *restrict bump_data_in,
                        const __global numtyp4 *restrict smooth_linear_data_in,
                        const __global int * dev_nbor,
                        const __global int * dev_packed,
                        __global acctyp4 *restrict ans,
                        __global acctyp *restrict engv,
                        const int eflag, const int vflag, const int inum,
                        const int nbor_pitch, const int t_per_atom) {
  int tid, ii, offset;
  atom_info(t_per_atom,ii,tid,offset);

  // for bump
  numtyp r, btmp;
  numtyp rinv;
  
  #ifndef ONETYPE
  __local numtyp4 lj1[MAX_SHARED_TYPES*MAX_SHARED_TYPES];
  __local numtyp4 lj3[MAX_SHARED_TYPES*MAX_SHARED_TYPES];
  __local numtyp4 bump_data[MAX_SHARED_TYPES*MAX_SHARED_TYPES];
  __local numtyp4 smooth_linear_data[MAX_SHARED_TYPES*MAX_SHARED_TYPES];
  __local numtyp sp_lj[4];
  if (tid<4)
    sp_lj[tid]=sp_lj_in[tid];
  if (tid<MAX_SHARED_TYPES*MAX_SHARED_TYPES) {
    lj1[tid]=lj1_in[tid];
    bump_data[tid]=bump_data_in[tid];
    smooth_linear_data[tid]=smooth_linear_data_in[tid];
    if (EVFLAG && eflag)
      lj3[tid]=lj3_in[tid];
  }
  __syncthreads();
  #else
  const numtyp lj1x=lj1_in[ONETYPE].x;
  const numtyp lj1y=lj1_in[ONETYPE].y;
  const numtyp cutsq=lj1_in[ONETYPE].z;
  const numtyp bumpx=bump_data_in[ONETYPE].x;
  const numtyp bumpy=bump_data_in[ONETYPE].y;
  const numtyp bumpz=bump_data_in[ONETYPE].z;
  const numtyp slx=smooth_linear_data_in[ONETYPE].x;
  const numtyp sly=smooth_linear_data_in[ONETYPE].y;
  const numtyp slz=smooth_linear_data_in[ONETYPE].z;
  numtyp lj3x, lj3y; //, lj3z;
  if (EVFLAG && eflag) {
    lj3x=lj3_in[ONETYPE].x;
    lj3y=lj3_in[ONETYPE].y;
    //lj3z=lj3_in[ONETYPE].z;
  }
  #endif

  int n_stride;
  local_allocate_store_pair();

  acctyp4 f;
  f.x=(acctyp)0; f.y=(acctyp)0; f.z=(acctyp)0;
  acctyp energy, virial[6];
  if (EVFLAG) {
    energy=(acctyp)0;
    for (int i=0; i<6; i++) virial[i]=(acctyp)0;
  }

  if (ii<inum) {
    int i, numj, nbor, nbor_end;
    nbor_info(dev_nbor,dev_packed,nbor_pitch,t_per_atom,ii,offset,i,numj,
              n_stride,nbor_end,nbor);

    numtyp4 ix; fetch4(ix,i,pos_tex); //x_[i];
    #ifndef ONETYPE
    int iw=ix.w;
    int itype=fast_mul((int)MAX_SHARED_TYPES,iw);
    numtyp factor_lj;
    #endif

    NOUNROLL
    for ( ; nbor<nbor_end; nbor+=n_stride) {
      int j=dev_packed[nbor];
      #ifndef ONETYPE
      factor_lj = sp_lj[sbmask(j)];
      j &= NEIGHMASK;
      #endif

      numtyp4 jx; fetch4(jx,j,pos_tex); //x_[j];
      #ifndef ONETYPE
      int mtype=itype+jx.w;
      numtyp cutsq=lj1[mtype].z;
      #endif

      // Compute r12
      numtyp delx = ix.x-jx.x;
      numtyp dely = ix.y-jx.y;
      numtyp delz = ix.z-jx.z;
      numtyp r2inv = delx*delx+dely*dely+delz*delz;

      if (r2inv<cutsq) {
        #ifndef ONETYPE
        numtyp lj1x=lj1[mtype].x;
        numtyp lj1y=lj1[mtype].y;
        numtyp bumpx=bump_data[mtype].x;
        numtyp bumpy=bump_data[mtype].y;
        numtyp bumpz=bump_data[mtype].z;
        numtyp slx=smooth_linear_data[mtype].x;
        numtyp sly=smooth_linear_data[mtype].y;
        numtyp slz=smooth_linear_data[mtype].z;
        #endif
        r = ucl_sqrt(r2inv);
        r2inv = ucl_recip(r2inv); // This makes r2inv the actual r^2 inverse
        rinv = ucl_recip(r);
        numtyp r6inv = r2inv*r2inv*r2inv;
        numtyp force = r6inv*(lj1x*r6inv-lj1y);
        force = rinv*(rinv*force - sly);
        #ifndef ONETYPE
        force*=factor_lj;
        #endif

        numtyp inv_bdiff;

        // for bump
        if(r >= bumpx && r <= bumpy) {
            inv_bdiff = ucl_recip(bumpy-bumpx);
            force += -bumpz*M_PI*sinpi((bumpy+bumpx-r-r)*inv_bdiff)*inv_bdiff*rinv;
        }

        f.x+=delx*force;
        f.y+=dely*force;
        f.z+=delz*force;

        if (EVFLAG && eflag) {
            #ifndef ONETYPE
            numtyp lj3x=lj3[mtype].x;
            numtyp lj3y=lj3[mtype].y;
            //numtyp lj3z=lj3[mtype].z;
            #endif
            numtyp e=r6inv*(lj3x*r6inv-lj3y);
            e -= slx - (r-slz)*sly;
            //bump
            if(r >= bumpx && r <= bumpy) {
                btmp = sinpi((bumpy-r)*inv_bdiff);
                e += bumpz*btmp*btmp;
            }
          #ifndef ONETYPE
          energy+=(e); //factor_lj*(e);//-lj3z);
          #else
          energy+=(e); //-lj3z);
          #endif
          
        }
        if (EVFLAG && vflag) {
          virial[0] += delx*delx*force;
          virial[1] += dely*dely*force;
          virial[2] += delz*delz*force;
          virial[3] += delx*dely*force;
          virial[4] += delx*delz*force;
          virial[5] += dely*delz*force;
        }
      }
    } // for nbor
  } // if ii
  store_answers(f,energy,virial,ii,inum,tid,t_per_atom,offset,eflag,vflag,
                ans,engv);
}

