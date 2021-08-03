// **************************************************************************
//                                   lj_bump.cu
//                             -------------------
//                              Mark Potter (RPI)
//
//  Device code for acceleration of the lj/cut/bump pair style
//
// __________________________________________________________________________
//    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
// __________________________________________________________________________
//
//    begin                :
//    email                : pottem3@rpi.edu
// ***************************************************************************

#include <math.h>

#if defined(NV_KERNEL) || defined(USE_HIP)
#include "lal_aux_fun1.h"
#ifndef _DOUBLE_DOUBLE
_texture( pos_tex,float4);
#else
_texture_2d( pos_tex,int4);
#endif
#else
#define pos_tex x_
#endif

__kernel void k_lj_bump(const __global numtyp4 *restrict x_,
                        const __global numtyp4 *restrict lj1,
                        const __global numtyp4 *restrict lj3,
                        const int lj_types,
                        const __global numtyp *restrict sp_lj,
                        const __global numtyp4 *restrict bump_data,
                        const __global int * dev_nbor,
                        const __global int * dev_packed,
                        __global acctyp4 *restrict ans,
                        __global acctyp *restrict engv,
                        const int eflag, const int vflag, const int inum,
                        const int nbor_pitch,
                        const int t_per_atom) {
  int tid, ii, offset;
  atom_info(t_per_atom,ii,tid,offset);

  // for bump
  numtyp rtmp, btmp;

  acctyp energy=(acctyp)0;
  acctyp4 f;
  f.x=(acctyp)0; f.y=(acctyp)0; f.z=(acctyp)0;
  acctyp virial[6];
  for (int i=0; i<6; i++)
    virial[i]=(acctyp)0;

  if (ii<inum) {
    int i, numj, nbor, nbor_end;
    __local int n_stride;
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
        r2inv=ucl_recip(r2inv);
        numtyp r6inv = r2inv*r2inv*r2inv;
        numtyp force = r2inv*r6inv*(lj1[mtype].x*r6inv-lj1[mtype].y);
        force*=factor_lj;

        // for bump
        rtmp = sqrt(r2inv);
        if(rtmp >= bump_data[mtype].x && rtmp <= bump_data[mtype].y) {
            force += -bump_data[mtype].z*M_PI*sinpi((bump_data[mtype].y+bump_data[mtype].x-rtmp-rtmp)/(bump_data[mtype].y-bump_data[mtype].x))/(bump_data[mtype].y-bump_data[mtype].x)/rtmp;
        }

        f.x+=delx*force;
        f.y+=dely*force;
        f.z+=delz*force;

        if (eflag>0) {
          numtyp e=r6inv*(lj3[mtype].x*r6inv-lj3[mtype].y);
          //bump
          if(rtmp >= bump_data[mtype].x && rtmp <= bump_data[mtype].y) {
            btmp = sinpi((bump_data[mtype].y-rtmp)/(bump_data[mtype].y-bump_data[mtype].x));
            e += bump_data[mtype].z*btmp*btmp;
          }
          energy+=factor_lj*(e-lj3[mtype].z);
        }
        if (vflag>0) {
          virial[0] += delx*delx*force;
          virial[1] += dely*dely*force;
          virial[2] += delz*delz*force;
          virial[3] += delx*dely*force;
          virial[4] += delx*delz*force;
          virial[5] += dely*delz*force;
        }
      }

    } // for nbor
    store_answers(f,energy,virial,ii,inum,tid,t_per_atom,offset,eflag,vflag,
                  ans,engv);
  } // if ii
}

__kernel void k_lj_bump_fast(const __global numtyp4 *restrict x_,
                        const __global numtyp4 *restrict lj1_in,
                        const __global numtyp4 *restrict lj3_in,
                        const __global numtyp *restrict sp_lj_in,
                        const __global numtyp4 *restrict bump_data_in,
                        const __global int * dev_nbor,
                        const __global int * dev_packed,
                        __global acctyp4 *restrict ans,
                        __global acctyp *restrict engv,
                        const int eflag, const int vflag, const int inum,
                        const int nbor_pitch, const int t_per_atom) {
  int tid, ii, offset;
  atom_info(t_per_atom,ii,tid,offset);

  // for bump
  numtyp rtmp, btmp;

  __local numtyp4 lj1[MAX_SHARED_TYPES*MAX_SHARED_TYPES];
  __local numtyp4 lj3[MAX_SHARED_TYPES*MAX_SHARED_TYPES];
  __local numtyp4 bump_data[MAX_SHARED_TYPES*MAX_SHARED_TYPES];
  __local numtyp sp_lj[4];
  if (tid<4)
    sp_lj[tid]=sp_lj_in[tid];
  if (tid<MAX_SHARED_TYPES*MAX_SHARED_TYPES) {
    lj1[tid]=lj1_in[tid];
    bump_data[tid]=bump_data_in[tid];
    if (eflag>0)
      lj3[tid]=lj3_in[tid];
  }

  acctyp energy=(acctyp)0;
  acctyp4 f;
  f.x=(acctyp)0; f.y=(acctyp)0; f.z=(acctyp)0;
  acctyp virial[6];
  for (int i=0; i<6; i++)
    virial[i]=(acctyp)0;

  __syncthreads();

  if (ii<inum) {
    int i, numj, nbor, nbor_end;
    __local int n_stride;
    nbor_info(dev_nbor,dev_packed,nbor_pitch,t_per_atom,ii,offset,i,numj,
              n_stride,nbor_end,nbor);

    numtyp4 ix; fetch4(ix,i,pos_tex); //x_[i];
    int iw=ix.w;
    int itype=fast_mul((int)MAX_SHARED_TYPES,iw);

    numtyp factor_lj;
    for ( ; nbor<nbor_end; nbor+=n_stride) {

      int j=dev_packed[nbor];
      factor_lj = sp_lj[sbmask(j)];
      j &= NEIGHMASK;

      numtyp4 jx; fetch4(jx,j,pos_tex); //x_[j];
      int mtype=itype+jx.w;

      // Compute r12
      numtyp delx = ix.x-jx.x;
      numtyp dely = ix.y-jx.y;
      numtyp delz = ix.z-jx.z;
      numtyp r2inv = delx*delx+dely*dely+delz*delz;

      if (r2inv<lj1[mtype].z) {
        r2inv=ucl_recip(r2inv);
        numtyp r6inv = r2inv*r2inv*r2inv;
        numtyp force = factor_lj*r2inv*r6inv*(lj1[mtype].x*r6inv-lj1[mtype].y);

        // for bump
        rtmp = sqrt(r2inv);
        if(rtmp >= bump_data[mtype].x && rtmp <= bump_data[mtype].y) {
            force += -bump_data[mtype].z*M_PI*sinpi((bump_data[mtype].y+bump_data[mtype].x-rtmp-rtmp)/(bump_data[mtype].y-bump_data[mtype].x))/(bump_data[mtype].y-bump_data[mtype].x)/rtmp;
        }

        f.x+=delx*force;
        f.y+=dely*force;
        f.z+=delz*force;

        if (eflag>0) {
          numtyp e=r6inv*(lj3[mtype].x*r6inv-lj3[mtype].y);
            //bump
            if(rtmp >= bump_data[mtype].x && rtmp <= bump_data[mtype].y) {
                btmp = sinpi((bump_data[mtype].y-rtmp)/(bump_data[mtype].y-bump_data[mtype].x));
                e += bump_data[mtype].z*btmp*btmp;
            }
          energy+=factor_lj*(e-lj3[mtype].z);
        }
        if (vflag>0) {
          virial[0] += delx*delx*force;
          virial[1] += dely*dely*force;
          virial[2] += delz*delz*force;
          virial[3] += delx*dely*force;
          virial[4] += delx*delz*force;
          virial[5] += dely*delz*force;
        }
      }

    } // for nbor
    store_answers(f,energy,virial,ii,inum,tid,t_per_atom,offset,eflag,vflag,
                  ans,engv);
  } // if ii
}

