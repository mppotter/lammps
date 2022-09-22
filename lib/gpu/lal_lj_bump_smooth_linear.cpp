/***************************************************************************
                          lj_bump_smooth_linear.cpp
                             -------------------
                              Mark Potter (RPI)

  Class for acceleration of the lj/bump/smooth/linear pair style.

 __________________________________________________________________________
    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
 __________________________________________________________________________

    begin                :
    email                : pottem3@rpi.edu
 ***************************************************************************/

#if defined(USE_OPENCL)
#include "lj_cl.h"
#elif defined(USE_CUDART)
const char *lj_bump_smooth_linear=0;
#else
#include "lj_bump_smooth_linear_cubin.h"
#endif

#include "lal_lj_bump_smooth_linear.h"
#include <cassert>
namespace LAMMPS_AL {
#define LJBumpSmoothLinearT LJBumpSmoothLinear<numtyp, acctyp>

extern Device<PRECISION,ACC_PRECISION> device;

template <class numtyp, class acctyp>
LJBumpSmoothLinearT::LJBumpSmoothLinear() : BaseAtomic<numtyp,acctyp>(), _allocated(false) {
}

template <class numtyp, class acctyp>
LJBumpSmoothLinearT::~LJBumpSmoothLinear() {
  clear();
}

template <class numtyp, class acctyp>
int LJBumpSmoothLinearT::bytes_per_atom(const int max_nbors) const {
  return this->bytes_per_atom_atomic(max_nbors);
}

template <class numtyp, class acctyp>
int LJBumpSmoothLinearT::init(const int ntypes,
                              double **host_cutsq, double **host_lj1,
                              double **host_lj2, double **host_lj3,
                              double **host_lj4, /*double **host_offset,*/
                              double *host_special_lj, const int nlocal,
                              const int nall, double **host_start_bump,
                              double **host_end_bump, double **host_energy_bump,
                              double **host_ljcut, double **host_dljcut,
                              double **host_cut, const int max_nbors,
                              const int maxspecial, const double cell_size,
                              const double gpu_split, FILE *_screen) {
  const int max_shared_types=this->device->max_shared_types();

  int onetype=0;
  #ifdef USE_OPENCL
  if (maxspecial==0)
    for (int i=1; i<ntypes; i++)
      for (int j=i; j<ntypes; j++)
        if (host_cutsq[i][j]>0) {
          if (onetype>0)
            onetype=-1;
          else if (onetype==0)
            onetype=i*max_shared_types+j;
        }
  if (onetype<0) onetype=0;
  #endif

  int success;
  success=this->init_atomic(nlocal,nall,max_nbors,maxspecial,cell_size,
                            gpu_split,_screen,lj_bump_smooth_linear,"k_lj_bump_smooth_linear",onetype);
  if (success!=0)
    return success;

  // If atom type constants fit in shared memory use fast kernel
  int lj_types=ntypes;
  shared_types=false;
  if (lj_types<=max_shared_types && this->_block_size>=max_shared_types) {
    lj_types=max_shared_types;
    shared_types=true;
  }
  _lj_types=lj_types;

  // Allocate a host write buffer for data initialization
  UCL_H_Vec<numtyp> host_write(lj_types*lj_types*32,*(this->ucl_device),
                               UCL_WRITE_ONLY);

  for (int i=0; i<lj_types*lj_types; i++)
    host_write[i]=0.0;

  lj1.alloc(lj_types*lj_types,*(this->ucl_device),UCL_READ_ONLY);
  this->atom->type_pack4(ntypes,lj_types,lj1,host_write,host_lj1,host_lj2,
                         host_cutsq);

  lj3.alloc(lj_types*lj_types,*(this->ucl_device),UCL_READ_ONLY);
  this->atom->type_pack4(ntypes,lj_types,lj3,host_write,host_lj3,host_lj4
                         , host_cut /*host_offset*/);

  bump_data.alloc(lj_types*lj_types,*(this->ucl_device),UCL_READ_ONLY);
  this->atom->type_pack4(ntypes,lj_types,bump_data,host_write,host_start_bump,
                         host_end_bump,host_energy_bump);

  smooth_linear_data.alloc(lj_types*lj_types,*(this->ucl_device),UCL_READ_ONLY);
  this->atom->type_pack4(ntypes,lj_types,smooth_linear_data,host_write,
                         host_ljcut,host_dljcut,host_cut);

  UCL_H_Vec<double> dview;
  sp_lj.alloc(4,*(this->ucl_device),UCL_READ_ONLY);
  dview.view(host_special_lj,4,*(this->ucl_device));
  ucl_copy(sp_lj,dview,false);

  _allocated=true;
  this->_max_bytes=lj1.row_bytes()+lj3.row_bytes()+sp_lj.row_bytes()
                      +bump_data.row_bytes()+smooth_linear_data.row_bytes();
  return 0;
}

template <class numtyp, class acctyp>
void LJBumpSmoothLinearT::reinit(const int ntypes, double **host_cutsq, double **host_lj1,
                                 double **host_lj2, double **host_lj3,
                                 double **host_lj4, /*double **host_offset,*/
                                 double **host_start_bump, double **host_end_bump,
                                 double **host_energy_bump,
                                 double **host_ljcut, double **host_dljcut,
                                 double **host_cut) {
  // Allocate a host write buffer for data initialization
  UCL_H_Vec<numtyp> host_write(_lj_types*_lj_types*32,*(this->ucl_device),
                               UCL_WRITE_ONLY);

  for (int i=0; i<_lj_types*_lj_types; i++)
    host_write[i]=0.0;

  this->atom->type_pack4(ntypes,_lj_types,lj1,host_write,host_lj1,host_lj2,
                         host_cutsq);
  this->atom->type_pack4(ntypes,_lj_types,lj3,host_write,host_lj3,host_lj4,
                         host_cut /*host_offset*/);
  this->atom->type_pack4(ntypes,_lj_types,bump_data,host_write,host_start_bump,
                         host_end_bump,host_energy_bump);
  this->atom->type_pack4(ntypes,_lj_types,lj3,host_write,host_ljcut,
                         host_dljcut,host_cut);
}

template <class numtyp, class acctyp>
void LJBumpSmoothLinearT::clear() {
  if (!_allocated)
    return;
  _allocated=false;

  lj1.clear();
  lj3.clear();
  sp_lj.clear();
  bump_data.clear();
  smooth_linear_data.clear();
  this->clear_atomic();
}

template <class numtyp, class acctyp>
double LJBumpSmoothLinearT::host_memory_usage() const {
  return this->host_memory_usage_atomic()+sizeof(LJBumpSmoothLinear<numtyp,acctyp>);
}

// ---------------------------------------------------------------------------
// Calculate energies, forces, and torques
// ---------------------------------------------------------------------------
template <class numtyp, class acctyp>
int LJBumpSmoothLinearT::loop(const int eflag, const int vflag) {
  // Compute the block size and grid size to keep all cores busy
  const int BX=this->block_size();
  int GX=static_cast<int>(ceil(static_cast<double>(this->ans->inum())/
                               (BX/this->_threads_per_atom)));

  int ainum=this->ans->inum();
  int nbor_pitch=this->nbor->nbor_pitch();
  this->time_pair.start();
  if (shared_types) {
    this->k_pair_sel->set_size(GX,BX);
    this->k_pair_sel->run(&this->atom->x, &lj1, &lj3, &sp_lj, &bump_data,
                          &smooth_linear_data,
                          &this->nbor->dev_nbor, &this->_nbor_data->begin(),
                          &this->ans->force, &this->ans->engv, &eflag,
                          &vflag, &ainum, &nbor_pitch,
                          &this->_threads_per_atom);
  } else {
    this->k_pair.set_size(GX,BX);
    this->k_pair.run(&this->atom->x, &lj1, &lj3, &_lj_types, &sp_lj, &bump_data,
                     &smooth_linear_data,
                     &this->nbor->dev_nbor, &this->_nbor_data->begin(),
                     &this->ans->force, &this->ans->engv, &eflag, &vflag,
                     &ainum, &nbor_pitch, &this->_threads_per_atom);
  }
  this->time_pair.stop();
  return GX;
}

template class LJBumpSmoothLinear<PRECISION,ACC_PRECISION>;
}
