/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef ANGLE_CLASS
// clang-format off
AngleStyle(spica/kk,AngleSPICAKokkos<LMPDeviceType>);
AngleStyle(spica/kk/device,AngleSPICAKokkos<LMPDeviceType>);
AngleStyle(spica/kk/host,AngleSPICAKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_ANGLE_SPICA_KOKKOS_H
#define LMP_ANGLE_SPICA_KOKKOS_H

#include "angle_spica.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

template<int NEWTON_BOND, int EVFLAG>
struct TagAngleSPICACompute{};

template<class DeviceType>
class AngleSPICAKokkos : public AngleSPICA {

 public:
  typedef DeviceType device_type;
  typedef EV_FLOAT value_type;

  AngleSPICAKokkos(class LAMMPS *);
  ~AngleSPICAKokkos() override;
  void compute(int, int) override;
  void coeff(int, char **) override;
  void read_restart(FILE *) override;

  template<int NEWTON_BOND, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagAngleSPICACompute<NEWTON_BOND,EVFLAG>, const int&, EV_FLOAT&) const;

  template<int NEWTON_BOND, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagAngleSPICACompute<NEWTON_BOND,EVFLAG>, const int&) const;

  //template<int NEWTON_BOND>
  KOKKOS_INLINE_FUNCTION
  void ev_tally(EV_FLOAT &ev, const int i, const int j, const int k,
                     F_FLOAT &eangle, F_FLOAT *f1, F_FLOAT *f3,
                     const F_FLOAT &delx1, const F_FLOAT &dely1, const F_FLOAT &delz1,
                     const F_FLOAT &delx2, const F_FLOAT &dely2, const F_FLOAT &delz2) const;

 protected:

  class NeighborKokkos *neighborKK;

  typename ArrayTypes<DeviceType>::t_x_array_randomread x;
  typename ArrayTypes<DeviceType>::t_f_array f;
  typename ArrayTypes<DeviceType>::t_int_2d anglelist;

  typename ArrayTypes<DeviceType>::tdual_efloat_1d k_eatom;
  typename ArrayTypes<DeviceType>::tdual_virial_array k_vatom;
  typename ArrayTypes<DeviceType>::t_efloat_1d d_eatom;
  typename ArrayTypes<DeviceType>::t_virial_array d_vatom;

  int nlocal,newton_bond;
  int eflag,vflag;

  typename ArrayTypes<DeviceType>::tdual_ffloat_1d k_k;
  typename ArrayTypes<DeviceType>::tdual_ffloat_1d k_theta0;
  typename ArrayTypes<DeviceType>::tdual_ffloat_1d k_repscale;
  typename ArrayTypes<DeviceType>::tdual_int_1d k_setflag;
  
  typename ArrayTypes<DeviceType>::t_ffloat_1d d_k;
  typename ArrayTypes<DeviceType>::t_ffloat_1d d_theta0;
  typename ArrayTypes<DeviceType>::t_ffloat_1d d_repscale;
  typename ArrayTypes<DeviceType>::t_int_1d d_setflag;
  
  void allocate() override;
};

}

#endif
#endif

