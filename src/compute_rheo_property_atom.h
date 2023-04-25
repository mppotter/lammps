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

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(rheo/property/atom,ComputeRHEOPropertyAtom);
// clang-format on
#else

#ifndef LMP_COMPUTE_RHEO_PROPERTY_ATOM_H
#define LMP_COMPUTE_RHEO_PROPERTY_ATOM_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeRHEOPropertyAtom : public Compute {
 public:
  ComputeRHEOPropertyAtom(class LAMMPS *, int, char **);
  ~ComputeRHEOPropertyAtom() override;
  void init() override;
  void compute_peratom() override;
  double memory_usage() override;

 private:
  int nvalues, nmax;
  int thermal_flag, interface_flag, surface_flag, shift_flag;
  int *index;
  double *buf;

  typedef void (ComputeRHEOPropertyAtom::*FnPtrPack)(int);
  FnPtrPack *pack_choice;    // ptrs to pack functions

  void pack_phase(int);
  void pack_chi(int);
  void pack_surface(int);
  void pack_surface_r(int);
  void pack_surface_divr(int);
  void pack_surface_nx(int);
  void pack_surface_ny(int);
  void pack_surface_nz(int);
  void pack_coordination(int);
  void pack_cv(int);
  void pack_shift_vx(int);
  void pack_shift_vy(int);
  void pack_shift_vz(int);
  void pack_atom_style(int);

  class FixRHEO *fix_rheo;
  class FixRHEOThermal *fix_thermal;
  class ComputeRHEOInterface *compute_interface;
  class ComputeRHEOKernel *compute_kernel;
  class ComputeRHEOSurface *compute_surface;
  class ComputeRHEOVShift *compute_vshift;

};

}    // namespace LAMMPS_NS

#endif
#endif
