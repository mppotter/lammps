/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef NSTENCIL_CLASS
// clang-format off
typedef NStencilGhostBin<0, 0, 0> NStencilFullGhostBin2d;
NStencilStyle(full/ghost/bin/2d,
              NStencilFullGhostBin2d,
              NS_FULL | NS_GhostBin | NS_2D | NS_ORTHO | NS_TRI);

typedef NStencilGhostBin<0, 1, 0> NStencilFullGhostBin3d;
NStencilStyle(full/ghost/bin/3d,
              NStencilFullGhostBin3d,
              NS_FULL | NS_GhostBin | NS_3D | NS_ORTHO | NS_TRI);

typedef NStencilGhostBin<1, 0, 0> NStencilHalfGhostBin2d;
NStencilStyle(half/ghost/bin/2d,
              NStencilHalfGhostBin2d,
              NS_HALF | NS_GhostBin | NS_2D | NS_ORTHO);

typedef NStencilGhostBin<1, 0, 1> NStencilHalfGhostBin2dTri;
NStencilStyle(half/ghost/bin/2d/tri,
              NStencilHalfGhostBin2dTri,
              NS_HALF | NS_GhostBin | NS_2D | NS_TRI);

typedef NStencilGhostBin<1, 1, 0> NStencilHalfGhostBin3d;
NStencilStyle(half/ghost/bin/3d,
              NStencilHalfGhostBin3d,
              NS_HALF | NS_GhostBin | NS_3D | NS_ORTHO);

typedef NStencilGhostBin<1, 1, 1> NStencilHalfGhostBin3dTri;
NStencilStyle(half/ghost/bin/3d/tri,
              NStencilHalfGhostBin3dTri,
              NS_HALF | NS_GhostBin | NS_3D | NS_TRI);
// clang-format on
#else

#ifndef LMP_NSTENCIL_GHOST_BIN_H
#define LMP_NSTENCIL_GHOST_BIN_H

#include "nstencil.h"

namespace LAMMPS_NS {

template<int HALF, int DIM_3D, int TRI>
class NStencilGhostBin : public NStencil {
 public:
  NStencilGhostBin(class LAMMPS *);
  void create() override;
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

*/
