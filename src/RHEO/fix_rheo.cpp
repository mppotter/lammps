/* ----------------------------------------------------------------------
 LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
 https://www.lammps.org/, Sandia National Laboratories
 LAMMPS development team: developers@lammps.org

 Copyright (2003) Sandia Corporation.  Under the terms of Contract
 DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
 certain rights in this software.  This software is distributed under
 the GNU General Public License.

 See the README file in the top-level LAMMPS directory.
 ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors:
   Joel Clemmer (SNL), Thomas O'Connor (CMU), Eric Palermo (CMU)
----------------------------------------------------------------------- */

#include "fix_rheo.h"

#include "atom.h"
#include "compute_rheo_grad.h"
#include "compute_rheo_interface.h"
#include "compute_rheo_kernel.h"
#include "compute_rheo_rhosum.h"
#include "compute_rheo_vshift.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "modify.h"
#include "update.h"
#include "utils.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixRHEO::FixRHEO(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), compute_grad(nullptr), compute_kernel(nullptr),
  compute_interface(nullptr), compute_rhosum(nullptr), compute_vshift(nullptr)
{
  time_integrate = 1;

  viscosity_fix_defined = 0;
  pressure_fix_defined = 0;
  thermal_fix_defined = 0;
  surface_fix_defined = 0;

  thermal_flag = 0;
  rhosum_flag = 0;
  shift_flag = 0;
  interface_flag = 0;
  surface_flag = 0;

  rho0 = 1.0;
  csq = 1.0;

  if (igroup != 0)
    error->all(FLERR,"fix rheo command requires group all");

  if (atom->rho_flag != 1)
    error->all(FLERR,"fix rheo command requires atom_style with density");
  if (atom->status_flag != 1)
    error->all(FLERR,"fix rheo command requires atom_style with status");

  if (narg < 5)
    error->all(FLERR,"Insufficient arguments for fix rheo command");

  h = utils::numeric(FLERR,arg[3],false,lmp);
  if (strcmp(arg[4],"Quintic") == 0) {
      kernel_style = QUINTIC;
  } else if (strcmp(arg[4],"CRK0") == 0) {
      kernel_style = CRK0;
  } else if (strcmp(arg[4],"CRK1") == 0) {
      kernel_style = CRK1;
  } else if (strcmp(arg[4],"CRK2") == 0) {
      kernel_style = CRK2;
  } else error->all(FLERR,"Unknown kernel style {} in fix rheo", arg[4]);
  zmin_kernel = utils::numeric(FLERR,arg[5],false,lmp);

  int iarg = 6;
  while (iarg < narg){
    if (strcmp(arg[iarg],"shift") == 0) {
      shift_flag = 1;
    } else if (strcmp(arg[iarg],"thermal") == 0) {
      thermal_flag = 1;
    } else if (strcmp(arg[iarg],"surface/detection") == 0) {
      surface_flag = 1;
    } else if (strcmp(arg[iarg],"interface/reconstruction") == 0) {
      interface_flag = 1;
    } else if (strcmp(arg[iarg],"rhosum") == 0) {
      rhosum_flag = 1;
      if(iarg + 1 >= narg) error->all(FLERR,"Illegal rhosum option in fix rheo");
      zmin_rhosum = utils::inumeric(FLERR,arg[iarg + 1],false,lmp);
      iarg += 1;
    } else if (strcmp(arg[iarg],"rho0") == 0) {
      if(iarg + 1 >= narg) error->all(FLERR,"Illegal rho0 option in fix rheo");
      rho0 = utils::numeric(FLERR,arg[iarg + 1],false,lmp);
      iarg += 1;
    } else if (strcmp(arg[iarg],"csq") == 0) {
      if(iarg+1 >= narg) error->all(FLERR,"Illegal csq option in fix rheo");
      csq = utils::numeric(FLERR,arg[iarg + 1],false,lmp);
      iarg += 1;
    } else {
      error->all(FLERR, "Illegal fix rheo command: {}", arg[iarg]);
    }
    iarg += 1;
  }
}

/* ---------------------------------------------------------------------- */

FixRHEO::~FixRHEO()
{
  if (compute_kernel) modify->delete_compute("rheo_kernel");
  if (compute_grad) modify->delete_compute("rheo_grad");
  if (compute_interface) modify->delete_compute("rheo_interface");
  if (compute_rhosum) modify->delete_compute("rheo_rhosum");
  if (compute_vshift) modify->delete_compute("rheo_vshift");
}


/* ----------------------------------------------------------------------
  Create necessary internal computes
------------------------------------------------------------------------- */

void FixRHEO::post_constructor()
{
  compute_kernel = dynamic_cast<ComputeRHEOKernel *>(modify->add_compute("rheo_kernel all rheo/kernel"));
  compute_kernel->fix_rheo = this;

  std::string cmd = "rheo_grad all rheo/grad velocity rho viscosity";
  if (thermal_flag) cmd += "temperature";
  compute_grad = dynamic_cast<ComputeRHEOGrad *>(modify->add_compute(cmd));
  compute_grad->fix_rheo = this;

  if (rhosum_flag) {
    compute_rhosum = dynamic_cast<ComputeRHEORhoSum *>(modify->add_compute("rheo_rhosum all rheo/rho/sum"));
    compute_rhosum->fix_rheo = this;
  }

  if (shift_flag) {
    compute_vshift = dynamic_cast<ComputeRHEOVShift *>(modify->add_compute("rheo_vshift all rheo/vshift"));
    compute_vshift->fix_rheo = this;
  }

  if (interface_flag) {
    compute_interface = dynamic_cast<ComputeRHEOInterface *>(modify->add_compute(fmt::format("rheo_interface all rheo/interface")));
    compute_interface->fix_rheo = this;
  }
}

/* ---------------------------------------------------------------------- */

int FixRHEO::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixRHEO::init()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;

  if (modify->get_fix_by_style("rheo").size() > 1)
    error->all(FLERR,"Can only specify one instance of fix rheo");
}

/* ---------------------------------------------------------------------- */

void FixRHEO::setup_pre_force(int /*vflag*/)
{
  // Check to confirm accessory fixes do not preceed FixRHEO
  // Note: these fixes set this flag in setup_pre_force()
  if (viscosity_fix_defined || pressure_fix_defined || thermal_fix_defined || surface_fix_defined)
    error->all(FLERR, "Fix RHEO must be defined before all other RHEO fixes");

  pre_force(0);
}

/* ---------------------------------------------------------------------- */

void FixRHEO::setup()
{
  // Confirm all accessory fixes are defined
  // Note: these fixes set this flag in setup_pre_force()
  if (!viscosity_fix_defined)
    error->all(FLERR, "Missing fix rheo/viscosity");

  if (!pressure_fix_defined)
    error->all(FLERR, "Missing fix rheo/pressure");

  if(!thermal_fix_defined && thermal_flag)
    error->all(FLERR, "Missing fix rheo/thermal");

  if(!surface_fix_defined && surface_flag)
    error->all(FLERR, "Missing fix rheo/surface");

  // Reset to zero for next run
  thermal_fix_defined = 0;
  viscosity_fix_defined = 0;
  pressure_fix_defined = 0;
  surface_fix_defined = 0;

  // Check fixes cover all atoms (doesnt ensure user covers atoms created midrun)
  // (pressure is currently required to be group all)
  auto visc_fixes = modify->get_fix_by_style("rheo/viscosity");
  auto therm_fixes = modify->get_fix_by_style("rheo/thermal");

  int *mask = atom->mask;
  int v_coverage_flag = 1;
  int t_coverage_flag = 1;
  int covered;
  for (int i = 0; i < atom->nlocal; i++) {
    covered = 0;
    for (auto fix in visc_fixes)
      if (mask[i] & fix->groupbit) covered = 1;
    if (!covered) v_coverage_flag = 0;
    if (thermal_flag) {
      covered = 0;
      for (auto fix in therm_fixes)
        if (mask[i] & fix->groupbit) covered = 1;
      if (!covered) v_coverage_flag = 0;
    }
  }

  if (!v_coverage_flag)
    error->one(FLERR, "Fix rheo/viscosity does not fully cover all atoms");
  if (!t_coverage_flag)
    error->one(FLERR, "Fix rheo/thermal does not fully cover all atoms");
}

/* ---------------------------------------------------------------------- */

void FixRHEO::initial_integrate(int /*vflag*/)
{
  // update v and x and rho of atoms in group
  int i, a, b;
  double dtfm, divu;
  int dim = domain->dimension;

  int *status = atom->status;
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rho = atom->rho;
  double *drho = atom->drho;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int rmass_flag = atom->rmass_flag;

  double **gradr = compute_grad->gradr;
  double **gradv = compute_grad->gradv;
  double **vshift = compute_vshift->array_atom;

  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  if (igroup == atom->firstgroup)
    nlocal = atom->nfirst;

  //Density Half-step
  for (i = 0; i < nlocal; i++) {
    if (status[i] & STATUS_NO_FORCE) continue;

    if (mask[i] & groupbit) {
      if (rmass_flag) {
        dtfm = dtf / rmass[i];
      } else {
        dtfm = dtf / mass[type[i]];
      }

      v[i][0] += dtfm * f[i][0];
      v[i][1] += dtfm * f[i][1];
      v[i][2] += dtfm * f[i][2];
    }
  }

  // Update gradients and interpolate solid properties
  compute_grad->forward_fields(); // also forwards v and rho for chi
  compute_interface->store_forces(); // Need to save, wiped in exchange
  compute_interface->compute_peratom();
  compute_grad->compute_peratom();

  // Position half-step
  for (i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      for (a = 0; a < dim; a++) {
        x[i][a] += dtv * v[i][a];
      }
    }
  }

  // Update density using div(u)
  if (!rhosum_flag) {
    for (i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        if (status[i] & STATUS_NO_FORCE) continue;
        if (!(status[i] & STATUS_FLUID)) continue;

        divu = 0;
        for (a = 0; a < dim; a++) {
          divu += gradv[i][a * (1 + dim)];
        }
        rho[i] += dtf * (drho[i] - rho[i] * divu);
      }
    }
  }

  // Shifting atoms
  if (shift_flag) {
    compute_vshift->correct_surfaces(); // COuld this be moved to preforce after the surface fix runs?
    for (i = 0; i < nlocal; i++) {

      if (!(status[i] & STATUS_SHIFT)) continue;

      if (mask[i] & groupbit) {
        for (a = 0; a < dim; a++) {
          x[i][a] += dtv * vshift[i][a];
          for (b = 0; b < dim; b++) {
            v[i][a] += dtv * vshift[i][b] * gradv[i][a * dim + b];
          }
        }

        if (!rhosum_flag) {
          for (a = 0; a < dim; a++) {
            rho[i] += dtv * vshift[i][a] * gradr[i][a];
          }
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixRHEO::pre_force(int /*vflag*/)
{
  if (rhosum_flag)
    compute_rhosum->compute_peratom();

  compute_grad->forward_fields(); // also forwards v and rho for chi
  compute_kernel->compute_peratom();
  compute_interface->compute_peratom();

  compute_grad->compute_peratom();
  compute_grad->forward_gradients();

  if (shift_flag)
    compute_vshift->compute_peratom();

  // Remove extra shifting/no force options options
  int *status = atom->status;
  int nall = atom->nlocal + atom->nghost;
  for (int i = 0; i < nall; i++) {
    if (mask[i] & groupbit) {
      status[i] &= ~STATUS_NO_FORCE;

      if (status[i] & STATUS_FLUID)
        status[i] &= ~STATUS_SHIFT;
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixRHEO::final_integrate() {
  int *status = atom->status;
  double **gradv = compute_grad->gradv;
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;

  double *rho = atom->rho;
  double *drho = atom->drho;
  int *type = atom->type;
  int *mask = atom->mask;
  double *mass = atom->mass;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup)
    nlocal = atom->nfirst;
  double dtfm, divu;
  double *rmass = atom->rmass;
  int rmass_flag = atom->rmass_flag;
  int i, a;

  int dim = domain->dimension;

  // Update velocity
  for (i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      if (status[i] & STATUS_NO_FORCE) continue;

      if (rmass_flag) {
        dtfm = dtf / rmass[i];
      } else {
        dtfm = dtf / mass[type[i]];
      }

      for (a = 0; a < dim; a++) {
        v[i][a] += dtfm * f[i][a];
      }
    }
  }

  // Update density using divu
  if (!rhosum_flag) {
    for (i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        if (status[i] & STATUS_NO_FORCE) continue;
        if (!(status[i] & STATUS_FLUID)) continue;

        divu = 0;
        for (a = 0; a < dim; a++) {
          divu += gradv[i][a * (1 + dim)];
        }
        rho[i] += dtf * (drho[i] - rho[i] * divu);
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixRHEO::reset_dt()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
}
