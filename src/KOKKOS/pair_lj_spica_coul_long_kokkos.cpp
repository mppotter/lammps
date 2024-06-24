// clang-format off
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

#include "pair_lj_spica_coul_long_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "ewald_const.h"
#include "force.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "respa.h"
#include "update.h"

#include "lj_spica_common.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace LJSPICAParms;
using namespace EwaldConst;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairLJSPICACoulLongKokkos<DeviceType>::PairLJSPICACoulLongKokkos(LAMMPS *lmp) : PairLJSPICACoulLong(lmp)
{
  respa_enable = 0;

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairLJSPICACoulLongKokkos<DeviceType>::~PairLJSPICACoulLongKokkos()
{
  if (copymode) return;

  if (allocated) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->destroy_kokkos(k_cutsq,cutsq);
    memoryKK->destroy_kokkos(k_cut_lj,cut_lj);
    memoryKK->destroy_kokkos(k_cut_ljsq,cut_ljsq);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairLJSPICACoulLongKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  eflag = eflag_in;
  vflag = vflag_in;

  if (neighflag == FULL) no_virial_fdotr_compute = 1;

  ev_init(eflag,vflag,0);

  // reallocate per-atom arrays if necessary

  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->create_kokkos(k_vatom,vatom,maxvatom,"pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }

  atomKK->sync(execution_space,datamask_read);
  k_cutsq.template sync<DeviceType>();
  k_params.template sync<DeviceType>();
  if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK);

  x = atomKK->k_x.view<DeviceType>();
  c_x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  q = atomKK->k_q.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  nlocal = atom->nlocal;
  nall = atom->nlocal + atom->nghost;
  special_lj[0] = force->special_lj[0];
  special_lj[1] = force->special_lj[1];
  special_lj[2] = force->special_lj[2];
  special_lj[3] = force->special_lj[3];
  special_coul[0] = force->special_coul[0];
  special_coul[1] = force->special_coul[1];
  special_coul[2] = force->special_coul[2];
  special_coul[3] = force->special_coul[3];
  qqrd2e = force->qqrd2e;
  newton_pair = force->newton_pair;

  // loop over neighbors of my atoms

  copymode = 1;

  EV_FLOAT ev;
  if (ncoultablebits)
    ev = pair_compute<PairLJSPICACoulLongKokkos<DeviceType>,CoulLongTable<1> >
      (this,(NeighListKokkos<DeviceType>*)list);
  else
    ev = pair_compute<PairLJSPICACoulLongKokkos<DeviceType>,CoulLongTable<0> >
      (this,(NeighListKokkos<DeviceType>*)list);

  if (eflag_global) {
    eng_vdwl += ev.evdwl;
    eng_coul += ev.ecoul;
  }

  if (vflag_global) {
    virial[0] += ev.v[0];
    virial[1] += ev.v[1];
    virial[2] += ev.v[2];
    virial[3] += ev.v[3];
    virial[4] += ev.v[4];
    virial[5] += ev.v[5];
  }

  if (eflag_atom) {
    k_eatom.template modify<DeviceType>();
    k_eatom.template sync<LMPHostType>();
  }

  if (vflag_atom) {
    k_vatom.template modify<DeviceType>();
    k_vatom.template sync<LMPHostType>();
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);

  copymode = 0;
}

/* ----------------------------------------------------------------------
   compute pair force between atoms i and j
   ---------------------------------------------------------------------- */

template<class DeviceType>
template<bool STACKPARAMS, class Specialisation>
KOKKOS_INLINE_FUNCTION
F_FLOAT PairLJSPICACoulLongKokkos<DeviceType>::
compute_fpair(const F_FLOAT& rsq, const int& i, const int&j, const int& itype, const int& jtype) const {
  (void) i;
  (void) j;
  const F_FLOAT r2inv = 1.0/rsq;
  const int ljt = (STACKPARAMS?m_params[itype][jtype].lj_type:params(itype,jtype).lj_type);

  const F_FLOAT lj_1 =  (STACKPARAMS?m_params[itype][jtype].lj1:params(itype,jtype).lj1);
  const F_FLOAT lj_2 =  (STACKPARAMS?m_params[itype][jtype].lj2:params(itype,jtype).lj2);

  /*if (ljt == LJ12_4) {

    const F_FLOAT r4inv=r2inv*r2inv;
    return r4inv*(lj_1*r4inv*r4inv - lj_2) * r2inv;

  } else if (ljt == LJ9_6) {

    const F_FLOAT r3inv = r2inv*sqrt(r2inv);
    const F_FLOAT r6inv = r3inv*r3inv;
    return r6inv*(lj_1*r3inv - lj_2) * r2inv;

  } else if (ljt == LJ12_6) {

    const double r6inv = r2inv*r2inv*r2inv;
    return r6inv*(lj_1*r6inv - lj_2) * r2inv;

  } else if (ljt == LJ12_5) {

    const F_FLOAT r5inv = r2inv*r2inv*sqrt(r2inv);
    const F_FLOAT r7inv = r5inv*r2inv;
    return r5inv*(lj_1*r7inv - lj_2) * r2inv;

  }
  if (ljt!=LJ12_4 && ljt!=LJ9_6 && ljt!=LJ12_6 && ljt!=LJ12_5) return 0.0;*/
  const F_FLOAT r4inv=r2inv*r2inv;
  const F_FLOAT r6inv=r2inv*r4inv;
  const F_FLOAT a = ljt==LJ12_4?r4inv:(ljt==LJ12_5?r4inv*sqrt(r2inv):r6inv);
  const F_FLOAT b = ljt==LJ12_4?r4inv:(ljt==LJ9_6?1.0/sqrt(r2inv):(ljt==LJ12_5?r2inv*sqrt(r2inv):r2inv));
  return a* ( lj_1*r6inv*b - lj_2 * r2inv);
}

/* ----------------------------------------------------------------------
   compute pair potential energy between atoms i and j
   ---------------------------------------------------------------------- */

template<class DeviceType>
template<bool STACKPARAMS, class Specialisation>
KOKKOS_INLINE_FUNCTION
F_FLOAT PairLJSPICACoulLongKokkos<DeviceType>::
compute_evdwl(const F_FLOAT& rsq, const int& i, const int&j, const int& itype, const int& jtype) const {
  (void) i;
  (void) j;
  const F_FLOAT r2inv = 1.0/rsq;
  const int ljt = (STACKPARAMS?m_params[itype][jtype].lj_type:params(itype,jtype).lj_type);

  const F_FLOAT lj_3 =  (STACKPARAMS?m_params[itype][jtype].lj3:params(itype,jtype).lj3);
  const F_FLOAT lj_4 =  (STACKPARAMS?m_params[itype][jtype].lj4:params(itype,jtype).lj4);
  const F_FLOAT offset =  (STACKPARAMS?m_params[itype][jtype].offset:params(itype,jtype).offset);

  if (ljt == LJ12_4) {
    const F_FLOAT r4inv=r2inv*r2inv;

    return r4inv*(lj_3*r4inv*r4inv - lj_4) - offset;

  } else if (ljt == LJ9_6) {
    const F_FLOAT r3inv = r2inv*sqrt(r2inv);
    const F_FLOAT r6inv = r3inv*r3inv;
    return r6inv*(lj_3*r3inv - lj_4) - offset;

  } else if (ljt == LJ12_6) {
    const double r6inv = r2inv*r2inv*r2inv;
    return r6inv*(lj_3*r6inv - lj_4) - offset;

  } else if (ljt == LJ12_5) {
    const F_FLOAT r5inv = r2inv*r2inv*sqrt(r2inv);
    const F_FLOAT r7inv = r5inv*r2inv;
    return r5inv*(lj_3*r7inv - lj_4) - offset;
  } else
    return 0.0;
}

/* ----------------------------------------------------------------------
   compute coulomb pair force between atoms i and j
------------------------------------------------------------------------- */

template<class DeviceType>
template<bool STACKPARAMS,  class Specialisation>
KOKKOS_INLINE_FUNCTION
F_FLOAT PairLJSPICACoulLongKokkos<DeviceType>::
compute_fcoul(const F_FLOAT& rsq, const int& /*i*/, const int&j,
              const int& /*itype*/, const int& /*jtype*/,
              const F_FLOAT& factor_coul, const F_FLOAT& qtmp) const {

/*

        r2inv = 1.0 / rsq;
        const int ljt = lj_type[itype][jtype];

        if (rsq < cut_coulsq) {
          if (!ncoultablebits || rsq <= tabinnersq) {
            r = sqrt(rsq);
            grij = g_ewald * r;
            expm2 = exp(-grij * grij);
            t = 1.0 / (1.0 + EWALD_P * grij);
            erfc = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5)))) * expm2;
            prefactor = qqrd2e * qtmp * q[j] / r;
            forcecoul = prefactor * (erfc + EWALD_F * grij * expm2);
            if (EFLAG) ecoul = prefactor * erfc;
            if (factor_coul < 1.0) {
              forcecoul -= (1.0 - factor_coul) * prefactor;
              if (EFLAG) ecoul -= (1.0 - factor_coul) * prefactor;
            }
          } else {
            union_int_float_t rsq_lookup;
            rsq_lookup.f = rsq;
            itable = rsq_lookup.i & ncoulmask;
            itable >>= ncoulshiftbits;
            fraction = (rsq_lookup.f - rtable[itable]) * drtable[itable];
            table = ftable[itable] + fraction * dftable[itable];
            forcecoul = qtmp * q[j] * table;
            if (EFLAG) ecoul = qtmp * q[j] * (etable[itable] + fraction * detable[itable]);
            if (factor_coul < 1.0) {
              table = ctable[itable] + fraction * dctable[itable];
              prefactor = qtmp * q[j] * table;
              forcecoul -= (1.0 - factor_coul) * prefactor;
              if (EFLAG) ecoul -= (1.0 - factor_coul) * prefactor;
            }
          }
        }
*/

  if (Specialisation::DoTable && rsq > tabinnersq) {
    union_int_float_t rsq_lookup;
    rsq_lookup.f = rsq;
    const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
    const F_FLOAT fraction = (rsq_lookup.f - d_rtable[itable]) * d_drtable[itable];
    const F_FLOAT table = d_ftable[itable] + fraction*d_dftable[itable];
    F_FLOAT forcecoul = qtmp*q[j] * table;
    if (factor_coul < 1.0) {
      const F_FLOAT table = d_ctable[itable] + fraction*d_dctable[itable];
      const F_FLOAT prefactor = qtmp*q[j] * table;
      forcecoul -= (1.0-factor_coul)*prefactor;
    }
    return forcecoul/rsq;
  } else {
    const F_FLOAT r = sqrt(rsq);
    const F_FLOAT grij = g_ewald * r;
    const F_FLOAT expm2 = exp(-grij*grij);
    const F_FLOAT t = 1.0 / (1.0 + EWALD_P*grij);
    const F_FLOAT rinv = 1.0/r;
    const F_FLOAT erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
    const F_FLOAT prefactor = qqrd2e * qtmp*q[j]*rinv;
    F_FLOAT forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
    if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor;

    return forcecoul*rinv*rinv;
  }
}

/* ----------------------------------------------------------------------
   compute coulomb pair potential energy between atoms i and j
------------------------------------------------------------------------- */

template<class DeviceType>
template<bool STACKPARAMS, class Specialisation>
KOKKOS_INLINE_FUNCTION
F_FLOAT PairLJSPICACoulLongKokkos<DeviceType>::
compute_ecoul(const F_FLOAT& rsq, const int& /*i*/, const int&j,
              const int& /*itype*/, const int& /*jtype*/, const F_FLOAT& factor_coul, const F_FLOAT& qtmp) const {
  if (Specialisation::DoTable && rsq > tabinnersq) {
    union_int_float_t rsq_lookup;
    rsq_lookup.f = rsq;
    const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
    const F_FLOAT fraction = (rsq_lookup.f - d_rtable[itable]) * d_drtable[itable];
    const F_FLOAT table = d_etable[itable] + fraction*d_detable[itable];
    F_FLOAT ecoul = qtmp*q[j] * table;
    if (factor_coul < 1.0) {
      const F_FLOAT table = d_ctable[itable] + fraction*d_dctable[itable];
      const F_FLOAT prefactor = qtmp*q[j] * table;
      ecoul -= (1.0-factor_coul)*prefactor;
    }
    return ecoul;
  } else {
    const F_FLOAT r = sqrt(rsq);
    const F_FLOAT grij = g_ewald * r;
    const F_FLOAT expm2 = exp(-grij*grij);
    const F_FLOAT t = 1.0 / (1.0 + EWALD_P*grij);
    const F_FLOAT erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
    const F_FLOAT prefactor = qqrd2e * qtmp*q[j]/r;
    F_FLOAT ecoul = prefactor * erfc;
    if (factor_coul < 1.0) ecoul -= (1.0-factor_coul)*prefactor;
    return ecoul;
  }
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

template<class DeviceType>
void PairLJSPICACoulLongKokkos<DeviceType>::allocate()
{
  PairLJSPICACoulLong::allocate();

  int n = atom->ntypes;

  memory->destroy(cutsq);
  memoryKK->create_kokkos(k_cutsq,cutsq,n+1,n+1,"pair:cutsq");
  d_cutsq = k_cutsq.template view<DeviceType>();

  d_cut_lj = typename AT::t_ffloat_2d("pair:cut_lj",n+1,n+1);
  d_cut_ljsq = typename AT::t_ffloat_2d("pair:cut_ljsq",n+1,n+1);
  d_cut_coulsq = typename AT::t_ffloat_2d("pair:cut_coulsq",n+1,n+1);

  k_params = Kokkos::DualView<params_lj_coul**,Kokkos::LayoutRight,DeviceType>("PairLJSPICACoulLong::params",n+1,n+1);
  params = k_params.template view<DeviceType>();
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

template<class DeviceType>
void PairLJSPICACoulLongKokkos<DeviceType>::settings(int narg, char **arg)
{
  if (narg > 2) error->all(FLERR,"Illegal pair_style command");

  PairLJSPICACoulLong::settings(1,arg);
}

/* ----------------------------------------------------------------------
   init tables
------------------------------------------------------------------------- */

template<class DeviceType>
void PairLJSPICACoulLongKokkos<DeviceType>::init_tables(double cut_coul, double *cut_respa)
{
  Pair::init_tables(cut_coul,cut_respa);

  typedef typename ArrayTypes<DeviceType>::t_ffloat_1d table_type;
  typedef typename ArrayTypes<LMPHostType>::t_ffloat_1d host_table_type;

  int ntable = 1;
  for (int i = 0; i < ncoultablebits; i++) ntable *= 2;


  // Copy rtable and drtable
  {
  host_table_type h_table("HostTable",ntable);
  table_type d_table("DeviceTable",ntable);
  for (int i = 0; i < ntable; i++) {
    h_table(i) = rtable[i];
  }
  Kokkos::deep_copy(d_table,h_table);
  d_rtable = d_table;
  }

  {
  host_table_type h_table("HostTable",ntable);
  table_type d_table("DeviceTable",ntable);
  for (int i = 0; i < ntable; i++) {
    h_table(i) = drtable[i];
  }
  Kokkos::deep_copy(d_table,h_table);
  d_drtable = d_table;
  }

  {
  host_table_type h_table("HostTable",ntable);
  table_type d_table("DeviceTable",ntable);

  // Copy ftable and dftable
  for (int i = 0; i < ntable; i++) {
    h_table(i) = ftable[i];
  }
  Kokkos::deep_copy(d_table,h_table);
  d_ftable = d_table;
  }

  {
  host_table_type h_table("HostTable",ntable);
  table_type d_table("DeviceTable",ntable);

  for (int i = 0; i < ntable; i++) {
    h_table(i) = dftable[i];
  }
  Kokkos::deep_copy(d_table,h_table);
  d_dftable = d_table;
  }

  {
  host_table_type h_table("HostTable",ntable);
  table_type d_table("DeviceTable",ntable);

  // Copy ctable and dctable
  for (int i = 0; i < ntable; i++) {
    h_table(i) = ctable[i];
  }
  Kokkos::deep_copy(d_table,h_table);
  d_ctable = d_table;
  }

  {
  host_table_type h_table("HostTable",ntable);
  table_type d_table("DeviceTable",ntable);

  for (int i = 0; i < ntable; i++) {
    h_table(i) = dctable[i];
  }
  Kokkos::deep_copy(d_table,h_table);
  d_dctable = d_table;
  }

  {
  host_table_type h_table("HostTable",ntable);
  table_type d_table("DeviceTable",ntable);

  // Copy etable and detable
  for (int i = 0; i < ntable; i++) {
    h_table(i) = etable[i];
  }
  Kokkos::deep_copy(d_table,h_table);
  d_etable = d_table;
  }

  {
  host_table_type h_table("HostTable",ntable);
  table_type d_table("DeviceTable",ntable);

  for (int i = 0; i < ntable; i++) {
    h_table(i) = detable[i];
  }
  Kokkos::deep_copy(d_table,h_table);
  d_detable = d_table;
  }
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType>
void PairLJSPICACoulLongKokkos<DeviceType>::init_style()
{
  PairLJSPICACoulLong::init_style();

  // error if rRESPA with inner levels

  if (update->whichflag == 1 && utils::strmatch(update->integrate_style,"^respa")) {
    int respa = 0;
    if (((Respa *) update->integrate)->level_inner >= 0) respa = 1;
    if (((Respa *) update->integrate)->level_middle >= 0) respa = 2;
    if (respa)
      error->all(FLERR,"Cannot use Kokkos pair style with rRESPA inner/middle");
  }

  // adjust neighbor list request for KOKKOS

  neighflag = lmp->kokkos->neighflag;
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                           !std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);
  if (neighflag == FULL) request->enable_full();
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

template<class DeviceType>
double PairLJSPICACoulLongKokkos<DeviceType>::init_one(int i, int j)
{
  double cutone = PairLJSPICACoulLong::init_one(i,j);

  k_params.h_view(i,j).lj1 = lj1[i][j];
  k_params.h_view(i,j).lj2 = lj2[i][j];
  k_params.h_view(i,j).lj3 = lj3[i][j];
  k_params.h_view(i,j).lj4 = lj4[i][j];
  k_params.h_view(i,j).offset = offset[i][j];
  k_params.h_view(i,j).cutsq = cutone*cutone;
  k_params.h_view(i,j).cut_lj = cut_lj[i][j];
  k_params.h_view(i,j).cut_ljsq = cut_ljsq[i][j];
  k_params.h_view(i,j).cut_coulsq = cut_coulsq;
  k_params.h_view(i,j).lj_type = lj_type[i][j];
  k_params.h_view(j,i) = k_params.h_view(i,j);
  if (i<MAX_TYPES_STACKPARAMS+1 && j<MAX_TYPES_STACKPARAMS+1) {
    m_params[i][j] = m_params[j][i] = k_params.h_view(i,j);
    m_cutsq[j][i] = m_cutsq[i][j] = cutone*cutone;
    m_cut_lj[j][i] = m_cut_lj[i][j] = cut_lj[i][j];
    m_cut_ljsq[j][i] = m_cut_ljsq[i][j] = cut_ljsq[i][j];
    m_cut_coulsq[j][i] = m_cut_coulsq[i][j] = cut_coulsq;
  }

  k_cutsq.h_view(i,j) = k_cutsq.h_view(j,i) = cutone*cutone;
  k_cutsq.template modify<LMPHostType>();
  k_params.template modify<LMPHostType>();

  return cutone;
}

namespace LAMMPS_NS {
template class PairLJSPICACoulLongKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairLJSPICACoulLongKokkos<LMPHostType>;
#endif
}

