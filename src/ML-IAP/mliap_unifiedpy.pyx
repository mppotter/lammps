# cython: language_level=3
# distutils: language = c++

import pickle
import numpy as np
import lammps.mliap

cimport cython
from cpython.ref cimport PyObject
from libc.stdlib cimport malloc, free


cdef extern from "../lammps.h" namespace "LAMMPS_NS":
    cdef cppclass LAMMPS:
        pass


cdef extern from "mliap_data.h" namespace "LAMMPS_NS":
    cdef cppclass MLIAPData:
        # ----- may not need -----
        int size_array_rows
        int size_array_cols
        int natoms
        int yoffset
        int zoffset
        int ndims_force
        int ndims_virial
        # -END- may not need -END-
        int size_gradforce
        # ----- write only -----
        double ** f
        double ** gradforce
        double ** betas         # betas for all atoms in list
        double ** descriptors   # descriptors for all atoms in list
        double * eatoms         # energies for all atoms in list
        double energy
        # -END- write only -END-
        int ndescriptors        # number of descriptors
        int nparams             # number of model parameters per element
        int nelements           # number of elements

        # data structures for grad-grad list (gamma)

        # ----- ignore for now -----
        int gamma_nnz           # number of non-zero entries in gamma
        double ** gamma         # gamma element
        int ** gamma_row_index  # row (parameter) index
        int ** gamma_col_index  # column (descriptor) index
        double * egradient      # energy gradient w.r.t. parameters
        # -END- ignore for now -END-

        # data structures for mliap neighbor list
        # only neighbors strictly inside descriptor cutoff

        int nlocal
        int nghost
        int ntotal
        int * elems
        int nlistatoms          # current number of atoms in neighborlist
        int * numneighs         # neighbors count for each atom
        int * iatoms            # index of each atom
        int * pair_i            # index of each i atom for each ij pair
        int * ielems            # element of each atom
        int nneigh_max          # number of ij neighbors allocated
        int npairs              # number of ij neighbor pairs
        int * jatoms            # index of each neighbor
        int * jelems            # element of each neighbor
        double ** rij           # distance vector of each neighbor
        # ----- write only -----
        double *** graddesc     # descriptor gradient w.r.t. each neighbor
        # -END- write only -END-
        int eflag               # indicates if energy is needed
        int vflag               # indicates if virial is needed


cdef extern from "mliap_unified.h" namespace "LAMMPS_NS":
    cdef cppclass MLIAPDummyDescriptor:
        MLIAPDummyDescriptor(PyObject *, LAMMPS *)
        int ndescriptors    # number of descriptors
        int nelements       # # of unique elements
        char **elements     # names of unique elements
        double cutmax       # maximum cutoff needed
        double rcutfac
        double *radelem     # element radii

        void compute_descriptors(MLIAPData *)
        void compute_forces(MLIAPData *)
        void set_elements(char **, int)

    cdef cppclass MLIAPDummyModel:
        MLIAPDummyModel(PyObject *, LAMMPS *, char * = NULL)
        int ndescriptors    # number of descriptors
        int nparams         # number of parameters per element
        int nelements;      # # of unique elements

        void compute_gradients(MLIAPData *)

    cdef void update_pair_energy(MLIAPData *, double *)
    cdef void update_pair_forces(MLIAPData *, double *)


# @property sans getter
def write_only_property(fset):
    return property(fget=None, fset=fset)


cdef class MLIAPDataPy:
    cdef MLIAPData * data

    def __cinit__(self):
        self.data = NULL
 
    def update_pair_energy(self, eij):
        cdef double[:] eij_arr = eij
        update_pair_energy(self.data, &eij_arr[0])
    
    def update_pair_forces(self, fij):
        cdef double[:, ::1] fij_arr = fij
        update_pair_forces(self.data, &fij_arr[0][0])

    @property
    def f(self):
        if self.data.f is NULL:
            return None
        return np.asarray(<double[:self.ntotal, :3]> &self.data.f[0][0])
    
    @property
    def size_gradforce(self):
        return self.data.size_gradforce
 
    @write_only_property
    def gradforce(self, value):
        if self.data.gradforce is NULL:
            raise ValueError("attempt to set NULL gradforce")
        cdef double[:, :] gradforce_view = <double[:self.ntotal, :self.size_gradforce]> &self.data.gradforce[0][0]
        cdef double[:, :] value_view = value
        gradforce_view[:] = value_view
 
    @write_only_property
    def betas(self, value):
        if self.data.betas is NULL:
            raise ValueError("attempt to set NULL betas")
        cdef double[:, :] betas_view = <double[:self.nlistatoms, :self.ndescriptors]> &self.data.betas[0][0]
        cdef double[:, :] value_view = value
        betas_view[:] = value_view

    @write_only_property
    def descriptors(self, value):
        if self.data.descriptors is NULL:
            raise ValueError("attempt to set NULL descriptors")
        cdef double[:, :] descriptors_view = <double[:self.nlistatoms, :self.ndescriptors]> &self.data.descriptors[0][0]
        cdef double[:, :] value_view = value
        descriptors_view[:] = value_view

    @write_only_property
    def eatoms(self, value):
        if self.data.eatoms is NULL:
            raise ValueError("attempt to set NULL eatoms")
        cdef double[:] eatoms_view = <double[:self.nlistatoms]> &self.data.eatoms[0]
        cdef double[:] value_view = value
        eatoms_view[:] = value_view

    @write_only_property
    def energy(self, value):
        self.data.energy = <double>value

    @property
    def ndescriptors(self):
        return self.data.ndescriptors

    @property
    def nparams(self):
        return self.data.nparams

    @property
    def nelements(self):
        return self.data.nelements

    # data structures for grad-grad list (gamma)

    @property
    def gamma_nnz(self):
        return self.data.gamma_nnz

    @property
    def gamma(self):
        if self.data.gamma is NULL:
            return None
        return np.asarray(<double[:self.nlistatoms, :self.gama_nnz]> &self.data.gamma[0][0])

    @property
    def gamma_row_index(self):
        if self.data.gamma_row_index is NULL:
            return None
        return np.asarray(<int[:self.nlistatoms, :self.gamma_nnz]> &self.data.gamma_row_index[0][0])

    @property
    def gamma_col_index(self):
        if self.data.gamma_col_index is NULL:
            return None
        return np.asarray(<int[:self.nlistatoms, :self.gamma_nnz]> &self.data.gamma_col_index[0][0])

    @property
    def egradient(self):
        if self.data.egradient is NULL:
            return None
        return np.asarray(<double[:self.nelements*self.nparams]> &self.data.egradient[0])

    # data structures for mliap neighbor list
    # only neighbors strictly inside descriptor cutoff

    @property
    def nlocal(self):
        return self.data.nlocal

    @property
    def nghost(self):
        return self.data.nghost

    @property
    def ntotal(self):
        return self.data.ntotal
    
    @property
    def elems(self):
        if self.data.elems is NULL:
            return None
        return np.asarray(<int[:self.ntotal]> &self.data.elems[0])

    @property
    def nlistatoms(self):
        return self.data.nlistatoms

    @property
    def numneighs(self):
        if self.data.numneighs is NULL:
            return None
        return np.asarray(<int[:self.nlistatoms]> &self.data.numneighs[0])

    @property
    def iatoms(self):
        if self.data.iatoms is NULL:
            return None
        return np.asarray(<int[:self.nlistatoms]> &self.data.iatoms[0])
    
    @property
    def pair_i(self):
        if self.data.pair_i is NULL:
            return None
        return np.asarray(<int[:self.npairs]> &self.data.pair_i[0])

    @property
    def ielems(self):
        if self.data.ielems is NULL:
            return None
        return np.asarray(<int[:self.nlistatoms]> &self.data.ielems[0])
    
    @property
    def npairs(self):
        return self.data.npairs

    @property
    def jatoms(self):
        if self.data.jatoms is NULL:
            return None
        return np.asarray(<int[:self.npairs]> &self.data.jatoms[0])
    
    @property
    def pair_j(self):
        return self.jatoms

    @property
    def jelems(self):
        if self.data.jelems is NULL:
            return None
        return np.asarray(<int[:self.npairs]> &self.data.jelems[0])

    @property
    def rij(self):
        if self.data.rij is NULL:
            return None
        return np.asarray(<double[:self.npairs, :3]> &self.data.rij[0][0])

    @write_only_property
    def graddesc(self, value):
        if self.data.graddesc is NULL:
            raise ValueError("attempt to set NULL graddesc")
        cdef double[:, :, :] graddesc_view = <double[:self.npairs, :self.ndescriptors, :3]> &self.data.graddesc[0][0][0]
        cdef double[:, :, :] value_view = value
        graddesc_view[:] = value_view

    @property
    def eflag(self):
        return self.data.eflag

    @property
    def vflag(self):
        return self.data.vflag


cdef class MLIAPUnifiedInterface:
    cdef MLIAPDummyModel * model
    cdef MLIAPDummyDescriptor * descriptor
    cdef unified_impl

    def __init__(self, unified_impl):
        self.model = NULL
        self.descriptor = NULL
        self.unified_impl = unified_impl
    
    def compute_gradients(self, data):
        self.unified_impl.compute_gradients(data)
    
    def compute_descriptors(self, data):
        self.unified_impl.compute_descriptors(data)
    
    def compute_forces(self, data):
        self.unified_impl.compute_forces(data)


cdef public void compute_gradients_python(unified_int, MLIAPData *data) with gil:
    pydata = MLIAPDataPy()
    pydata.data = data
    unified_int.compute_gradients(pydata)


cdef public void compute_descriptors_python(unified_int, MLIAPData *data) with gil:
    pydata = MLIAPDataPy()
    pydata.data = data
    unified_int.compute_descriptors(pydata)


cdef public void compute_forces_python(unified_int, MLIAPData *data) with gil:
    pydata = MLIAPDataPy()
    pydata.data = data
    unified_int.compute_forces(pydata)


cdef public object mliap_unified_connect(char *fname, MLIAPDummyModel * model,
                                         MLIAPDummyDescriptor * descriptor) with gil:
    str_fname = fname.decode('utf-8')
    with open(str_fname, 'rb') as pfile:
        unified = pickle.load(pfile)

    unified_int = MLIAPUnifiedInterface(unified)
    unified_int.model = model
    unified_int.descriptor = descriptor

    unified.interface = unified_int

    if unified.ndescriptors is None:
        raise ValueError("no descriptors set")

    unified_int.descriptor.ndescriptors = <int>unified.ndescriptors
    unified_int.descriptor.rcutfac = <double>unified.rcutfac
    unified_int.model.ndescriptors = <int>unified.ndescriptors
    unified_int.model.nparams = <int>unified.nparams

    if unified.element_types is None:
        raise ValueError("no element type set")
    
    cdef int nelements = <int>len(unified.element_types)
    cdef char **elements = <char**>malloc(nelements * sizeof(char*))
    if not elements:
        raise MemoryError("failed to allocate memory for element names")
    cdef char *elem_name
    for i, elem in enumerate(unified.element_types):
        elem_name_bytes = elem.encode('UTF-8')
        elem_name = elem_name_bytes
        elements[i] = &elem_name[0]
    unified_int.descriptor.set_elements(elements, nelements)
    unified_int.model.nelements = nelements

    free(elements)
    return unified_int
