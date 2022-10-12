/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_CUDA_LOCKS_HPP
#define KOKKOS_CUDA_LOCKS_HPP

#include <Kokkos_Macros.hpp>

#ifdef KOKKOS_ENABLE_CUDA

#include <cstdint>

#include <Cuda/Kokkos_Cuda_Error.hpp>

#ifdef KOKKOS_ENABLE_IMPL_DESUL_ATOMICS
#include <desul/atomics/Lock_Array_Cuda.hpp>
#endif

namespace Kokkos {
namespace Impl {

struct CudaLockArrays {
  std::int32_t* atomic;
  std::int32_t n;
};

/// \brief This global variable in Host space is the central definition
///        of these arrays.
extern CudaLockArrays g_host_cuda_lock_arrays;

/// \brief After this call, the g_host_cuda_lock_arrays variable has
///        valid, initialized arrays.
///
/// This call is idempotent.
void initialize_host_cuda_lock_arrays();

/// \brief After this call, the g_host_cuda_lock_arrays variable has
///        all null pointers, and all array memory has been freed.
///
/// This call is idempotent.
void finalize_host_cuda_lock_arrays();

}  // namespace Impl
}  // namespace Kokkos

namespace Kokkos {
namespace Impl {

/// \brief This global variable in CUDA space is what kernels use
///        to get access to the lock arrays.
///
/// When relocatable device code is enabled, there can be one single
/// instance of this global variable for the entire executable,
/// whose definition will be in Kokkos_Cuda_Locks.cpp (and whose declaration
/// here must then be extern.
/// This one instance will be initialized by initialize_host_cuda_lock_arrays
/// and need not be modified afterwards.
///
/// When relocatable device code is disabled, an instance of this variable
/// will be created in every translation unit that sees this header file
/// (we make this clear by marking it static, meaning no other translation
///  unit can link to it).
/// Since the Kokkos_Cuda_Locks.cpp translation unit cannot initialize the
/// instances in other translation units, we must update this CUDA global
/// variable based on the Host global variable prior to running any kernels
/// that will use it.
/// That is the purpose of the ensure_cuda_lock_arrays_on_device function.
__device__
#ifdef KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE
    __constant__ extern
#endif
    CudaLockArrays g_device_cuda_lock_arrays;

#define CUDA_SPACE_ATOMIC_MASK 0x1FFFF

/// \brief Acquire a lock for the address
///
/// This function tries to acquire the lock for the hash value derived
/// from the provided ptr. If the lock is successfully acquired the
/// function returns true. Otherwise it returns false.
__device__ inline bool lock_address_cuda_space(void* ptr) {
  size_t offset = size_t(ptr);
  offset        = offset >> 2;
  offset        = offset & CUDA_SPACE_ATOMIC_MASK;
  return (0 == atomicCAS(&g_device_cuda_lock_arrays.atomic[offset], 0, 1));
}

/// \brief Release lock for the address
///
/// This function releases the lock for the hash value derived
/// from the provided ptr. This function should only be called
/// after previously successfully acquiring a lock with
/// lock_address.
__device__ inline void unlock_address_cuda_space(void* ptr) {
  size_t offset = size_t(ptr);
  offset        = offset >> 2;
  offset        = offset & CUDA_SPACE_ATOMIC_MASK;
  atomicExch(&g_device_cuda_lock_arrays.atomic[offset], 0);
}

}  // namespace Impl
}  // namespace Kokkos

// Make lock_array_copied an explicit translation unit scope thingy
namespace Kokkos {
namespace Impl {
namespace {
static int lock_array_copied = 0;
inline int eliminate_warning_for_lock_array() { return lock_array_copied; }
}  // namespace

#ifdef KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE
inline
#else
static
#endif
    void
    copy_cuda_lock_arrays_to_device() {
  if (lock_array_copied == 0) {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_device_cuda_lock_arrays,
                                                  &g_host_cuda_lock_arrays,
                                                  sizeof(CudaLockArrays)));
  }
  lock_array_copied = 1;
}

#ifndef KOKKOS_ENABLE_IMPL_DESUL_ATOMICS

#ifdef KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE
inline void ensure_cuda_lock_arrays_on_device() {}
#else
inline static void ensure_cuda_lock_arrays_on_device() {
  copy_cuda_lock_arrays_to_device();
}
#endif

#else

#ifdef KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE
inline void ensure_cuda_lock_arrays_on_device() {}
#else
// Still Need COPY_CUDA_LOCK_ARRAYS for team scratch etc.
inline static void ensure_cuda_lock_arrays_on_device() {
  copy_cuda_lock_arrays_to_device();
  desul::ensure_cuda_lock_arrays_on_device();
}
#endif

#endif /* defined( KOKKOS_ENABLE_IMPL_DESUL_ATOMICS ) */

}  // namespace Impl
}  // namespace Kokkos

#endif /* defined( KOKKOS_ENABLE_CUDA ) */

#endif /* #ifndef KOKKOS_CUDA_LOCKS_HPP */
