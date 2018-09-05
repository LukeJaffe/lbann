////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

#include "lbann/objective_functions/weight_regularization/l2.hpp"
#include "lbann/models/model.hpp"
#ifdef LBANN_HAS_GPU
#include "lbann/utils/cublas.hpp"
#endif // LBANN_HAS_GPU

namespace {

  /** Compute the entry-wise sum of squares of a local matrix. */
  EvalType sum_of_squares(const Mat& mat) {
    const El::Int height = mat.Height();
    const El::Int width = mat.Width();
    const El::Int ldim = mat.LDim();
    const auto& __restrict__ buf = mat.LockedBuffer();
    EvalType sqsum = EvalType(0);
    if (ldim == height) {
      // Parallelize single loop if data is contiguous
      const El::Int size = height*width;
      #pragma omp parallel for reduction(+:sqsum)
      for (El::Int i = 0; i < size; ++i) {
        const EvalType val = buf[i];
        sqsum += val * val;
      }
    } else {
      // Parallelize double loop if data is not contiguous
      #pragma omp parallel for reduction(+:sqsum) collapse(2)
      for (El::Int j = 0; j < width; ++j) {
        for (El::Int i = 0; i < height; ++i) {
          const EvalType val = buf[i + j*ldim];
          sqsum += val * val;
        }
      }
    }
    return sqsum;
  }

} // namespace

namespace lbann {

l2_weight_regularization::l2_weight_regularization(EvalType scale_factor)
  : objective_function_term(scale_factor),
    m_sqsum(0),
    m_allreduce_started(false) {}

void l2_weight_regularization::setup(model& m) {
  objective_function_term::setup(m);

  // Check that term has no layer pointers
  if (!m_layers.empty()) {
    LBANN_ERROR("attempted to setup L2 weight regularization with layer pointers");
  }

  // Add all weights in model if no weights pointers are provided
  if (m_weights.empty()) {
    for (weights* w : m.get_weights()) {
      if (w->get_optimizer() != nullptr) {
        m_weights.push_back(w);
      }
    }
  }

}

void l2_weight_regularization::start_evaluation() {
  if (m_scale_factor == EvalType(0)) { return; }
  const int num_weights = m_weights.size();

  // Each weights' local contribution to L2 regularization term
  CPUMat sqsums;
  El::Zeros(sqsums, num_weights, 1);

#ifdef LBANN_HAS_GPU

  // Check whether any weights are on GPU
  bool using_gpus = false;
  for (const auto& w : m_weights) {
    if (w->get_values().GetLocalDevice() == El::Device::GPU) {
      using_gpus = true;
      break;
    }
  }

  // Compute L2 regularization term for weights on GPU
  // Note: cuBLAS is set to device pointer mode to pipeline GPU
  // kernels. Local contributions are only computed on one process in
  // each matrix's redundant communicator.
  if (using_gpus) {
    auto&& handle = El::GPUManager::cuBLASHandle();
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

    // Initialize workspace
    GPUMat sqsums_d;
#ifdef HYDROGEN_HAVE_CUB
    sqsums_d.SetMemoryMode(1); // CUB memory pool
#endif
    El::Zeros(sqsums_d, num_weights, 1);

    // Compute local contributions
    for (int i = 0; i < num_weights; ++i) {
      const auto& vals = m_weights[i]->get_values();
      if (vals.Participating()
          && vals.GetLocalDevice() == El::Device::GPU
          && vals.RedundantRank() == i % vals.RedundantSize()
          && vals.LocalWidth() > 0 && vals.LocalHeight() > 0) {
        if (vals.LocalWidth() == 1 || vals.LDim() == vals.LocalHeight()) {
          cublas::dot(handle,
                      vals.LocalHeight() * vals.LocalWidth(),
                      vals.LockedBuffer(), 1,
                      vals.LockedBuffer(), 1,
                      sqsums_d.Buffer(i, 0));
        } else {
          /// @todo Support non-contiguous data
          std::stringstream err;
          err << "weights \"" << m_weights[i]->get_name() << "\" "
              << "has a non-contiguous weight matrix "
              << "(local height = " << vals.LocalHeight() << ", "
              << "local width = " << vals.LocalWidth() << ", "
              << "leading dim = " << vals.LDim() << "), "
              << "but L2 regularization currently only supports "
              << "contiguous weight data";
          LBANN_ERROR(err.str());
        }
      }
    }

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    El::Copy(sqsums_d, sqsums);
  }

#endif // LBANN_HAS_GPU

  // Compute local contributions on CPU
  // Note: Only compute local contribution on one process in each
  // redundant communicator.
  m_sqsum = EvalType(0);
  for (int i = 0; i < num_weights; ++i) {
    const auto& vals = m_weights[i]->get_values();
    if (vals.Participating()
        && vals.GetLocalDevice() == El::Device::CPU
        && vals.RedundantRank() == i % vals.RedundantSize()) {
      sqsums(i, 0) = sum_of_squares(vals.LockedMatrix());      
    }
    m_sqsum += sqsums(i, 0);
  }

  // Start aggregating local contributions
  get_comm().nb_allreduce(&m_sqsum,
                          1,
                          get_comm().get_model_comm(),
                          m_allreduce_req);
  m_allreduce_started = true;

}

EvalType l2_weight_regularization::finish_evaluation() {
  if (m_scale_factor == EvalType(0)) { return EvalType(0); }
  if (m_allreduce_started) {
    get_comm().wait(m_allreduce_req);
  }
  m_allreduce_started = false;
  return m_scale_factor * m_sqsum / 2;
}

void l2_weight_regularization::compute_weight_regularization() {
  if (m_scale_factor == EvalType(0)) { return; }

  // Compute gradient of L2 regularization term for weights
  for (auto&& w : m_weights) {
    w->get_optimizer()->add_to_gradient(w->get_values(), m_scale_factor);
  }

}

} // namespace lbann
