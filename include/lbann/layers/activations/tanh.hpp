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

#ifndef LBANN_LAYER_ACTIVATION_TANH_HPP_INCLUDED
#define LBANN_LAYER_ACTIVATION_TANH_HPP_INCLUDED

#include "lbann/layers/activations/activation.hpp"
#include "lbann/utils/cudnn.hpp"

namespace lbann {

/** Hyperbolic tangent. */
template <data_layout T_layout, El::Device Dev>
class tanh_layer : public entrywise_activation_layer {

 private:
#ifdef LBANN_HAS_CUDNN
  /** Activation cuDNN descriptor. */
  cudnnActivationDescriptor_t m_activation_cudnn_desc;
  /** Tensor cuDNN descriptors. */
  cudnn::entrywise_layer_tensor_manager m_tensors_cudnn_desc;
#endif // LBANN_HAS_CUDNN

 public:

  tanh_layer(lbann_comm *comm)
    : entrywise_activation_layer(comm)
#ifdef LBANN_HAS_CUDNN
    , m_activation_cudnn_desc(nullptr),
      m_tensors_cudnn_desc(this)
#endif // LBANN_HAS_CUDNN
  {}

  tanh_layer(const tanh_layer& other)
    : entrywise_activation_layer(other)
#ifdef LBANN_HAS_CUDNN
    , m_activation_cudnn_desc(nullptr),
      m_tensors_cudnn_desc(other.m_tensors_cudnn_desc)
#endif // LBANN_HAS_CUDNN
  {
#ifdef LBANN_HAS_CUDNN
    cudnn::copy_activation_desc(other.m_activation_cudnn_desc,
                                m_activation_cudnn_desc);
    m_tensors_cudnn_desc.set_layer(this);
#endif // LBANN_HAS_CUDNN
  }

  tanh_layer& operator=(const tanh_layer& other) {
    entrywise_activation_layer::operator=(other);
#ifdef LBANN_HAS_CUDNN
    cudnn::copy_activation_desc(other.m_activation_cudnn_desc,
                                m_activation_cudnn_desc);
    m_tensors_cudnn_desc = other.m_tensors_cudnn_desc;
    m_tensors_cudnn_desc.set_layer(this);
#endif // LBANN_HAS_CUDNN
    return *this;
  }

  ~tanh_layer() {
#ifdef LBANN_HAS_CUDNN
    if (m_activation_cudnn_desc != nullptr) {
      cudnnDestroyActivationDescriptor(m_activation_cudnn_desc);
    }
#endif // LBANN_HAS_CUDNN
  }

  tanh_layer* copy() const override { return new tanh_layer(*this); }
  std::string get_type() const override { return "tanh"; }

  /** Returns description of ctor params */
  std::string get_description() const override {
    return std::string {} +
     " tanh" + " dataLayout: " + this->get_data_layout_string(get_data_layout());
  }

  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  void setup_gpu() override {
    entrywise_activation_layer::setup_gpu();
#ifndef LBANN_HAS_CUDNN
    LBANN_ERROR("cuDNN not detected");
#else
    if (m_activation_cudnn_desc != nullptr) {
      CHECK_CUDNN(cudnnDestroyActivationDescriptor(m_activation_cudnn_desc));
      m_activation_cudnn_desc = nullptr;
    }
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&m_activation_cudnn_desc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(m_activation_cudnn_desc,
                                             CUDNN_ACTIVATION_TANH,
                                             CUDNN_PROPAGATE_NAN,
                                             0.0));
#endif // LBANN_HAS_CUDNN
  }

 protected:

  DataType activation(DataType x) const override {
    return std::tanh(x);
  }

  DataType activation_derivative(DataType x) const override {
    const auto& coshx = std::cosh(x);
    return 1 / (coshx * coshx);
  }

  void fp_compute() override;
  void bp_compute() override;

};

} // namespace lbann

#endif // LBANN_LAYER_ACTIVATION_TANH_HPP_INCLUDED
