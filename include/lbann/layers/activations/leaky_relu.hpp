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

#ifndef LEAKY_RELU_HPP_INCLUDED
#define LEAKY_RELU_HPP_INCLUDED

#include "lbann/layers/activations/activation.hpp"

namespace lbann {

/** Leaky rectified linear unit activation function.
 *  This is a ReLU variant that avoids the dying ReLU problem where a
 *  ReLU neuron can stop updating. See:
 *  Maas, Andrew L., Awni Y. Hannun, and Andrew Y. Ng. "Rectifier
 *  nonlinearities improve neural network acoustic models."
 *  Proc. ICML. Vol. 30. No. 1. 2013.
 */
template <data_layout T_layout, El::Device Dev>
class leaky_relu_layer : public entrywise_activation_layer {
 public:
  /** Leak is the amount of signal to permit for negative values. */
  leaky_relu_layer(lbann_comm *comm,
                   DataType leak = DataType(0.01))
    : entrywise_activation_layer(comm), m_leak(leak) {}
  leaky_relu_layer* copy() const override { return new leaky_relu_layer(*this); }
  std::string get_type() const override { return "leaky relu"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

 protected:
  DataType activation(DataType x) const override {
    return std::max(m_leak * x, x);
  }
  DataType activation_derivative(DataType x) const override {
    return (x > DataType(0)) ? DataType(1) : m_leak;
  }
 private:
  DataType m_leak;
};

} // namespace lbann

#endif // LEAKY_RELU_HPP_INCLUDED
