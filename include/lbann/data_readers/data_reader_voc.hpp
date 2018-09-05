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
//
// data_reader_voc .hpp .cpp - data reader class for ImageNet dataset
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_VOC_HPP
#define LBANN_DATA_READER_VOC_HPP

#include "data_reader_image.hpp"
#include "cv_process.hpp"

namespace lbann {
class data_reader_voc : public image_data_reader {
 public:
  data_reader_voc(bool shuffle) = delete;
  data_reader_voc(const std::shared_ptr<cv_process>& pp, bool shuffle = true);
  data_reader_voc(const data_reader_voc&);
  data_reader_voc& operator=(const data_reader_voc&);
  ~data_reader_voc() override;

  data_reader_voc* copy() const override { return new data_reader_voc(*this); }

  std::string get_type() const override {
    return "data_reader_voc";
  }

  virtual int get_num_responses() const {
    return 384208;
  }

  std::string get_image_dir() const {
    return "/p/lscratchf/brainusr/datasets/VOCdevkit/VOC2007/JPEGImages/";
  };

  /**
   * This parses the header of the CSV to determine column information.
   */
  void load() override;

 protected:
  void set_defaults() override;
  virtual bool replicate_processor(const cv_process& pp);
  virtual CPUMat create_datum_view(CPUMat& X, const int mb_idx) const;
  bool fetch_datum(CPUMat& X, int data_id, int mb_idx, int tid) override;
  /// Fetch the response associated with data_id.
  bool fetch_response(CPUMat& Y, int data_id, int mb_idx, int tid) override;
  bool fetch_label(CPUMat& Y, int data_id, int mb_idx, int tid) override;
  std::vector<std::string> m_image_list; ///< list of image files

  /// sets up a data_store.
  void setup_data_store(model *m) override;

  /// Initialize the ifstreams vector.
  void setup_ifstreams();

 protected:
  /// preprocessor duplicated for each omp thread
  std::vector<std::unique_ptr<cv_process> > m_pps;

  /// String value that separates data.
  char m_separator = ',';
  /// Number of columns (from the left) to skip.
  int m_skip_cols = 0;
  /// Number of rows to skip.
  int m_skip_rows = 0;
  /// Whether the CSV file has a header.
  bool m_has_header = false;
  /**
   * Column containing labels. -1 is used for the last column.
   * The label column ignores skipped columns, and can be among columns that are
   * skipped.
   */
  int m_label_col = -1;
  /// Column containing responses; functions the same as the label column.
  int m_response_col = -1;
  /// Whether to fetch labels.
  bool m_disable_labels = true;
  /// Whether to fetch responses.
  bool m_disable_responses = false;
  /// Number of columns (including the label column and skipped columns).
  int m_num_cols = 0;
  /// Number of samples.
  int m_num_samples = 0;
  /// Number of label classes.
  int m_num_labels = 0;
  /// Input file streams (per-thread).
  std::vector<std::ifstream*> m_ifstreams;
  /**
   * Index mapping lines (samples) to their start offset within the file.
   * This excludes the header, but includes a final entry indicating the length
   * of the file.
   */
  std::vector<std::streampos> m_index;
  /// Store labels.
  std::vector<int> m_labels;
  /// Store responses.
  std::vector<std::vector<std::vector<std::vector<float>>>> m_responses;
};

}  // namespace lbann

#endif  // LBANN_DATA_READER_VOC_HPP
