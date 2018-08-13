/*
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
// lbann_data_reader_voc_data_reader class for Pascal VOC 2007 dataset
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_voc.hpp"
#include <cstdio>
#include <string>
#include <omp.h>

namespace lbann {

data_reader_voc::data_reader_voc(bool shuffle)
  : csv_reader(shuffle) {
  set_response_col(1);
  enable_responses();
  set_label_col(0);
  set_separator(',');
  // First five columns are metadata, not the sample.
  set_skip_cols(0);
  // Header is broken, so skip it.
  set_skip_rows(0);
  set_has_header(false);
  // Transform to binary classification.
  set_label_transform(
    [] (const std::string& s) -> int {
      return stoi(s);
    });
}

data_reader_voc::data_reader_voc()
  : data_reader_voc(true) {}

}  // namespace lbann
*/

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
// lbann_data_reader_voc_data_reader class for Pascal VOC 2007 dataset
////////////////////////////////////////////////////////////////////////////////

#include "data_reader_voc.hpp"
#include <unordered_set>
#include <cstdio>
#include <string>
#include <omp.h>

namespace lbann {

enum voc_idx_t {
    file_name_idx,
    cls_id_idx,
    difficult_idx,
    w_idx,
    h_idx,
    xmin_idx,
    ymin_idx,
    xmax_idx,
    ymax_idx,
};

data_reader_voc::data_reader_voc(bool shuffle)
  : csv_reader(shuffle) {
  //set_response_col(2);
  enable_responses();
  //set_label_col(3);
  // First five columns are metadata, not the sample.
  set_skip_cols(0);
  // Header is broken, so skip it.
  set_skip_rows(0);
  set_has_header(false);
  set_separator(',');
  // Transform to binary classification.
  set_label_transform(
    [] (const std::string& s) -> int {
      return stoi(s);
    });
}

data_reader_voc::data_reader_voc()
  : data_reader_voc(true) {}

void data_reader_voc::load() {
  bool master = true;//m_comm->am_world_master();
  setup_ifstreams();
  std::ifstream& ifs = *m_ifstreams[0];
  //const El::mpi::Comm world_comm = m_comm->get_world_comm();
  // Parse the header to determine how many columns there are.
  // Skip rows if needed.
  if (master) {
    skip_rows(ifs, m_skip_rows);
  }
  //m_comm->broadcast<int>(0, m_skip_rows, world_comm);

  //This will be broadcast from root to other procs, and will
  //then be converted to std::vector<int> m_labels; this is because
  //El::mpi::Broadcast<std::streampos> doesn't work
  std::vector<long long> index;

  if (master) {
    std::string line;
    std::streampos header_start = ifs.tellg();
    // TODO: Skip comment lines.
    if (std::getline(ifs, line)) {
      m_num_cols = std::count(line.begin(), line.end(), m_separator) + 1;
    } else {
      throw lbann_exception(
        "csv_reader: failed to read header in " + get_data_filename());
    }
    if (ifs.eof()) {
      throw lbann_exception(
        "csv_reader: reached EOF after reading header");
    }
    // If there was no header, skip back to the beginning.
    if (!m_has_header) {
      ifs.clear();
      ifs.seekg(header_start, std::ios::beg);
    }
    // Construct an index mapping each line (sample) to its offset.
    // TODO: Skip comment lines.
    // Used to count the number of label classes.
    std::unordered_set<int> label_classes;
    index.push_back(ifs.tellg());

//////////////////////////////////////////////
// make sure to set_absolute_sample_count()
/////////////////////////////////////////////
    int num_samples_to_use = get_absolute_sample_count();
    int line_num = 0;
    if (num_samples_to_use == 0) {
      num_samples_to_use = -1;
    }
    while (std::getline(ifs, line)) {
      if (line_num == num_samples_to_use) {
        break;
      }
      ++line_num;
      std::cout << "\n==> Data element " << line_num << ":" << std::endl;
      // Verify the line has the right number of columns.
      if (std::count(line.begin(), line.end(), m_separator) + 1 != m_num_cols) {
        throw lbann_exception(
          "csv_reader: line " + std::to_string(line_num) +
          " does not have right number of entries");
      }
      index.push_back(ifs.tellg());

        size_t cur_pos = 0;
        std::string file_name;
        int cls_id, difficult;
        float w, h, xmin, ymin, xmax, ymax;
        float sxmin, symin, sxmax, symax;
        for (int col = 0; col < m_num_cols; ++col) {
          size_t end_pos = line.find_first_of(m_separator, cur_pos);
          switch (col) {
              case file_name_idx:
                file_name = line.substr(cur_pos, end_pos - cur_pos);
                std::cout << "file name: " << file_name << std::endl;
                break;
                //label_classes.insert(label);
                //m_labels.push_back(label);
              case cls_id_idx:
                cls_id = stoi(line.substr(cur_pos, end_pos - cur_pos));
                std::cout << "class id: " << cls_id << std::endl;
                break;
              case difficult_idx:
                difficult = stoi(line.substr(cur_pos, end_pos - cur_pos));
                std::cout << "difficult: " << difficult << std::endl;
                break;
              case w_idx:
                w = stof(line.substr(cur_pos, end_pos - cur_pos));
                std::cout << "width: " << w << std::endl;
                break;
              case h_idx:  
                h = stof(line.substr(cur_pos, end_pos - cur_pos));
                std::cout << "height: " << h << std::endl;
                break;
              case xmin_idx:
                xmin = stof(line.substr(cur_pos, end_pos - cur_pos));
                sxmin = xmin / w;
                std::cout << "sxmin: " << sxmin << std::endl;
                break;
              case ymin_idx:
                ymin = stof(line.substr(cur_pos, end_pos - cur_pos));
                symin = ymin / h;
                std::cout << "symin: " << symin << std::endl;
                break;
              case xmax_idx:
                xmax = stof(line.substr(cur_pos, end_pos - cur_pos));
                sxmax = xmax / w;
                std::cout << "sxmax: " << sxmax << std::endl;
                break;
              case ymax_idx:
                ymax = stof(line.substr(cur_pos, end_pos - cur_pos));
                symax = ymax / h;
                std::cout << "symax: " << symax << std::endl;
                break;
              default:
                ;
          }
          cur_pos = end_pos + 1;
        }
      /*
      // Extract the label.
      if (!m_disable_labels) {
        size_t cur_pos = 0;
        for (int col = 0; col < m_num_cols; ++col) {
          size_t end_pos = line.find_first_of(m_separator, cur_pos);
          if (col == m_label_col) {
            int label = m_label_transform(line.substr(cur_pos, end_pos - cur_pos));
            std::cout << "label:" << label << std::endl;
            label_classes.insert(label);
            m_labels.push_back(label);
            break;
          }
          cur_pos = end_pos + 1;
        }
      }
      // Possibly extract the response.
      if (!m_disable_responses) {
        size_t cur_pos = 0;
        for (int col = 0; col < m_num_cols; ++col) {
          size_t end_pos = line.find_first_of(m_separator, cur_pos);
          if (col == m_response_col) {
            DataType response = m_response_transform(
              line.substr(cur_pos, end_pos - cur_pos));
            std::cout << "response:" << response << std::endl;
            m_responses.push_back(response);
            break;
          }
          cur_pos = end_pos + 1;
        }
      }
      */
    }

    if (!ifs.eof() && num_samples_to_use == 0) {
       //If we didn't get to EOF, something went wrong.
      throw lbann_exception(
        "csv_reader: did not reach EOF");
    }
    /*
    if (!m_disable_labels) {
      // Do some simple validation checks on the classes.
      // Ensure the elements begin with 0, and there are no gaps.
      auto minmax = std::minmax_element(label_classes.begin(), label_classes.end());
      if (*minmax.first != 0) {
        throw lbann_exception(
          "csv_reader: classes are not indexed from 0");
      }
      if (*minmax.second != (int) label_classes.size() - 1) {
        throw lbann_exception(
          "csv_reader: label classes are not contiguous");
      }
      m_num_labels = label_classes.size();
    }
    */
    ifs.clear();
  } // if (master)

  //m_comm->broadcast<int>(0, m_num_cols, world_comm);
  m_label_col = m_num_cols - 1;

  //bcast the index vector
  //m_comm->world_broadcast<long long>(0, index);
  m_num_samples = index.size() - 1;
  if (m_master) std::cerr << "num samples: " << m_num_samples << "\n";

  m_index.reserve(index.size());
  for (auto t : index) {
    m_index.push_back(t);
  }

  //optionally bcast the response vector
  if (!m_disable_responses) {
    m_response_col = m_num_cols - 1;
    //m_comm->world_broadcast<DataType>(0, m_responses);
  }

  //optionally bcast the label vector
  if (!m_disable_labels) {
    //m_comm->world_broadcast<int>(0, m_labels);
    m_num_labels = m_labels.size();
  }
  std::cout << "==> Done reading data!" << std::endl;
  // Reset indices.
  //m_shuffled_indices.resize(m_num_samples);
  //std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  //select_subset_of_data();
}

}  // namespace lbann

