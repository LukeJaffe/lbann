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

// SSD 300 standard parameters
const int NUM_DEFAULT = 8732;
const int NUM_COORD = 4;
const float IMAGE_SIZE = 300;
const int NUM_SCALES = 6;
const int NUM_CLASSES = 21;
const std::vector<int> FEATURE_MAPS = {38, 19, 10, 5, 3, 1};
const std::vector<int> MIN_SIZES = {30, 60, 111, 162, 213, 264};
const std::vector<int> MAX_SIZES = {60, 111, 162, 213, 264, 315};
const std::vector<float> STEPS = {8, 16, 32, 64, 100, 300};
const std::vector<std::vector<int>> ASPECT_RATIOS = {{2}, {2, 3}, {2, 3}, {2, 3}, {2}, {2}};

// Matching parameters
const float IOU_THRESH = 0.5;

#if 1
// Clamp all values in matrix to [0, 1]
inline std::vector<std::vector<float>> _clamp(std::vector<std::vector<float>> mat)
{
    for (uint i = 0; i < mat.size(); i++)
        for (uint j = 0; j < mat[i].size(); j++)
        {
            if (mat[i][j] < 0.0)
            {
                //std::cout << mat[i][j] << std::endl;
                mat[i][j] = 0.0;
            }
            else if (mat[i][j] > 1.0)
            {
                //std::cout << mat[i][j] << std::endl;
                mat[i][j] = 1.0;
            }
        }
    return mat;
}
#else
// Clamp all values in matrix to [0, 1]
inline std::vector<std::vector<float>> _clamp(std::vector<std::vector<float>> mat)
{
    return mat;
}
#endif

inline bool _cmpf(float a, float b, float eps = 0.005f)
{
        return (fabs(a - b) < eps);
}

// Compare 2 box matrices
inline bool _compare(std::vector<std::vector<float>> a, std::vector<std::vector<float>> b)
{
    int no_match = 0;
    for (uint i = 0; i < a.size(); i++)
        for (uint j = 0; j < a[i].size(); j++)
            if (!_cmpf(a[i][j], b[i][j]))
            {
                no_match++;
                //std::cout << a[i][j] << ", " << b[i][j] << std::endl;
            }
    std::cout << "Num no match: " << no_match << std::endl;
    return false;
}

// Convert default boxes from "center" form to "bound" form
inline std::vector<std::vector<float>> _convert_bound(std::vector<std::vector<float>> center_boxes)
{
    //std::cout << "center boxes size: " << center_boxes.size() << std::endl;
    // Initialize default box matrix
    std::vector<std::vector<float>> bound_boxes(center_boxes.size());
    for (uint i = 0; i < center_boxes.size(); i++)
           bound_boxes[i].resize(center_boxes[i].size());
    // Convert boxes from center to bound form one by one
    float cx, cy, w, h;
    float xmin, ymin, xmax, ymax;
    for (uint i = 0; i < center_boxes.size(); i++)
    {
        // Extra coordinates of current center box
        cx = center_boxes[i][0], cy = center_boxes[i][1], w = center_boxes[i][2], h = center_boxes[i][3];
        //std::cout << cx << ", " << cy << ", " << w << ", " << h << std::endl;
        // Do conversion
        xmin = cx - (w / 2);
        xmax = cx + (w / 2);
        ymin = cy - (h / 2);
        ymax = cy + (h / 2);
        // Store the box in bound form
        bound_boxes[i] = std::vector<float>{xmin, ymin, xmax, ymax};
    }
    return _clamp(bound_boxes);
}

// Convert default boxes from "bound" form to "center" form
inline std::vector<std::vector<float>> _convert_center(std::vector<std::vector<float>> bound_boxes)
{
    //std::cout << "bound boxes size: " << bound_boxes.size() << std::endl;
    // Initialize default box matrix
    std::vector<std::vector<float>> center_boxes(bound_boxes.size());
    for (uint i = 0; i < bound_boxes.size(); i++)
           center_boxes[i].resize(bound_boxes[i].size());
    // Convert boxes from center to bound form one by one
    float cx, cy, w, h;
    float xmin, ymin, xmax, ymax;
    for (uint i = 0; i < bound_boxes.size(); i++)
    {
        // Extra coordinates of current center box
        xmin = bound_boxes[i][0], ymin = bound_boxes[i][1], xmax = bound_boxes[i][2], ymax = bound_boxes[i][3];
        //std::cout << cx << ", " << cy << ", " << w << ", " << h << std::endl;
        // Do conversion
        cx = (xmin + xmax) / 2.0;
        cy = (ymin + ymax) / 2.0;
        w = xmax - xmin;
        h = ymax - ymin;
        // Store the box in bound form
        center_boxes[i] = std::vector<float>{cx, cy, w, h};
    }
    return _clamp(center_boxes);
}

// Compute default boxes in "center" (cx, cy, w, h) form
inline std::vector<std::vector<float>> _get_default_boxes(
    const float image_size, 
    const int num_scales,
    const std::vector<int> &feature_maps,
    const std::vector<int> &min_sizes,
    const std::vector<int> &max_sizes,
    const std::vector<float> &steps,
    const std::vector<std::vector<int>> &aspect_ratios,
    const int num_default,
    const int num_coord
    )
{
    // Declare variables
    int f;
    float f_k, cx, cy, s_k, s_k_prime, ar;
    // Initialize default box matrix
    std::vector<std::vector<float>> center_boxes(num_default);
    for (int i = 0; i < num_default; i++)
           center_boxes[i].resize(num_coord);
    // Populate default box matrix
    int t = 0;
    for (int k = 0; k < num_scales; k++)
    {
        f = feature_maps[k];
        for (int i = 0; i < f; i++)
        {
            for (int j = 0; j < f; j++)
            {
                f_k = image_size / steps[k];

                // unit center x,y
                cx = ((float)j + 0.5) / f_k;
                cy = ((float)i + 0.5) / f_k;
                //std::cout << cx << std::endl;

                // aspect ratio: 1
                // rel size: image_size
                s_k = min_sizes[k] / image_size;
                center_boxes[t++] = std::vector<float>{cx, cy, s_k, s_k};

                // aspect_ratio: 1
                // rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrtf(s_k * (max_sizes[k] / image_size));
                center_boxes[t++] = std::vector<float>{cx, cy, s_k_prime, s_k_prime};

                // rest of aspect ratios
                for (uint a = 0; a < aspect_ratios[k].size(); a++)
                {
                    ar = aspect_ratios[k][a];
                    center_boxes[t++] = std::vector<float>{cx, cy, s_k * sqrtf(ar), s_k / sqrtf(ar)};
                    center_boxes[t++] = std::vector<float>{cx, cy, s_k / sqrtf(ar), s_k * sqrtf(ar)};
                }
            }
        }
    }
    // Clamp default box matrix to [0, 1] if requested
    return _clamp(center_boxes);
}


// Take 1 boxes in (xmin, ymin, xmax, ymax) format, compute area
inline float _get_area(std::vector<float> a)
{
    float w, h, area;
    w = a[2] - a[0];
    h = a[3] - a[1];
    area = w*h;
    return area;
}

// Take 2 boxes in (xmin, ymin, xmax, ymax) format, compute intersection
inline float _get_inter(std::vector<float> a, std::vector<float> b)
{
    float lmx, gmx, lmy, gmy, xs, ys, i;
    lmx = std::min(a[2], b[2]);
    gmx = std::max(a[0], b[0]);
    lmy = std::min(a[3], b[3]);
    gmy = std::max(a[1], b[1]);
    xs = std::max(lmx - gmx, 0.0f);
    ys = std::max(lmy - gmy, 0.0f);
    i = xs * ys;
    return i;
}

// Take 2 boxes in (xmin, ymin, xmax, ymax) format, compute union
inline float _get_union(std::vector<float> a, std::vector<float> b, float i)
{
    float aa, ab, u;
    aa = _get_area(a);
    ab = _get_area(b);
    u = aa + ab - i;
    return u;
}

// Take 2 boxes in (xmin, ymin, xmax, ymax) format, compute iou
inline float _get_iou(std::vector<float> a, std::vector<float> b)
{
    float i, u, iou;
    i = _get_inter(a, b);
    u = _get_union(a, b, i);
    iou = i / u;
    return iou; 
}

// Take truth box, default box in (cx, cy, w, h) format, compute encoding for SSD
inline std::vector<float> _encode_offset(std::vector<float> truth_box, std::vector<float> default_box)
{
    float g_cx, g_cy, g_w, g_h;
    g_cx = (truth_box[0] - default_box[0]) / default_box[2];
    g_cy = (truth_box[1] - default_box[1]) / default_box[3];
    g_w = std::log(truth_box[2] / default_box[2]);
    g_h = std::log(truth_box[3] / default_box[3]);
    std::vector<float> offset{g_cx, g_cy, g_w, g_h};
    return offset;
}

template <typename T>
std::vector<size_t> sort_indices(const std::vector<T> &v) 
{
    // initialize original index locations
    std::vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

    return idx;
}

typedef struct ssd_truth
{
    std::vector<std::vector<uint8_t>> cls_vec;
    std::vector<std::vector<float>> off_vec;
} ssd_truth_t;

inline ssd_truth_t _match_image(
        std::vector<std::vector<float>> default_boxes, 
        std::vector<std::vector<float>> default_centers, 
        std::vector<std::vector<float>> truth_boxes,
        std::vector<std::vector<float>> truth_centers,
        std::vector<uint> truth_labels,
        float iou_thresh,
        uint num_classes)
{
    float iou;
    uint i, j, k = 0;
    // Declare, initialize variables
    std::vector<std::vector<size_t>> ind_mat(truth_boxes.size());
    for (i = 0; i < truth_boxes.size(); i++)
        ind_mat[i].resize(default_boxes.size());
    std::vector<float> iou_vec(truth_boxes.size()*default_boxes.size());
    // Get IOU for each truth/default box pair
    for (i = 0; i < truth_boxes.size(); i++)
    {
        for (j = 0; j < default_boxes.size(); j++)
        {
            iou = _get_iou(truth_boxes[i], default_boxes[j]);
            iou_vec[k++] = iou;
        }
    }
    // Sort IOU scores
    std::vector<size_t> idx_vec = sort_indices(iou_vec);
    // Setup vectors to keep track of matching
    std::vector<uint8_t> default_mark_vec(default_boxes.size());
    std::vector<uint8_t> truth_mark_vec(truth_boxes.size());
    uint truth_mark_sum = 0;
    // Surjective matching algorithm
    for (auto idx: idx_vec)
    {
        iou = iou_vec[idx];
        i = idx / default_boxes.size();
        j = idx % default_boxes.size();
        // If no truth box has been assigned to this default box
        if (default_mark_vec[i] == 0)
        {
            // If the iou is >= than the required thresh
            if (iou >= iou_thresh)
            {
                ind_mat[i][j] = 1;
                default_mark_vec[j] = 1;
                truth_mark_vec[i] = 1;
                ++truth_mark_sum;
            }
            // OR if iou < than required thresh, but no default box has been assigned to this truth box
            else if (truth_mark_vec[i] == 0)
            {
                ind_mat[i][j] = 1;
                default_mark_vec[j] = 1;
                truth_mark_vec[i] = 1;
                ++truth_mark_sum;
            }
        }
        // Finish condition: each truth box has been assigned to a default box
        if (truth_mark_sum == truth_mark_vec.size())
            break;
    }
    // TODO: Add tests for ind_mat to make sure it meets required conditions
    // Create vector of class labels
    std::vector<std::vector<uint8_t>> cls_vec(default_boxes.size());
    for (i = 0; i < default_boxes.size(); i++)
        cls_vec[i].resize(num_classes);
    for (i = 0; i < ind_mat.size(); i++)
        for (j = 0; j < ind_mat.size(); j++)
            if (ind_mat[i][j])
                cls_vec[j][truth_labels[i]] = 1;
    // Create vector of offset values
    std::vector<std::vector<float>> off_vec(default_boxes.size());
    for (i = 0; i < off_vec.size(); i++)
        off_vec[i].resize(default_boxes[i].size());
    for (i = 0; i < ind_mat.size(); i++)
        for (j = 0; j < ind_mat.size(); j++)
            if (ind_mat[i][j])
                off_vec[j] = _encode_offset(truth_centers[i], default_centers[j]);
    // Create result object
    ssd_truth_t ssd_truth = 
    {
        cls_vec,
        off_vec
    };
    return ssd_truth;
}

void test_convert()
{
  std::cout << "==> Test 1:" << std::endl;
  std::vector<std::vector<float>> default_center_boxes = _get_default_boxes(IMAGE_SIZE, NUM_SCALES, FEATURE_MAPS, MIN_SIZES, MAX_SIZES, STEPS, ASPECT_RATIOS, NUM_DEFAULT, NUM_COORD);
  std::vector<std::vector<float>> default_bound_boxes = _convert_bound(default_center_boxes);
  std::vector<std::vector<float>> default_center_boxes2 = _convert_center(default_bound_boxes);
  std::vector<std::vector<float>> default_bound_boxes2 = _convert_bound(default_center_boxes2);
  _compare(default_center_boxes, default_center_boxes2);
  _compare(default_bound_boxes, default_bound_boxes2);
}

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
  std::vector<std::vector<std::vector<float>>> all_boxes;
  std::vector<std::vector<uint>> all_labels;

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

    // Store boxes
    std::vector<std::vector<float>> img_boxes;
    std::vector<uint> img_labels;
    std::vector<float> box;

    std::string prev_file_name;
    std::string curr_file_name;
    bool first = true;

    while (std::getline(ifs, line)) {
      if (line_num == num_samples_to_use) {
        break;
      }
      ++line_num;
      //std::cout << "\n==> Data element " << line_num << ":" << std::endl;
      // Verify the line has the right number of columns.
      if (std::count(line.begin(), line.end(), m_separator) + 1 != m_num_cols) {
        throw lbann_exception(
          "csv_reader: line " + std::to_string(line_num) +
          " does not have right number of entries");
      }
      index.push_back(ifs.tellg());

        // Clear the box vector
        box.clear();

        size_t cur_pos = 0;
        int cls_id=0, difficult=0;
        float w=0, h=0, xmin, ymin, xmax, ymax;
        float sxmin, symin, sxmax, symax;
        for (int col = 0; col < m_num_cols; ++col) {
          size_t end_pos = line.find_first_of(m_separator, cur_pos);
          switch (col) {
              case file_name_idx:
                curr_file_name = line.substr(cur_pos, end_pos - cur_pos);
                //std::cout << "file name: " << curr_file_name << std::endl;
                break;
                //label_classes.insert(label);
                //m_labels.push_back(label);
              case cls_id_idx:
                cls_id = stoi(line.substr(cur_pos, end_pos - cur_pos));
                //std::cout << "class id: " << cls_id << std::endl;
                break;
              case difficult_idx:
                difficult = stoi(line.substr(cur_pos, end_pos - cur_pos));
                //std::cout << "difficult: " << difficult << std::endl;
                break;
              case w_idx:
                w = stof(line.substr(cur_pos, end_pos - cur_pos));
                //std::cout << "width: " << w << std::endl;
                break;
              case h_idx:  
                h = stof(line.substr(cur_pos, end_pos - cur_pos));
                //std::cout << "height: " << h << std::endl;
                break;
              case xmin_idx:
                xmin = stof(line.substr(cur_pos, end_pos - cur_pos));
                sxmin = xmin / w;
                box.push_back(sxmin);
                //std::cout << "sxmin: " << sxmin << std::endl;
                break;
              case ymin_idx:
                ymin = stof(line.substr(cur_pos, end_pos - cur_pos));
                symin = ymin / h;
                box.push_back(symin);
                //std::cout << "symin: " << symin << std::endl;
                break;
              case xmax_idx:
                xmax = stof(line.substr(cur_pos, end_pos - cur_pos));
                sxmax = xmax / w;
                box.push_back(sxmax);
                //std::cout << "sxmax: " << sxmax << std::endl;
                break;
              case ymax_idx:
                ymax = stof(line.substr(cur_pos, end_pos - cur_pos));
                symax = ymax / h;
                box.push_back(symax);
                //std::cout << "symax: " << symax << std::endl;
                break;
              default:
                ;
          }
          cur_pos = end_pos + 1;
        }


        if (first)
        {
            prev_file_name = curr_file_name;
            first = false;
        }
        else if (curr_file_name != prev_file_name)
        {
            prev_file_name = curr_file_name;
            all_boxes.push_back(img_boxes);
            all_labels.push_back(img_labels);
            img_boxes.clear();
            img_labels.clear();
        }
        img_boxes.push_back(box);
        img_labels.push_back(cls_id);
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
    // Make sure to get the last set of boxes
    all_boxes.push_back(img_boxes);

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
  /*
  for (uint i = 0; i < all_boxes.size(); i++)
  {
        std::cout << "NEW IMAGE" << std::endl;
        for (uint j = 0; j < all_boxes[i].size(); j++)
        {
            for (uint k = 0; k < all_boxes[i][j].size(); k++)
            {
                std::cout << all_boxes[i][j][k] << ", ";
            }
            std::cout << std::endl;
        }
  }
  */
  // Reset indices.
  //m_shuffled_indices.resize(m_num_samples);
  //std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  //select_subset_of_data();
  std::cout << "==> Generating default boxes..." << std::endl;
  std::vector<std::vector<float>> default_centers = _get_default_boxes(IMAGE_SIZE, NUM_SCALES, FEATURE_MAPS, MIN_SIZES, MAX_SIZES, STEPS, ASPECT_RATIOS, NUM_DEFAULT, NUM_COORD);
  std::vector<std::vector<float>> default_bounds = _convert_bound(default_centers);

  // Convert ground truth boxes to center format
  std::vector<std::vector<float>> truth_centers;

  std::cout << "==> Matching..." << std::endl;
  ssd_truth_t ssd_truth;
  std::vector<std::vector<std::vector<uint8_t>>> full_cls_vec(all_boxes.size());
  std::vector<std::vector<std::vector<float>>> full_off_vec(all_boxes.size());
  for (uint i = 0; i < all_boxes.size(); i++)
  {
      std::cout << i << " / " << all_boxes.size() << std::endl;
      truth_centers = _convert_center(all_boxes[i]);
      ssd_truth = _match_image(default_bounds, default_centers, all_boxes[i], truth_centers, all_labels[i], IOU_THRESH, NUM_CLASSES);
      full_cls_vec[i] = ssd_truth.cls_vec;
      full_off_vec[i] = ssd_truth.off_vec;
  }
}

}  // namespace lbann

