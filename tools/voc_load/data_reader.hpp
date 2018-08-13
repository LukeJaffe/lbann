#ifndef _DATA_READER_HPP_
#define _DATA_READER_HPP_

#include <string>
#include <vector>
#include "lbann/base.hpp"
#include "lbann/utils/exception.hpp"
#include <omp.h>

namespace lbann {

class generic_data_reader {
 public:
  generic_data_reader(bool) : m_master(true) {}
  generic_data_reader(const generic_data_reader&) = default;
  generic_data_reader& operator=(const generic_data_reader&) = default;
  virtual generic_data_reader* copy() const { return nullptr; }
  virtual ~generic_data_reader() {}

  virtual std::string get_type() const = 0;

  void set_data_filename(std::string s) { m_data_fn = s; }
  std::string get_data_filename() const { return m_data_fn; }
  std::string get_file_dir() const { return ""; }

  virtual void load() = 0;

  void set_absolute_sample_count(size_t s ) { m_absolute_sample_count = s; }
  size_t get_absolute_sample_count() const { return m_absolute_sample_count; }
  virtual int get_linearized_data_size() const = 0;
  virtual int get_linearized_response_size() const { return 0; };
  virtual int get_linearized_label_size() const = 0;
  virtual int get_num_labels() const = 0;
  virtual const std::vector<int> get_data_dims() const = 0;
  virtual void save_image(Mat& pixels, const std::string filename, bool do_scale = true) {};

  virtual void fetch_data(CPUMat& X, const size_t sid, const size_t mb_size);
  virtual void fetch_responses(CPUMat& Y, const size_t sid, const size_t mb_size);

 protected:
  virtual bool fetch_datum(CPUMat& X, int data_id, int mb_idx, int tid) = 0;
  virtual bool fetch_response(CPUMat& Y, int data_id, int mb_idx, int tid) = 0;
  virtual bool fetch_label(CPUMat& Y, int data_id, int mb_idx, int tid) = 0;
  int get_current_mini_batch_size() const { return 128; }

  std::string m_data_fn;
  bool m_master;
  size_t m_absolute_sample_count;
};

template<typename T>
inline void set_minibatch_item(CPUMat& M, const int mb_idx, const T* const ptr, const size_t count) {
  if ((count > 0u) && (ptr == nullptr)) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
                          " :: attempt to dereference a nullptr ");
  }
  for (size_t i = 0u; i < count; ++i) {
    M.Set(static_cast<El::Int>(i), static_cast<El::Int>(mb_idx), static_cast<DataType>(ptr[i]));
  }
}

inline void generic_data_reader::fetch_data(CPUMat& X, const size_t sid, const size_t mb_size) {
  #pragma omp parallel for
  for (size_t s = 0u; s < mb_size; s++) {
    fetch_datum(X, static_cast<int>(sid + s), static_cast<int>(s), omp_get_thread_num());
  }
}

inline void generic_data_reader::fetch_responses(CPUMat& X, const size_t sid, const size_t mb_size) {
  #pragma omp parallel for
  for (size_t s = 0u; s < mb_size; s++) {
    fetch_response(X, static_cast<int>(sid + s), static_cast<int>(s), omp_get_thread_num());
  }
}

} // namespace lbann

#endif // _DATA_READER_HPP_
