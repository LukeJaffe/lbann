//#define HYDROGEN_IMPORTS_CUDA_HPP_
//#define HYDROGEN_IMPORTS_CUBLAS_HPP_
//#define EL_CORE_MEMORY_IMPL_HPP_

#include <iostream>
#include <iomanip>
#include <string>
#include <set>
#include <vector>
#include "data_reader_voc.hpp"
#include <chrono>

inline double get_time() {
  using namespace std::chrono;
  return duration_cast<duration<double>>(
           steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::cout << "Usage: > " << argv[0] << " data_file" << std::endl;
    return 0;
  }

  std::string data_file(argv[1]);

  using namespace lbann;
  using namespace std;

  std::cout << std::fixed;
  std::cout << std::setprecision(3);

  generic_data_reader *reader = nullptr;
  reader = new data_reader_voc(true);
  reader->set_data_filename(data_file);
  // set_absolute_sample_count() either here or in csv_reader::load()
  reader->load();

  return 0;
}
