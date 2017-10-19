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
// lbann_image_utils .cpp .hpp - Image I/O utility functions
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/image_utils.hpp"
#include <stdlib.h>
#include <stdio.h>

#ifdef __LIB_OPENCV
using namespace cv;
#endif


#define BMP_HEADER_MARKER   ((unsigned short) ('M' << 8) | 'B')

#pragma pack(push)
#pragma pack(2)
typedef struct __BMP_FILEHEADER {
  unsigned short bfType;
  unsigned long  bfSize;
  unsigned short bfReserved1;
  unsigned short bfReserved2;
  unsigned long  bfOffBits;

} BMP_FILEHEADER;

typedef struct __BMP_INFOHEADER {
  unsigned long  biSize;
  long           biWidth;
  long           biHeight;
  unsigned short biPlanes;
  unsigned short biBitCount;
  unsigned long  biCompression;
  unsigned long  biSizeImage;
  long           biXPelsPerMeter;
  long           biYPelsPerMeter;
  unsigned long  biClrUsed;
  unsigned long  biClrImportant;

} BMP_INFOHEADER;

typedef struct __BMP_RGBQUAD {
  unsigned char rgbBlue;
  unsigned char rgbGreen;
  unsigned char rgbRed;
  unsigned char rgbReserved;

} BMP_RGBQUAD;

typedef struct __BMP_INFO {
  BMP_INFOHEADER bmiHeader;
  BMP_RGBQUAD    bmiColors[1];

} BMP_INFO;
#pragma pack(pop)


bool lbann::image_utils::loadBMP(const std::string& Imagefile, int& Width, int& Height, int& BPP, bool Flip, unsigned char *&Pixels) {
  FILE *infile = fopen(Imagefile.c_str(), "rb");
  if (infile == NULL) {
    fprintf(stderr, "can't open %s\n", Imagefile.c_str());
    return false;
  }

  // Read Bitmap File Header
  BMP_FILEHEADER header;
  fread(&header, sizeof(BMP_FILEHEADER), 1, infile);
  if (header.bfType != BMP_HEADER_MARKER) {
    fclose(infile);
    return false;
  }

  // Read Bitmap Info
  int bisize = header.bfOffBits - sizeof(BMP_FILEHEADER);
  BMP_INFO *info = (BMP_INFO *)malloc(bisize);
  fread(info, bisize, 1, infile);

  // Check Palette Count
  if (info->bmiHeader.biClrUsed != 0 || info->bmiHeader.biBitCount != 24) {
    free(info);
    fclose(infile);
    return false;
  }

  // Read DIB Bits
  int bitrowsize = ((info->bmiHeader.biWidth * info->bmiHeader.biBitCount + 31) / 32) * 4;
  int bitsize = bitrowsize * info->bmiHeader.biHeight;
  unsigned char *bits = (unsigned char *)malloc(bitsize);
  fread(bits, bitsize, 1, infile);

  // Set Pixels
  Width = info->bmiHeader.biWidth;
  Height = info->bmiHeader.biHeight;
  BPP = 3;
  Pixels = new unsigned char[Width * Height * BPP];
  for (int y = 0; y < Height; y++) {
    for (int x = 0; x < Width; x++) {
      int offset = (Flip) ? ((Height - 1 - y) * Width + x) : (y * Width + x);
      Pixels[offset] = bits[y * bitrowsize + x * 3 + 2];
      Pixels[offset + Height*Width] = bits[y * bitrowsize + x * 3 + 1];
      Pixels[offset + 2*Height*Width] = bits[y * bitrowsize + x * 3 + 0];
    }
  }

  free(info);
  free(bits);
  fclose(infile);

  return true;
}

bool lbann::image_utils::saveBMP(const std::string& Imagefile, int Width, int Height, int BPP, bool Flip, unsigned char *Pixels) {
  if (BPP != 3) {
    return false;
  }



  return false;
}

bool lbann::image_utils::loadPGM(const std::string& Imagefile, int& Width, int& Height, int& BPP, bool Flip, unsigned char *&Pixels) {
  FILE *infile = fopen(Imagefile.c_str(), "rb");
  if (infile == NULL) {
    fprintf(stderr, "can't open %s\n", Imagefile.c_str());
    return false;
  }

  char format[5];
  fscanf(infile, "%s", format);
  int width, height;
  fscanf(infile, "%d%d", &width, &height);
  int maxpixel;
  fscanf(infile, "%d", &maxpixel);

  Width = width;
  Height = height;
  BPP = 1;
  Pixels = new unsigned char[Width * Height * BPP];

  for (int y = 0; y < Height; y++) {
    for (int x = 0; x < Width; x++) {
      int offset = (Flip) ? ((Height - 1 - y) * Width + x) : (y * Width + x);
      int pixel = fgetc(infile);
      Pixels[offset] = (unsigned char)((double)pixel / maxpixel * 255.0);
    }
  }

  fclose(infile);
  return true;
}

bool lbann::image_utils::savePGM(const std::string& Imagefile, int Width, int Height, int BPP, bool Flip, unsigned char *Pixels) {
  if (BPP != 1) {
    return false;
  }

  FILE *outfile = fopen(Imagefile.c_str(), "wb");
  if (outfile == NULL) {
    fprintf(stderr, "can't create %s\n", Imagefile.c_str());
    return false;
  }

  fprintf(outfile, "P5\n");
  fprintf(outfile, "%d %d\n", Width, Height);
  fprintf(outfile, "255\n");

  for (int y = 0; y < Height; y++) {
    for (int x = 0; x < Width; x++) {
      int offset = (Flip) ? ((Height - 1 - y) * Width + x) : (y * Width + x);
      fputc(Pixels[offset], outfile);
    }
  }

  fclose(outfile);
  return true;
}

bool lbann::image_utils::loadPNG(const std::string& Imagefile, int& Width, int& Height, bool Flip, uchar *&Pixels) {
#ifdef __LIB_OPENCV
  cv::Mat image = cv::imread(Imagefile, _LBANN_CV_COLOR_);
  if (image.empty()) {
    return false;
  }

  Width = image.cols;
  Height = image.rows;

  for (int y = 0; y < Height; y++) {
    for (int x = 0; x < Width; x++) {
      cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
      int offset = (Flip) ? ((Height - 1 - y) * Width + x) : (y * Width + x);
      Pixels[offset]                  = pixel[_LBANN_CV_BLUE_];
      Pixels[offset + Height*Width]   = pixel[_LBANN_CV_GREEN_];
      Pixels[offset + 2*Height*Width] = pixel[_LBANN_CV_RED_];
    }
  }

  return true;
#else
  return false;
#endif
}

bool lbann::image_utils::savePNG(const std::string& Imagefile, int Width, int Height, bool Flip, uchar *Pixels) {
#ifdef __LIB_OPENCV
  cv::Mat image = cv::Mat(Height, Width, CV_8UC3);

  for (int y = 0; y < Height; y++) {
    for (int x = 0; x < Width; x++) {
      cv::Vec3b pixel;
      int offset = (Flip) ? ((Height - 1 - y) * Width + x) : (y * Width + x);
      pixel[_LBANN_CV_BLUE_] = Pixels[offset];
      pixel[_LBANN_CV_GREEN_] = Pixels[offset + Height*Width];
      pixel[_LBANN_CV_RED_] = Pixels[offset + 2*Height*Width];
      image.at<cv::Vec3b>(y, x) = pixel;
    }
  }
  cv::imwrite(Imagefile, image);

  return true;
#else
  return false;
#endif
}

bool lbann::image_utils::loadJPG(const std::string& Imagefile, int& Width, int& Height, unsigned char *&Pixels, bool random_crop) {

#ifdef __LIB_OPENCV

  // Load image from file
  cv::Mat raw_image = cv::imread(Imagefile, _LBANN_CV_COLOR_);
  if (raw_image.empty()) {
    return false;
  }
  int raw_width = raw_image.cols;
  int raw_height = raw_image.rows;

  // First resize such that the shorter dimension is 256x256, then crop the
  // central 256x256 pixels.
  if (raw_width <= raw_height) {
    double scale = 256.0 / raw_width;
    cv::Mat scaled_image;
    cv::resize(raw_image, scaled_image, cv::Size(), scale, scale);
    cv::Rect crop(0, (scaled_image.rows - 256 + 1) / 2, 256, 256);
    raw_image = cv::Mat(scaled_image, crop);
  } else {
    double scale = 256.0 / raw_height;
    cv::Mat scaled_image;
    cv::resize(raw_image, scaled_image, cv::Size(), scale, scale);
    cv::Rect crop((scaled_image.cols - 256 + 1) / 2, 0, 256, 256);
    raw_image = cv::Mat(scaled_image, crop);
  }
  raw_width = 256;
  raw_height = 256;
  
  // Get random crop of image
  int crop_x_start, crop_y_start;
  if(random_crop) {
    crop_x_start = fast_rand_int(get_fast_generator(), raw_width - Width + 1);
    crop_y_start = fast_rand_int(get_fast_generator(), raw_height - Height + 1);
  }
  else {
    crop_x_start = (raw_width - Width + 1) / 2;
    crop_y_start = (raw_height - Height + 1) / 2;
  }
  cv::Rect crop(crop_x_start, crop_y_start, Width, Height);
  cv::Mat image(raw_image, crop);

  for (int y = 0; y < Height; y++) {
    for (int x = 0; x < Width; x++) {
      cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
      int offset = y * Width + x;
      Pixels[offset]                  = pixel[_LBANN_CV_BLUE_];
      Pixels[offset + Height*Width]   = pixel[_LBANN_CV_GREEN_];
      Pixels[offset + 2*Height*Width] = pixel[_LBANN_CV_RED_];
    }
  }

  return true;
#else
  return false;
#endif
}

bool lbann::image_utils::loadJPG(std::vector<unsigned char>& image_buf, int& Width, int& Height, bool Flip, unsigned char *&Pixels) {
#ifdef __LIB_OPENCV
  cv::Mat image = cv::imdecode(image_buf, _LBANN_CV_COLOR_);
  //cv::Mat image = cv::imdecode(image_buf, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
  if (image.empty()) {
    return false;
  }

  Width = image.cols;
  Height = image.rows;

  for (int y = 0; y < Height; y++) {
    for (int x = 0; x < Width; x++) {
      cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
      int offset = (Flip) ? ((Height - 1 - y) * Width + x) : (y * Width + x);
      Pixels[offset]                  = pixel[_LBANN_CV_BLUE_];
      Pixels[offset + Height*Width]   = pixel[_LBANN_CV_GREEN_];
      Pixels[offset + 2*Height*Width] = pixel[_LBANN_CV_RED_];
    }
  }

  return true;
#else
  return false;
#endif
}

bool lbann::image_utils::saveJPG(const std::string& Imagefile, int Width, int Height, bool Flip, unsigned char *Pixels) {
#ifdef __LIB_OPENCV
  cv::Mat image = cv::Mat(Height, Width, CV_8UC3);

  for (int y = 0; y < Height; y++) {
    for (int x = 0; x < Width; x++) {
      cv::Vec3b pixel;
      int offset = (Flip) ? ((Height - 1 - y) * Width + x) : (y * Width + x);
      pixel[_LBANN_CV_BLUE_]  = Pixels[offset];
      pixel[_LBANN_CV_GREEN_] = Pixels[offset + Height*Width];
      pixel[_LBANN_CV_RED_]   = Pixels[offset + 2*Height*Width];
      image.at<cv::Vec3b>(y, x) = pixel;
    }
  }
  imwrite(Imagefile, image);

  return true;
#else
  return false;
#endif
}

bool lbann::image_utils::load_image(const std::string& filename,
                                    int& Width, int& Height, int& Type, cv_process& pp, std::vector<uint8_t>& buf) {
#ifdef __LIB_OPENCV
  cv::Mat image = cv::imread(filename, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
  bool ok = !image.empty() && pp.preprocess(image);
  ok = ok && cv_utils::copy_cvMat_to_buf(image, buf, pp);
  pp.disable_normalizer();

  _LBANN_MILD_EXCEPTION(!ok, "Image preprocessing or copying failed.", false)

  Width  = image.cols;
  Height = image.rows;
  Type   = image.type();
  return ok;
#else
  return false;
#endif // __LIB_OPENCV
}

bool lbann::image_utils::save_image(const std::string& filename,
                                    const int Width, const int Height, const int Type, cv_process& pp, const std::vector<uint8_t>& buf) {
#ifdef __LIB_OPENCV
  pp.determine_inverse_normalization();
  cv::Mat image = cv_utils::copy_buf_to_cvMat(buf, Width, Height, Type, pp);
  bool ok = !image.empty() && pp.postprocess(image);

  _LBANN_MILD_EXCEPTION(!ok, "Either the image is empty or postprocessing has failed.", false)

  return (ok && cv::imwrite(filename, image));
#else
  return false;
#endif // __LIB_OPENCV
}

/**
 *  @param filename The name of the image file to read in
 *  @param Width    The width of the image read
 *  @param Height   The height of the image read
 *  @param Type     The type of the image read (OpenCV code used for cv::Mat)
 *  @param pp       The pre-processing parameters
 *  @param data     The pre-processed image data to be stored in El::Matrix<DataType> format
 */
bool lbann::image_utils::load_image(const std::string& filename,
                                    int& Width, int& Height, int& Type, cv_process& pp, ::Mat& data) {
#ifdef __LIB_OPENCV
  cv::Mat image = cv::imread(filename, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
  bool ok = !image.empty() && pp.preprocess(image);
  ok = ok && cv_utils::copy_cvMat_to_buf(image, data, pp);
  // Disabling normalizer is needed because normalizer is not necessarily
  // called during preprocessing but implicitly applied during data copying to
  // reduce overhead.
  pp.disable_normalizer();

  _LBANN_MILD_EXCEPTION(!ok, "Image preprocessing or copying failed.", false)

  Width  = image.cols;
  Height = image.rows;
  Type   = image.type();
  return ok;
#else
  return false;
#endif // __LIB_OPENCV
}

/**
 *  @param filename The name of the image file to read in
 *  @param Width    The width of a patch from the image read
 *  @param Height   The height of a patch from the image read
 *  @param Type     The type of the image patches (OpenCV code used for cv::Mat)
 *  @param pp       The pre-processing parameters
 *  @param data     The pre-processed image data to be stored in El::Matrix<DataType> format
 */
bool lbann::image_utils::load_image(const std::string& filename,
                                    int& Width, int& Height, int& Type, cv_process_patches& pp, std::vector<::Mat>& data) {
#ifdef __LIB_OPENCV
  cv::Mat image = cv::imread(filename, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);

  std::vector<cv::Mat> patches;
  bool ok = !image.empty() && pp.preprocess(image, patches);
  if ((patches.size() == 0u) || (patches.size() != data.size())) {
    return false;
  }

  for(size_t i=0u; ok && (i < patches.size()); ++i) {
    ok = cv_utils::copy_cvMat_to_buf(patches[i], data[i], pp);
  }

  // Disabling normalizer is needed because normalizer is not necessarily
  // called during preprocessing but implicitly applied during data copying to
  // reduce overhead.
  pp.disable_normalizer();

  _LBANN_MILD_EXCEPTION(!ok, "Image preprocessing or copying failed.", false)

  Width  = patches[0].cols;
  Height = patches[0].rows;
  Type   = patches[0].type();
  return ok;
#else
  return false;
#endif // __LIB_OPENCV
}

/**
 *  @param filename The name of the image file to write
 *  @param Width    The width of the image to be written
 *  @param Height   The height of the image to be written
 *  @param Type     The type of the image to be written (OpenCV code used for cv::Mat)
 *  @param pp       The post-processing parameters
 *  @param data     The image data in El::Matrix<DataType> format to post-process and write
 */
bool lbann::image_utils::save_image(const std::string& filename,
                                    const int Width, const int Height, const int Type, cv_process& pp, const ::Mat& data) {
#ifdef __LIB_OPENCV
  pp.determine_inverse_normalization();
  cv::Mat image = cv_utils::copy_buf_to_cvMat(data, Width, Height, Type, pp);
  bool ok = !image.empty() && pp.postprocess(image);

  _LBANN_MILD_EXCEPTION(!ok, "Image postprocessing has failed.", false)

  return (ok && cv::imwrite(filename, image));
#else
  return false;
#endif // __LIB_OPENCV
}

/**
 *  @param inbuf   The buffer that contains the raw bytes read from an image file
 *                 This can be for example, const std:vector<uchar>& or const cv::Mat&.
 *                 http://docs.opencv.org/trunk/d4/d32/classcv_1_1__InputArray.html
 *  @param Width   The width of the image consturcted out of inbuf
 *  @param Height  The height of the image consructed
 *  @param Type    The type of the image constructed (OpenCV code used for cv::Mat)
 *  @param pp      The pre-processing parameters
 *  @param data    The pre-processed image data. A set of sub-matrix Views can be used to store the data.
 */
bool lbann::image_utils::import_image(cv::InputArray inbuf,
                                      int& Width, int& Height, int& Type, cv_process& pp, ::Mat& data) {
#ifdef __LIB_OPENCV
  cv::Mat image = cv::imdecode(inbuf, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
  bool ok = !image.empty() && pp.preprocess(image);
  ok = ok && cv_utils::copy_cvMat_to_buf(image, data, pp);
  pp.disable_normalizer();

  _LBANN_MILD_EXCEPTION(!ok, "Image preprocessing or copying failed.", false)

  Width  = image.cols;
  Height = image.rows;
  Type   = image.type();
  return ok;
#else
  return false;
#endif // __LIB_OPENCV
}

/**
 *  @param inbuf   The buffer that contains the raw bytes read from an image file
 *                 This can be for example, const std:vector<uchar>& or const cv::Mat&.
 *                 http://docs.opencv.org/trunk/d4/d32/classcv_1_1__InputArray.html
 *  @param Width   The width of a patch from the image consturcted out of inbuf
 *  @param Height  The height of a patch from the image consructed
 *  @param Type    The type of the image patches (OpenCV code used for cv::Mat)
 *  @param pp      The pre-processing parameters
 *  @param data    The pre-processed image data. A set of sub-matrix Views can be used to store the data.
 */
bool lbann::image_utils::import_image(cv::InputArray inbuf,
                                      int& Width, int& Height, int& Type, cv_process_patches& pp, std::vector<::Mat>& data) {
#ifdef __LIB_OPENCV
  cv::Mat image = cv::imdecode(inbuf, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);

  std::vector<cv::Mat> patches;
  bool ok = !image.empty() && pp.preprocess(image, patches);
  if ((patches.size() == 0u) || (patches.size() != data.size())) {
    return false;
  }

  for(size_t i=0u; ok && (i < patches.size()); ++i) {
    ok = cv_utils::copy_cvMat_to_buf(patches[i], data[i], pp);
  }

  pp.disable_normalizer();

  _LBANN_MILD_EXCEPTION(!ok, "Image preprocessing or copying failed.", false)

  Width  = patches[0].cols;
  Height = patches[0].rows;
  Type   = patches[0].type();
  return ok;
#else
  return false;
#endif // __LIB_OPENCV
}

/**
 *  @param fileExt The format extension name of image file: e.g., ".jpeg", ".png"
 *  @param outbuf  The preallocated buffer to contain the bytes to be written into an image file
 *  @param Width   The width of the image to be consturcted based on the given data of ::Mat
 *  @param Height  The height of the image
 *  @param Type    The type of the image (OpenCV code used for cv::Mat)
 *  @param pp      The post-processing parameters
 *  @param data    The image data. A sub-matrix View can be passed instead of the entire matrix.
 */
bool lbann::image_utils::export_image(const std::string& fileExt, std::vector<uchar>& outbuf,
                                      const int Width, const int Height, const int Type, cv_process& pp, const ::Mat& data) {
#ifdef __LIB_OPENCV
  pp.determine_inverse_normalization();
  cv::Mat image = cv_utils::copy_buf_to_cvMat(data, Width, Height, Type, pp);
  bool ok = !image.empty() && pp.postprocess(image);

  _LBANN_MILD_EXCEPTION(!ok, "Either the image is empty or postprocessing has failed.", false)
  _LBANN_MILD_EXCEPTION(fileExt.empty(), "Empty file format extension!", false)

  const std::string ext = ((fileExt[0] != '.')? ("." + fileExt) : fileExt);

  static const size_t max_img_header_size = 1024;
  const size_t capacity = cv_utils::image_data_amount(image) + max_img_header_size;

  if (outbuf.size() < capacity) {
    //std::cout << "bytes reserved for the image: " << image_data_amount(image) << std::endl;
    outbuf.resize(capacity);
  }

  return (ok && cv::imencode(ext, image, outbuf));
#else
  return false;
#endif // __LIB_OPENCV
}
