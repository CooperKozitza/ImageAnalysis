#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

// The number of denoise operations applied to the image before post-processing.
constexpr int DENOISE_COUNT = 10;
// The size of the denoising kernel.
constexpr int DENOISE_RAD = 6;
// The percentile that gets set as the "black-point." 4 = 25%, 3 = 33%, 2 = 50%
constexpr int PERCENTILE_THRESHOLD_DIVISOR = 4;

constexpr int SOBEL_X[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
constexpr int SOBEL_Y[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

// A singular application of the sobel kernel on a pixel at (x, y)
float sobel_operator(const std::vector<float> &input_pixels, int x, int y,
                     int width, int height) {
  float gx = 0, gy = 0;

  size_t start_x = std::max(x - 1, 0), end_x = std::min(x + 1, width - 1);
  size_t start_y = std::max(y - 1, 0), end_y = std::min(y + 1, height - 1);

  for (size_t dy = start_y; dy < end_y + 1; ++dy) {
    for (size_t dx = start_x; dx < end_x + 1; ++dx) {
      float pixel_value = input_pixels[dy * width + dx];

      gx += pixel_value * SOBEL_X[dy - (y - 1)][dx - (x - 1)];
      gy += pixel_value * SOBEL_Y[dy - (y - 1)][dx - (x - 1)];
    }
  }

  return std::abs(gx) + std::abs(gy);
}

// Averages the values of all the pixels within the radius DENOISE_RAD around
// the pixel at (x, y)
float denoise_operator(const std::vector<float> &input_pixels, int x, int y,
                       int width, int height) {
  float g = 0.0f;

  int start_x = std::max(x - DENOISE_RAD, 0),
      end_x = std::min(x + DENOISE_RAD, width - 1);
  int start_y = std::max(y - DENOISE_RAD, 0),
      end_y = std::min(y + DENOISE_RAD, height - 1);

  int divisor = (end_x - start_x + 1) * (end_y - start_y + 1);

  for (int dy = start_y; dy <= end_y; ++dy) {
    int row_offset = dy * width;
    for (int dx = start_x; dx <= end_x; ++dx) {
      g += input_pixels[row_offset + dx];
    }
  }
  return g / static_cast<float>(divisor);
}

// Applies a kernel across multiple load-balanced threads
std::vector<float> apply_kernel(
    std::vector<float> &input_pixels, size_t width, size_t height,
    std::function<float(const std::vector<float> &, int, int, int, int)>
        kernel_func) {
  unsigned int max_thread_count = std::thread::hardware_concurrency();
  std::vector<std::thread> threads(max_thread_count);

  std::vector<float> output_pixels(width * height, 0);

  size_t chunk_size = width / max_thread_count;

  for (size_t thread_idx = 0; thread_idx < max_thread_count; ++thread_idx) {
    threads[thread_idx] = std::thread([&, thread_idx]() {
      size_t chunk_start = chunk_size * thread_idx;
      size_t chunk_end = (thread_idx == max_thread_count - 1)
                             ? width
                             : chunk_start + chunk_size;

      for (size_t y = 0; y < height; ++y) {
        for (size_t x = chunk_start; x < chunk_end; ++x) {
          output_pixels[y * width + x] =
              kernel_func(input_pixels, x, y, width, height);
        }
      }
    });
  }

  for (std::thread &thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  return output_pixels;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    throw std::runtime_error("No file provided");
  }

  const char *file_path = argv[1];
  int width, height, channels;
  unsigned char *image = stbi_load(file_path, &width, &height, &channels, 0);
  if (!image) {
    throw std::runtime_error("Unable to load image");
  }

  std::vector<float> input_pixels(width * height, 0);
  for (int i = 0; i < width * height; i++) {
    int index = i * channels;
    float average = 0;
    if (channels >= 3) {
      average = (image[index] + image[index + 1] + image[index + 2]) / 3.0f;
    }
    input_pixels[i] = average;
  }

  std::vector<float> output_pixels =
      apply_kernel(input_pixels, width, height, sobel_operator);
  for (int i = 0; i < DENOISE_COUNT; ++i) {
    output_pixels = apply_kernel(output_pixels, width, height, denoise_operator);
  }

  const float max_pixel_value =
      *std::max_element(output_pixels.begin(), output_pixels.end());
  const float normalization_factor = 255.0f / max_pixel_value;
  const size_t percentile_index =
      output_pixels.size() / PERCENTILE_THRESHOLD_DIVISOR;

  std::vector<float> pixel_values_for_percentile = output_pixels;
  pixel_values_for_percentile.erase(
      std::remove(pixel_values_for_percentile.begin(),
                  pixel_values_for_percentile.end(), 0),
      pixel_values_for_percentile.end());
  std::nth_element(pixel_values_for_percentile.begin(),
                   pixel_values_for_percentile.begin() + percentile_index,
                   pixel_values_for_percentile.end());
  float percentile_value =
      pixel_values_for_percentile[percentile_index] * normalization_factor;

  std::vector<unsigned char> output_image(width * height);
  for (size_t i = 0; i < output_pixels.size(); ++i) {
    float normalized_value = output_pixels[i] * normalization_factor;
    output_image[i] = normalized_value > percentile_value ? 0 : 255;
  }

  stbi_write_png("output.png", width, height, 1, output_image.data(), width);

  stbi_image_free(image);

  return 0;
}
