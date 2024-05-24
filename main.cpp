#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

// The number of denoise operations applied to the image before post-processing.
constexpr int DENOISE_COUNT = 8;
constexpr int DENOISE_RAD = 9;

constexpr int BLUR_COUNT = 20;
constexpr int BLUR_RAD = 3;

constexpr int CERTAINTY = 5;

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
float blur_operator(const std::vector<float> &input_pixels, int x, int y,
                    int width, int height) {
  float g = 0.0f;

  int start_x = std::max(x - BLUR_RAD, 0),
      end_x = std::min(x + BLUR_RAD, width - 1);
  int start_y = std::max(y - BLUR_RAD, 0),
      end_y = std::min(y + BLUR_RAD, height - 1);

  int divisor = (end_x - start_x + 1) * (end_y - start_y + 1);

  for (int dy = start_y; dy <= end_y; ++dy) {
    int row_offset = dy * width;
    for (int dx = start_x; dx <= end_x; ++dx) {
      g += input_pixels[row_offset + dx];
    }
  }
  return g / static_cast<float>(divisor);
}

float dialate_operator(const std::vector<float> &input_pixels, int x, int y,
                       int width, int height) {
  if (input_pixels[y * width + x] == 0.0f) {
    return 0;
  }

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

  std::vector<float> pixels(width * height, 0);

  size_t chunk_size = width / max_thread_count;

  for (size_t thread_idx = 0; thread_idx < max_thread_count; ++thread_idx) {
    threads[thread_idx] = std::thread([&, thread_idx]() {
      size_t chunk_start = chunk_size * thread_idx;
      size_t chunk_end = (thread_idx == max_thread_count - 1)
                             ? width
                             : chunk_start + chunk_size;

      for (size_t y = 0; y < height; ++y) {
        for (size_t x = chunk_start; x < chunk_end; ++x) {
          pixels[y * width + x] =
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
  return pixels;
}

std::vector<unsigned char> dialate(std::vector<unsigned char> &input_pixels,
                                   size_t width, size_t height) {
  std::vector<float> pixels(width * height);
  for (size_t i = 0; i < input_pixels.size(); ++i) {
    pixels[i] = static_cast<float>(input_pixels[i]);
  }

  for (int i = 0; i < DENOISE_COUNT; ++i) {
    pixels = apply_kernel(pixels, width, height, dialate_operator);
    for (size_t i = 0; i < pixels.size(); ++i) {
      pixels[i] = pixels[i] > 127 ? 255 : 0;
    }
  }

  std::vector<unsigned char> output_pixels(width * height);
  for (size_t i = 0; i < pixels.size(); ++i) {
    output_pixels[i] = pixels[i];
  }

  return output_pixels;
}

void process_image(const char *file_path, const char *output_path) {
  int width, height, channels;
  unsigned char *image = stbi_load(file_path, &width, &height, &channels, 0);
  if (!image) {
    throw std::runtime_error("unable to load image");
  }

  std::cout << "-loaded image " << file_path << std::endl;

  std::vector<float> pixels(width * height, 0);
  for (int i = 0; i < width * height; i++) {
    int index = i * channels;
    float average = 0;
    if (channels >= 3) {
      average = (image[index] + image[index + 1] + image[index + 2]) / 3.0f;
    }
    pixels[i] = average;
  }

  std::cout << "-reduced channels" << std::endl;

  pixels = apply_kernel(pixels, width, height, blur_operator);

  pixels = apply_kernel(pixels, width, height, sobel_operator);

  std::cout << "-finished edge detection" << std::endl;

  for (int i = 0; i < BLUR_COUNT; ++i) {
    pixels = apply_kernel(pixels, width, height, blur_operator);
    std::cout << "-blur %" << (static_cast<float>(i) / BLUR_COUNT * 100)
              << " complete" << std::endl;
  }

  std::cout << "-mapping pixel values" << std::endl;

  std::unordered_map<unsigned char, unsigned int> pixel_frequency;
  for (unsigned char pixel_value : pixels) {
    if (pixel_value < 1 || pixel_value > 50) {
      continue;
    }

    ++pixel_frequency[pixel_value];
  }

  unsigned char threshold =
      std::max_element(
          pixel_frequency.begin(), pixel_frequency.end(),
          [](const std::pair<int, int> &a, const std::pair<int, int> &b) {
            return a.second < b.second;
          })
          ->first;

  std::cout << "-calculating values... (t=" << static_cast<int>(threshold)
            << ")" << std::endl;

  std::vector<unsigned char> output_image(width * height);
  for (size_t i = 0; i < pixels.size(); ++i) {
    unsigned char dist =
        std::abs(static_cast<unsigned char>(pixels[i]) - threshold);
    output_image[i] = dist < CERTAINTY ? 255 : 0;
  }

  output_image = dialate(output_image, width, height);

  std::cout << "-saving as " << output_path << std::endl;

  stbi_write_png(output_path, width, height, 1, output_image.data(), width);
  stbi_image_free(image);
}

int main(const int argc, const char **argv) {
  if (argc < 2) {
    throw std::runtime_error("no files provided");
  }

  for (size_t i = 1; i < argc; ++i) {
    std::string output_path = "output_" + std::to_string(i) + ".png";
    process_image(argv[i], output_path.c_str());
  }

  return 0;
}
