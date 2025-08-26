#pragma once
#ifndef UNDISTORTER_H
#define UNDISTORTER_H

#include <opencv2/opencv.hpp>
#include <memory>

class Undistorter {
public:
  virtual ~Undistorter() = default;
  virtual void undistort(cv::InputArray img, cv::OutputArray res) = 0;
  [[nodiscard]] virtual const cv::Mat& getK() const noexcept = 0;
  [[nodiscard]] virtual const cv::Mat& getOriginalK() const noexcept = 0;
  [[nodiscard]] virtual int getOutputWidth() const noexcept = 0;
  [[nodiscard]] virtual int getOutputHeight() const noexcept = 0;
  [[nodiscard]] virtual bool is_valid() const noexcept = 0;
};

class UndistorterOpenCV final : public Undistorter {
public:
  explicit UndistorterOpenCV(const std::string& config_fn);
  ~UndistorterOpenCV() override = default;

  UndistorterOpenCV(const UndistorterOpenCV&) = delete;
  UndistorterOpenCV& operator=(const UndistorterOpenCV&) = delete;
  UndistorterOpenCV(UndistorterOpenCV&&) = default;
  UndistorterOpenCV& operator=(UndistorterOpenCV&&) = default;

  void undistort(cv::InputArray img, cv::OutputArray res) override;
  [[nodiscard]] const cv::Mat& getK() const noexcept override { return m_K; }
  [[nodiscard]] const cv::Mat& getOriginalK() const noexcept override { return m_originalK; }
  [[nodiscard]] int getOutputWidth() const noexcept override { return m_out_width; }
  [[nodiscard]] int getOutputHeight() const noexcept override { return m_out_height; }
  [[nodiscard]] bool is_valid() const noexcept override { return m_valid; }

private:
  cv::Mat m_K, m_originalK;
  int m_out_width = 0, m_out_height = 0;
  int m_in_width = 0, m_in_height = 0;
  cv::Mat m_map1, m_map2;
  bool m_valid = false;

  std::vector<float> m_input_calibration;
  float m_output_calibration;
};

#endif
