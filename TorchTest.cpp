#include "FileUtils.hpp"
#include "Linear.hpp"
#include "Paths.h"
#include "ReLU.hpp"
#include "Reshape.hpp"
#include "Sequential.hpp"
#include "SpatialConvolution.hpp"
#include "SpatialConvolutionFactory.hpp"
#include "SpatialMaxPooling.hpp"
#include "Tanh.hpp"
#include "Tensor.hpp"
#include "TorchData.hpp"

#include <math.h>
#include <stddef.h>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#define mtorch_FLOAT_PRECISION 1e-6f
#define LOOSE_EPSILON 0.000001f
#define SAFE_DELETE(x) if (x != NULL) { delete x; x = NULL; }

using namespace mtorch;

class TorchLibTest {
public:
    TorchLibTest() {}
    virtual ~TorchLibTest() {}

protected:
    void testmtorchValue(mtorch::Tensor<float>* data, const std::string& filename,
      float precision);
    void assertTrue(bool value, const std::string& module_name);

};

void TorchLibTest::testmtorchValue(mtorch::Tensor<float>* data, const std::string& filename,
  float precision = mtorch_FLOAT_PRECISION) {

  float* correct_data;
  float* model_data;

  Tensor<float>* correct_data_tensor = Tensor<float>::loadFromFile(filename);

  if (!correct_data_tensor->isSameSizeAs(*data)) {
    std::cout << "Test FAILED (size mismatch)!: " << filename << std::endl;
  } else {

    correct_data = correct_data_tensor->getData();
    model_data = data->getData();
    bool data_correct = true;

    for (uint32_t i = 0; i < data->nelems() && data_correct; i++) {

      float delta = fabsf(model_data[i] - correct_data[i]) ;
      if (delta > precision && (delta /
        std::max<float>(fabsf(correct_data[i]), LOOSE_EPSILON)) > precision) {
        data_correct = false;

        for (uint32_t repeat = 0; repeat < 5; repeat++) {
          for (uint32_t cnt = 0; cnt < 60; cnt++) {
            std::cout << "*";
          }
          std::cout << std::endl;
        }
        std::cout << "index " << i << " incorrect!: " << std::endl;
        std::cout << std::fixed << std::setprecision(15);
        std::cout << "model_data[" << i << "] = " << model_data[i] << std::endl;
        std::cout << "correct_data[" << i << "] = " << correct_data[i] << std::endl;

        for (uint32_t repeat = 0; repeat < 5; repeat++) {
          for (uint32_t cnt = 0; cnt < 60; cnt++) {
            std::cout << "*";
          }
          std::cout << std::endl;
        }
      }
    }
    if (data_correct) {
      std::cout << "Test PASSED: " << filename << std::endl;
    } else {
      std::cout << "Test FAILED!: " << filename << std::endl;
    }
  }
  delete correct_data_tensor;
}

void TorchLibTest::assertTrue(bool value, const std::string& module_name) {
  if (value) {
    std::cout << "Test PASSED: " << module_name << std::endl;
  } else {
    std::cout << "Test FAILED!: " << module_name << std::endl;
  }
}

class FinalTest : public TorchLibTest {
public:

    void testsAllModules() {

        const std::string test_data_path = "TorchTestData";

        const uint32_t num_feats_in = 5;
        const uint32_t num_feats_out = 10;
        const uint32_t width = 10;
        const uint32_t height = 10;
        const uint32_t filt_height = 5;
        const uint32_t filt_width = 5;
        float din[width * height * num_feats_in];

        // CPU weights and biases for SpatialConvolution stage
        float cweights[num_feats_out * num_feats_in * filt_height * filt_width];
        float cbiases[num_feats_out];

        // CPU weights and biases for Linear stage
        const uint32_t lin_size_in = num_feats_in * width * height;
        const uint32_t lin_size_out = 20;
        float lweights[lin_size_in * lin_size_out];
        float lbiases[lin_size_out];


    try {

        std::cout << "Beginning mtorch tests..." << std::endl;

        const uint32_t isize[3] = {width, height, num_feats_in};
        Tensor<float>* data_in = new Tensor<float>(3, isize);

        for (uint32_t f = 0; f < num_feats_in; f++) {
        float val = (float)(f+1) - (float)(width * height) / 16.0f;
        for (uint32_t v = 0; v < height; v++) {
            for (uint32_t u = 0; u < width; u++) {
            din[f * width * height + v * width + u] = val;
            val += 1.0f / 8.0f;
            }
        }
        }
        data_in->setData(din);
        testmtorchValue(data_in, "data_in.bin");

        TorchData* output = NULL;
        Tensor<float>* data = Tensor<float>::clone(*data_in);

        // Test Tanh, Threshold and SpatialConvolutionMap in a Sequential container
        // (this means we can also test Sequential at the same time)
        Sequential stages;
        {
            // ***********************************************
            // Test Tanh
            stages.add(new Tanh());
            stages.forwardProp(*data, &output);
            testmtorchValue(TO_TENSOR_PTR(output), "tanh_result.bin");
            SAFE_DELETE(output);

            // ***********************************************
            // Test Threshold
            data = Tensor<float>::clone(*data_in);
            const float threshold = 0.5f;
            const float val = 0.1f;
            stages.add(new mtorch::Threshold());
            ((mtorch::Threshold*)stages.get(1))->threshold = threshold;
            ((mtorch::Threshold*)stages.get(1))->val = val;
            stages.forwardProp(*data, &output);
            testmtorchValue(TO_TENSOR_PTR(output),"threshold.bin");
        }

        TorchData* output_conv = NULL;

        // ***********************************************
        // Test SpatialConvolution
        {
            SpatialConvolution* conv = SpatialConvolutionFactory::create(num_feats_in, num_feats_out, filt_height, filt_width);
            for (int32_t i = 0; i < static_cast<int32_t>(num_feats_out); i++) {
                cbiases[i] = (float)(i + 1) / (float)num_feats_out - 0.5f;
            }
            const float sigma_x_sq = 1.0f;
            const float sigma_y_sq = 1.0f;
            const uint32_t filt_dim = filt_width * filt_height;
            for (int32_t fout = 0; fout < static_cast<int32_t>(num_feats_out); fout++) {
                for (int32_t fin = 0; fin < static_cast<int32_t>(num_feats_in); fin++) {
                    int32_t i = fout * num_feats_out + fin;
                    float scale = ((float)(i + 1) / (float)(num_feats_out * num_feats_in));
                    for (int32_t v = 0; v < static_cast<int32_t>(filt_height); v++) {
                        for (int32_t u = 0; u < static_cast<int32_t>(filt_width); u++) {
                            float x = (float)u - (float)(filt_width - 1) / 2.0f;
                            float y = (float)v - (float)(filt_height - 1) / 2.0f;
                            cweights[fout * filt_dim * num_feats_in + fin * filt_dim + v * filt_width + u] =
                                scale * expf(-((x*x) / (2.0f*sigma_x_sq) + (y*y) / (2.0f*sigma_y_sq)));
                        }
                    }
                }
            }
            conv->setWeights(cweights);
            conv->setBiases(cbiases);
            conv->forwardProp(*output, &output_conv);
            testmtorchValue(TO_TENSOR_PTR(output_conv),"spatial_convolution.bin");
            SAFE_DELETE(output_conv);

        const uint32_t padding = 6;
        SpatialConvolution* convmm = SpatialConvolutionFactory::create(num_feats_in, num_feats_out, filt_height,
            filt_width, padding, padding);
        Tensor<float>::copy(*convmm->weights(), *conv->weights());
        Tensor<float>::copy(*convmm->biases(), *conv->biases());
        convmm->forwardProp(*output, &output_conv);
        testmtorchValue(TO_TENSOR_PTR(output_conv),"spatial_convolution_mm_padding.bin");
        SAFE_DELETE(output_conv);
        SAFE_DELETE(output);
        delete conv;
        delete convmm;
        }

        // ***********************************************
        // Test SpatialMaxPooling
        {
        data = Tensor<float>::clone(*data_in);
        const uint32_t pool_u = 2;
        const uint32_t pool_v = 2;
        SpatialMaxPooling max_pool_stage(pool_v, pool_u, 0, 0, 0, 0);
        max_pool_stage.forwardProp(*data, &output);
        testmtorchValue(TO_TENSOR_PTR(output),"spatial_max_pooling.bin");
        }
        SAFE_DELETE(data);
        SAFE_DELETE(output);

        // ***********************************************
        // Test Linear
        {
        Sequential lin_stage;
        lin_stage.add(new Reshape(1, &lin_size_in));

        Linear* lin = new Linear(lin_size_in, lin_size_out);
        lin_stage.add(lin);
        // Weight matrix is M (rows = outputs) x N (columns = inputs)
        // It is stored column major with the M dimension stored contiguously
        for (uint32_t n = 0; n < lin_size_in; n++) {
            for (uint32_t m = 0; m < lin_size_out; m++) {
            uint32_t out_i = n * lin_size_out + m;
            uint32_t k = m * lin_size_in + n + 1;
            lweights[out_i] = (float)k / (float)(lin_size_in * lin_size_out);
            }
        }

        for (uint32_t i = 0; i < lin_size_out; i++) {
            lbiases[i] = (float)(i+1) / (float)(lin_size_out);
        }
        data = Tensor<float>::clone(*data_in);
        lin->setBiases(lbiases);
        lin->setWeights(lweights);
        lin_stage.forwardProp(*data, &output);
        testmtorchValue(TO_TENSOR_PTR(output),"linear.bin");
        SAFE_DELETE(output);
        }

        /*
        // ***********************************************
        // Profile convolution
        {
        const uint32_t fin = 128, fout = 512, kw = 11, kh = 11, pad = 5,
            imw = 90, imh = 60;
        const double t_test = 5.0;
        double t_start, t_end;
        uint64_t niters;
        SpatialConvolution conv(fin, fout, kh, kw, pad);
        SpatialConvolutionMM conv_mm(fin, fout, kh, kw, pad);
        uint32_t size[3] = {imw, imh, fin};
        Tensor<float>* input = new Tensor<float>(3, size);
        clk::Clk clk;

        Tensor<float>::fill(*conv.weights(), 1);
        Tensor<float>::fill(*conv.biases(), 1);
        Tensor<float>::fill(*input, 1);

        std::cout << "\tProfiling SpatialConvolutionMM for " << t_test <<
            " seconds" << std::endl;
        t_start = clk.getTime();
        t_end = t_start;
        niters = 0;
        while (t_end - t_start < t_test) {
            conv_mm.forwardProp(*input);
            niters++;
            mtorch::Sync();
            t_end = clk.getTime();
        }
        std::cout << "\t\tExecution time: " << (t_end - t_start) / (double)niters
            << " seconds per FPROP" << std::endl;

        std::cout << "\tProfiling SpatialConvolution for " << t_test <<
            " seconds" << std::endl;
        t_start = clk.getTime();
        t_end = t_start;
        niters = 0;
        while (t_end - t_start < t_test) {
            conv.forwardProp(*input);
            niters++;
            mtorch::Sync();
            t_end = clk.getTime();
        }
        std::cout << "\t\tExecution time: " << (t_end - t_start) / (double)niters
            << " seconds per FPROP" << std::endl;

        delete input;
        }
        */

        SAFE_DELETE(data_in);

    } catch (std::runtime_error & e) {
        std::cout << "Exception caught!" << std::endl;
        std::cout << e.what() << std::endl;
        exit(1);
    };
    }
};

int main() {
    FinalTest t;
    t.testsAllModules();
    return 0;
}
