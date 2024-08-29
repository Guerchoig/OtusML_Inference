#pragma once
#include "eigen/Dense"
#include <iostream>
#include <sstream>
#include <vector>

using long_double = double;
constexpr int image_size = 784;
constexpr int nof_classes = 10;
constexpr long_double max_pix_value = 255.0;

// Contens test data for one image
struct sample_t
{
    int label;
    Eigen::Matrix<long_double, image_size + 1, 1> vec;
    sample_t(std::ifstream &in);
};

// Enclose a trained synthetic model, composed of
// nof_classes logistic models
struct model_t
{
    Eigen::Matrix<long_double, nof_classes, image_size + 1> coefs;
    model_t(std::string path);
};

// Find the model, which describes the best the given test sample
int find_best_model(const sample_t &sample, const model_t &model);

inline bool get_params(int argc, char **argv, std::string &test_path, std::string &model_path)
{
    bool res = true;

    switch (argc)
    {
    case 3:
        test_path = std::string(argv[1]);
        model_path = std::string(argv[2]);
        break;
    default:
        std::cout << "The use is: fashio_mnist <test_path> <model_path>\n";
        res = false;
        break;
    }
    return res;
}
