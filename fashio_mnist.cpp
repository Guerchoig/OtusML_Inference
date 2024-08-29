#include "fashio_mnist.h"
#include "eigen/Dense"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <utility>
#include <cmath>

// Load and construct model
model_t::model_t(std::string path)
{
    try
    {
        std::ifstream in(path);
        int row = 0;
        while (!in.eof())
        {
            std::string s;
            std::getline(in, s);
            if (!s.size())
                continue;
            std::stringstream ss(s);
            int col = 0;
            while (!ss.eof())
            {
                long_double coef;
                ss >> coef;
                coefs(row, col++) = coef;
            }
            row++;
        }
    }
    catch (std::ifstream::failure &e)
    {
        std::cerr << e.what() << '\n';
    }
}

// Load and construct one sample
sample_t::sample_t(std::ifstream &in)
{
    try
    {
        in >> label;
        vec(0, 0) = 1.0; // free member
        for (int i = 1; i < image_size + 1; ++i)
        {
            int buf;
            in >> buf;
            vec(i, 0) = buf / max_pix_value;
        }
    }
    catch (std::ifstream::failure &e)
    {
        std::cerr << e.what() << '\n';
    }
}

// Find which logistic model predict best given sample
int find_best_model(const sample_t &sample, const model_t &model)
{

    auto vec = (model.coefs * sample.vec).array();
    auto sigma = [](long_double degree) -> double
    {
        return 1.0 / (1.0 + std::exp(-degree));
    };
    int best_i = 0;
    long_double best_probability = 0.0;
    for (int i = 0; i < nof_classes; ++i)
    {
        auto coefficient = vec(i, 0);
        auto probability = sigma(coefficient);
        if (probability > best_probability)
        {
            best_probability = probability;
            best_i = i;
        }
    }
    return best_i;
}

// Verify the inference of logistic classification model
// on a test dataset
int main(int argc, char **argv)
{

    std::string test_path;
    std::string model_path;
    if (!get_params(argc, argv, test_path, model_path))
        return 0;
    model_t model(model_path);
    std::ifstream in_test(test_path);
    int successes = 0;
    long_double tests = 0;
    int i = 0;
    while (!in_test.eof())
    {
        tests++;
        sample_t spl(in_test);
        auto best_fit = find_best_model(spl, model);
        if (spl.label == best_fit)
            successes++;
        i++;
    }
    long_double accuracy = successes / tests;
    std::cout << accuracy << '\n';
}