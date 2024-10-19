#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <string>
#include <chrono>
#include <fstream>
#include <mkl.h>

#define eps 1e-20

using namespace sycl;

auto nvidia_selector = [](const device& dev) {
    const std::string name = dev.get_info<info::device::name>();
    if (name.find("NVIDIA") != std::string::npos) {
        return 1;  // Максимальный приоритет для NVIDIA устройств
    }
    return -1;  // Отбрасываем другие устройства
};

static auto exception_handler = [](sycl::exception_list e_list) {
    for (std::exception_ptr const& e : e_list) {
        try {
            std::rethrow_exception(e);
        }
        catch (std::exception const& e) {
#if _DEBUG
            std::cout << "Failure" << std::endl;
#endif
            std::terminate();
        }
    }
};

void matrix_multiply(queue& q, const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, size_t N);

void Available_platforms();
