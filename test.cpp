#include "func.hpp"

int main() 
{
    const size_t N = 20000; // Размер матриц

    // Инициализация матриц
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C_cpu(N * N, 0.0);
    std::vector<double> C_gpu(N * N, 0.0);
    //std::vector<double> C_seq(N * N, 0.0);
    std::vector<double> C_mkl(N * N, 0.0);
    // CSRMatrix A_csr = read_matrix_from_file("A1.txt");
    // CSRMatrix B_csr = read_matrix_from_file("A1.txt");

    // Вывод информации о доступных платформах
    Available_platforms();

    // Заполнение матриц случайными числами
    for (size_t i = 0; i < N * N; ++i) {
        A[i] = static_cast<double>(rand()) / RAND_MAX;
        B[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    // Создаем очередь для выполнения на устройстве (NVIDIA GPU)
    try {
        // Создаем очередь для CPU
        queue cpu_queue(cpu_selector_v, [](exception_list e_list) {
            for (std::exception_ptr const& e : e_list) {
                try {
                    std::rethrow_exception(e);
                } catch (std::exception const& e) {
                    std::cout << "CPU exception: " << e.what() << std::endl;
                }
            }
        });

        // Создаем очередь для NVIDIA GPU
        queue gpu_queue(nvidia_selector, [](exception_list e_list) {
            for (std::exception_ptr const& e : e_list) {
                try {
                    std::rethrow_exception(e);
                } catch (std::exception const& e) {
                    std::cout << "GPU exception: " << e.what() << std::endl;
                }
            }
        });

        // Запускаем умножение матриц на CPU
        auto cpu_start = std::chrono::high_resolution_clock::now();
        matrix_multiply(cpu_queue, A, B, C_cpu, N);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        std::cout << "CPU matrix multiplication time: " 
                  << std::chrono::duration<double>(cpu_end - cpu_start).count() << " seconds" << std::endl;

        // Запускаем умножение матриц на NVIDIA GPU
        auto gpu_start = std::chrono::high_resolution_clock::now();
        matrix_multiply(gpu_queue, A, B, C_gpu, N);
        auto gpu_end = std::chrono::high_resolution_clock::now();
        std::cout << "GPU matrix multiplication time: " 
                  << std::chrono::duration<double>(gpu_end - gpu_start).count() << " seconds" << std::endl;

        // auto start = std::chrono::high_resolution_clock::now();
        // for (int i = 0; i < N; ++i)
        // {
        //     for (int j = 0; j < N; ++j)
        //     {
        //         for (int k = 0; k < N; ++k)
        //         {
        //             C_seq[i * N + j] += A[i * N + k] * B[k * N + j];
        //         }
        //     }
        // }
        // auto end = std::chrono::high_resolution_clock::now();
        // auto res = std::chrono::duration<double>(end - start).count();
        // std::cout << "Sequantal result: " << res << std::endl;

        A.resize(N, N);
        B.resize(N, N);
        C_mkl.resize(N, N);

        double *A_mkl = A.data();
        double *B_mkl = B.data();
        double *C_mkl_ptr = C_mkl.data();

        auto start = std::chrono::high_resolution_clock::now();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A_mkl, N, B_mkl, N, 1.0, C_mkl_ptr, N);
        auto end = std::chrono::high_resolution_clock::now();
        auto res = std::chrono::duration<double>(end - start).count();
        std::cout << "MKL result: " << res << std::endl;

        // Проверяем корректность вычислений
        bool correct = true;
        for (size_t i = 0; i < N * N; ++i) {
            if (std::abs(C_cpu[i] - C_gpu[i]) > 1e-6 && std::abs(C_cpu[i] - C_mkl_ptr[i]) > 1e-6) {
                std::cout << "Mismatch at index " << i << ": CPU result = " << C_cpu[i] 
                          << ", GPU result = " << C_gpu[i] << "MKL result = " << C_mkl[i] << std::endl;
                correct = false;
                break;
            }
        }

        if (correct) {
            std::cout << "Results are correct!" << std::endl;
        }

        // start = std::chrono::high_resolution_clock::now();
        // CSRMatrix C_csr_cpu = sparse_matrix_multiply(A_csr, B_csr, cpu_queue);
        // end = std::chrono::high_resolution_clock::now();
        // res = std::chrono::duration<double>(end - start).count();
        // std::cout << "CSR CPU result: " << res << std::endl;

        // start = std::chrono::high_resolution_clock::now();
        // CSRMatrix C_csr_gpu = sparse_matrix_multiply(A_csr, B_csr, gpu_queue);
        // end = std::chrono::high_resolution_clock::now();
        // res = std::chrono::duration<double>(end - start).count();
        // std::cout << "CSR GPU result: " << res << std::endl;

    } catch (sycl::exception const& e) {
        std::cerr << "SYCL exception caught: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}