#include "func.hpp"

void matrix_multiply(queue& q, const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, size_t N) {
    // Создаем буферы для матриц A, B и C
    buffer<double, 2> a_buf(A.data(), range<2>(N, N));
    buffer<double, 2> b_buf(B.data(), range<2>(N, N));
    buffer<double, 2> c_buf(C.data(), range<2>(N, N));

    // Запускаем вычисления на устройстве
    q.submit([&](handler& h) {
        accessor A_access = a_buf.get_access<access::mode::read>(h);
        accessor B_access = b_buf.get_access<access::mode::read>(h);
        accessor C_access = c_buf.get_access<access::mode::write>(h);

        h.parallel_for<class MatrixMul>(range<2>(N, N), [=](id<2> index) {
            size_t row = index[0];
            size_t col = index[1];
            double sum = 0.0;

            for (size_t k = 0; k < N; ++k) {
                sum += A_access[row][k] * B_access[k][col];
            }

            C_access[row][col] = sum;
            });
        });
    q.wait();
}

// Функция умножения двух разреженных матриц в формате CSR
CSRMatrix sparse_matrix_multiply(const CSRMatrix& A, const CSRMatrix& B, queue& q) {
    if (A.cols != B.rows) {
        throw std::runtime_error("Размеры матриц не совпадают для умножения.");
    }

    // Результирующая матрица
    CSRMatrix C;
    C.rows = A.rows;
    C.cols = B.cols;

    std::vector<int> row_ptr_C(A.rows + 1, 0);
    std::vector<int> col_ind_C;
    std::vector<float> values_C;

    // Параллельное умножение разреженных матриц
    q.submit([&](handler& h) {
        // Используем буферы для передачи данных на устройство
        buffer<int> buf_row_ptr_A(A.row_ptr.data(), range<1>(A.row_ptr.size()));
        buffer<int> buf_col_ind_A(A.col_ind.data(), range<1>(A.col_ind.size()));
        buffer<float> buf_values_A(A.values.data(), range<1>(A.values.size()));

        buffer<int> buf_row_ptr_B(B.row_ptr.data(), range<1>(B.row_ptr.size()));
        buffer<int> buf_col_ind_B(B.col_ind.data(), range<1>(B.col_ind.size()));
        buffer<float> buf_values_B(B.values.data(), range<1>(B.values.size()));

        buffer<int> buf_row_ptr_C(row_ptr_C.data(), range<1>(row_ptr_C.size()));

        // Использование аксессоров
        auto acc_row_ptr_A = buf_row_ptr_A.get_access<access::mode::read>(h);
        auto acc_col_ind_A = buf_col_ind_A.get_access<access::mode::read>(h);
        auto acc_values_A = buf_values_A.get_access<access::mode::read>(h);

        auto acc_row_ptr_B = buf_row_ptr_B.get_access<access::mode::read>(h);
        auto acc_col_ind_B = buf_col_ind_B.get_access<access::mode::read>(h);
        auto acc_values_B = buf_values_B.get_access<access::mode::read>(h);

        auto acc_row_ptr_C = buf_row_ptr_C.get_access<access::mode::write>(h);

        // Параллельный цикл по строкам матрицы A
        h.parallel_for(range<1>(A.rows), [=](id<1> i) {
            int row_start_A = acc_row_ptr_A[i];
            int row_end_A = acc_row_ptr_A[i + 1];

            for (int j = row_start_A; j < row_end_A; ++j) {
                int col_A = acc_col_ind_A[j];
                float val_A = acc_values_A[j];

                int row_start_B = acc_row_ptr_B[col_A];
                int row_end_B = acc_row_ptr_B[col_A + 1];

                for (int k = row_start_B; k < row_end_B; ++k) {
                    int col_B = acc_col_ind_B[k];
                    float val_B = acc_values_B[k];

                    // Здесь можно добавить в результирующую матрицу
                    // например, с использованием атомарных операций
                    acc_row_ptr_C[i]++; // Увеличиваем количество ненулевых элементов
                }
            }
            });
        }).wait();

    // На этом этапе можно продолжить построение матрицы C (заполнение col_ind_C и values_C)

    C.row_ptr = row_ptr_C;
    C.col_ind = col_ind_C;
    C.values = values_C;

    return C;
}

// Функция для чтения матрицы в формате CSR из файла
CSRMatrix read_matrix_from_file(const std::string& filename) {
    CSRMatrix matrix;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл: " + filename);
    }

    // Считываем количество строк и столбцов
    file >> matrix.rows >> matrix.cols;

    // Считываем массив row_ptr
    matrix.row_ptr.resize(matrix.rows + 1);
    for (int i = 0; i <= matrix.rows; ++i) {
        file >> matrix.row_ptr[i];
    }

    // Считываем массив col_ind
    int non_zero_elements = matrix.row_ptr.back(); // Последний элемент row_ptr - это количество ненулевых элементов
    matrix.col_ind.resize(non_zero_elements);
    for (int i = 0; i < non_zero_elements; ++i) {
        file >> matrix.col_ind[i];
    }

    // Считываем массив values
    matrix.values.resize(non_zero_elements);
    for (int i = 0; i < non_zero_elements; ++i) {
        file >> matrix.values[i];
    }

    file.close();
    return matrix;
}

void Available_platforms()
{
    // Получаем платформы
    std::vector<platform> platforms = platform::get_platforms();

    for (const auto& plat : platforms) {
        std::cout << "Platform: " << plat.get_info<info::platform::name>() << "\n";

        // Получаем устройства на каждой платформе
        std::vector<device> devices = plat.get_devices();
        for (const auto& dev : devices) {
            std::cout << "  Device: " << dev.get_info<info::device::name>() << "\n";
            std::cout << "    Type: "
                << (dev.is_gpu() ? "GPU" : (dev.is_cpu() ? "CPU" : "Other"))
                << "\n";
            std::cout << "    Max Compute Units: "
                << dev.get_info<info::device::max_compute_units>()
                << "\n";
            std::cout << "    Global Memory Size: "
                << dev.get_info<info::device::global_mem_size>()
                << " bytes\n";
        }
    }

    std::cout << "\n" << std::endl;
}

void read_mmf_matrix(const std::string& filename, std::vector<double>& matrix, size_t& rows, size_t& cols) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл: " + filename);
    }

    std::string line;
    // Пропускаем комментарии в начале файла
    while (std::getline(file, line)) {
        if (line[0] != '%') {
            break;
        }
    }

    std::istringstream iss(line);
    iss >> rows >> cols;  // Читаем размеры матрицы

    matrix.resize(rows * cols, 0.0);  // Инициализируем матрицу

    // Читаем значения из файла
    size_t row, col;
    double value;
    while (file >> row >> col >> value) {
        // MMF формат использует 1-индексацию, поэтому переводим в 0-индексацию
        matrix[(row - 1) * cols + (col - 1)] = value;
    }

    file.close();
}

// Функция для умножения матриц с использованием SYCL
void mmf_matrix_multiply(queue& q, const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, size_t N) {
    buffer<double, 2> a_buf(A.data(), range<2>(N, N));
    buffer<double, 2> b_buf(B.data(), range<2>(N, N));
    buffer<double, 2> c_buf(C.data(), range<2>(N, N));

    q.submit([&](handler& h) {
        accessor A_access = a_buf.get_access<access::mode::read>(h);
        accessor B_access = b_buf.get_access<access::mode::read>(h);
        accessor C_access = c_buf.get_access<access::mode::write>(h);

        h.parallel_for(range<2>(N, N), [=](id<2> index) {
            size_t row = index[0];
            size_t col = index[1];
            double sum = 0.0;

            for (size_t k = 0; k < N; ++k) {
                sum += A_access[row][k] * B_access[k][col];
            }

            C_access[row][col] = sum;
        });
    }).wait();
}

/*
int main() {
    try {
        const std::string file_A = "matrix_A.mtx";  // Файл с первой матрицей в формате MMF
        const std::string file_B = "matrix_B.mtx";  // Файл с второй матрицей в формате MMF

        std::vector<double> A, B, C;
        size_t rows_A, cols_A, rows_B, cols_B;

        // Чтение матриц из файлов
        read_mmf_matrix(file_A, A, rows_A, cols_A);
        read_mmf_matrix(file_B, B, rows_B, cols_B);

        if (cols_A != rows_B) {
            throw std::runtime_error("Невозможно умножить матрицы: несоответствие размеров.");
        }

        // Инициализация результирующей матрицы
        C.resize(rows_A * cols_B, 0.0);

        // Создание очереди SYCL (для NVIDIA GPU или другого устройства)
        queue q(default_selector{}, [](sycl::exception_list e_list) {
            for (std::exception_ptr const& e : e_list) {
                try {
                    std::rethrow_exception(e);
                } catch (std::exception const& e) {
                    std::cout << "SYCL exception: " << e.what() << std::endl;
                }
            }
        });

        auto start = std::chrono::high_resolution_clock::now();

        // Выполняем умножение матриц
        matrix_multiply(q, A, B, C, rows_A);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Matrix multiplication took " << diff.count() << " seconds." << std::endl;

        // Вывод результата
        std::ofstream result_file("result_matrix.mtx");
        for (size_t i = 0; i < rows_A; ++i) {
            for (size_t j = 0; j < cols_B; ++j) {
                result_file << i + 1 << " " << j + 1 << " " << C[i * cols_B + j] << std::endl;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

*/