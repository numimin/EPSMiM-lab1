#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <barrier>
#include <thread>
#include <atomic>

#include <smmintrin.h>
#include <immintrin.h>

double f(int n, double tau) {
    constexpr double f0 = 1.0;
    constexpr double t0 = 1.5;
    constexpr double gamma = 4.0;
    const double a = 2 * M_PI * f0 * (n * tau - t0);
    return std::exp((- a * a) / (gamma * gamma)) * std::sin(a);
}

const __m256d two = _mm256_set_pd(2, 2, 2, 2);
const __m256d zeros = _mm256_set_pd(0, 0, 0, 0);

__m256d calculate(std::vector<double>& prev, const std::vector<double>& curr,
    const std::vector<double>& p,
    int y, int x, int actual_nx,
    __m256d m_hxrec, __m256d m_hyrec, __m256d m_tau) 
{
    int index = y * actual_nx + x;
    __m256d uc = _mm256_loadu_pd(&curr[index]);
    __m256d ur = _mm256_loadu_pd(&curr[y * actual_nx + x + 1]);
    __m256d ul = _mm256_loadu_pd(&curr[y * actual_nx + x - 1]);
    __m256d ut = _mm256_loadu_pd(&curr[(y + 1) * actual_nx + x]);
    __m256d ud = _mm256_loadu_pd(&curr[(y - 1) * actual_nx + x]);
    __m256d pc = _mm256_loadu_pd(&p[index]);
    __m256d pd = _mm256_loadu_pd(&p[(y - 1) * actual_nx + x]);
    __m256d pdl = _mm256_loadu_pd(&p[(y - 1) * actual_nx + x - 1]);
    __m256d pl = _mm256_loadu_pd(&p[y * actual_nx + x - 1]);

    __m256d uPrev = _mm256_loadu_pd(&prev[index]);

    __m256d uc2 = _mm256_mul_pd(two, uc);
    __m256d uc2_uprev = _mm256_sub_pd(uc2, uPrev);

    __m256d ur_uc = _mm256_sub_pd(ur, uc);
    __m256d pd_pc = _mm256_add_pd(pd, pc);
    __m256d prod1 = _mm256_mul_pd(ur_uc, pd_pc);

    __m256d ul_uc = _mm256_sub_pd(ul, uc);
    __m256d pdl_pl = _mm256_add_pd(pdl, pl);
    __m256d prod2 = _mm256_mul_pd(ul_uc, pdl_pl);

    __m256d sum1 = _mm256_add_pd(prod1, prod2);
    sum1 = _mm256_mul_pd(sum1, m_hxrec);

    __m256d ut_uc = _mm256_sub_pd(ut, uc);
    __m256d pl_pc = _mm256_add_pd(pl, pc);
    prod1 = _mm256_mul_pd(ut_uc, pl_pc);

    __m256d ud_uc = _mm256_sub_pd(ud, uc);
    __m256d pdl_pd = _mm256_add_pd(pdl, pd);
    prod2 = _mm256_mul_pd(ud_uc, pdl_pd);

    __m256d sum2 = _mm256_add_pd(prod1, prod2);
    sum2 = _mm256_mul_pd(sum2, m_hyrec);

    __m256d sum3 = _mm256_add_pd(sum1, sum2);
    sum3 = _mm256_mul_pd(m_tau, sum3);
    sum3 = _mm256_mul_pd(m_tau, sum3);

    __m256d result = _mm256_add_pd(uc2_uprev, sum3);

    _mm256_storeu_pd(&prev[index], result);

    __m256d minus_result = _mm256_sub_pd(zeros, result);
    __m256d abs_result = _mm256_max_pd(result, minus_result);

    __m256d permuted_result = _mm256_permute2f128_pd(abs_result, abs_result, 1);
    __m256d m1 = _mm256_max_pd(abs_result, permuted_result);
    __m256d m2 = _mm256_permute_pd(m1, 5);
    return _mm256_max_pd(m1, m2);
}

struct ThreadData {
    int thread_number;
    int thread_count; 
    std::vector<double>* u;
    std::vector<double>* p;
    __m256d m_hxrec; 
    __m256d m_hyrec; 
    __m256d m_tau;
    double tau;
    int nx; 
    int ny; 
    int actual_nx;
    int nt;
    int sx;
    int sy;
    std::vector<std::atomic_int>* flags;
    std::vector<double>* maxElements;

    ThreadData() {}
    ThreadData(
        int thread_number,
        int thread_count, 
        std::vector<double>* u,
        std::vector<double>* p,
        __m256d m_hxrec,
        __m256d m_hyrec, 
        __m256d m_tau,
        double tau,
        int nx,
        int ny, 
        int actual_nx,
        int nt,
        int sx,
        int sy,
        std::vector<std::atomic_int>* flags,
        std::vector<double>* maxElements): thread_number {thread_number},
                 thread_count {thread_count},
                 u {u}, p {p},
                 m_hxrec {m_hxrec}, m_hyrec {m_hyrec}, m_tau {m_tau}, tau {tau},
                 nx {nx}, ny {ny}, actual_nx {actual_nx}, nt {nt}, sx {sx}, sy {sy},
                 flags {flags}, maxElements {maxElements}
                 {}
};

void synchronize(std::vector<std::atomic_int>* flags, int thread_count, int thread_number) {
    const int count = (*flags)[thread_number * 256].fetch_add(1) + 1;
    while (true) {
        if (count == (*flags)[thread_count * 256]) {
            return;
        }
        bool all = true;
        for (int i = 0; i < thread_count; i++) {
            all = (*flags)[i * 256] == count;
            if (!all) {
                break;
            }
        }
        if (all) {
            (*flags)[thread_count * 256].store(count);
            (*flags)[thread_count * 256].notify_all();
            return;
        }
        (*flags)[thread_count * 256].wait(count - 1);
    }
}

void calculate_thread(ThreadData* data) {
    const int thread_number = data->thread_number;
    const int thread_count = data->thread_count;
    auto u = data->u;
    auto p = data->p;
    const auto m_hxrec = data->m_hxrec;
    const auto m_hyrec = data->m_hyrec;
    const auto m_tau = data->m_tau;
    const double tau = data->tau;
    const int nx = data->nx;
    const int ny = data->ny;
    const int actual_nx = data->actual_nx;
    const int nt = data->nt;
    const int sx = data->sx;
    const int sy = data->sy;
    auto flags = data->flags;
    auto maxElements = data->maxElements;

    const int y_count = ny / thread_count;
    int first_y = y_count * thread_number;
    int last_y = (thread_number == thread_count - 1) ? ny : (y_count * (thread_number + 1));
    for (int y = first_y; y < last_y; y++) {
        for (int x = 0; x < nx; x++) {
            if (x < nx / 2) {
                (*p)[y * actual_nx + x] = 0.1 * 0.1;
            } else {
                (*p)[y * actual_nx + x] = 0.2 * 0.2;
            }
            u[0][y * actual_nx + x] = 0;
            u[1][y * actual_nx + x] = 0;
        }
    }

    synchronize(flags, thread_count, thread_number);

    first_y = (thread_number == 0) ? first_y + 1 : first_y;
    last_y = (thread_number == thread_count - 1) ? last_y - 1 : last_y;

    int prevIndex = 0;
    int currIndex = 1;

    for (int i = 0; i < nt; i += 5) {
        double maxElement = 0;
        //Prepare data
        for (int y = first_y; y < first_y + 7; y++) {
            for (int x = 1; x < nx - 1; x += 4) {
                calculate(u[prevIndex], u[currIndex], *p, 
                    y, x, actual_nx, 
                    m_hxrec, m_hyrec, m_tau);
            }
            u[prevIndex][y * actual_nx + nx - 1] = 0;

            if (y == sy) {
                u[prevIndex][sy * actual_nx + sx] += tau * tau * f(i, tau);                
            }
        }

        for (int y = first_y; y < first_y + 6; y++) {
            for (int x = 1; x < nx - 1; x += 4) {
                calculate(u[currIndex], u[prevIndex], *p, 
                    y, x, actual_nx, 
                    m_hxrec, m_hyrec, m_tau);
            }
            u[currIndex][y * actual_nx + nx - 1] = 0;

            if (y == sy) {
                u[currIndex][sy * actual_nx + sx] += tau * tau * f(i + 1, tau);                
            }
        }

        for (int y = first_y; y < first_y + 4; y++) {
            for (int x = 1; x < nx - 1; x += 4) {
                calculate(u[prevIndex], u[currIndex], *p, 
                    y, x, actual_nx, 
                    m_hxrec, m_hyrec, m_tau);
            }
            u[prevIndex][y * actual_nx + nx - 1] = 0;

            if (y == sy) {
                u[prevIndex][sy * actual_nx + sx] += tau * tau * f(i + 2, tau);                
            }
        }

        for (int y = first_y; y < first_y + 3; y++) {
            for (int x = 1; x < nx - 1; x += 4) {
                calculate(u[currIndex], u[prevIndex], *p, 
                    y, x, actual_nx, 
                    m_hxrec, m_hyrec, m_tau);
            }
            u[currIndex][y * actual_nx + nx - 1] = 0;

            if (y == sy) {
                u[currIndex][sy * actual_nx + sx] += tau * tau * f(i + 3, tau);                
            }
        }

        //main loop
        for (int y = first_y + 7; y < last_y; y++) {
            for (int x = 1; x < nx - 1; x += 4) {
                __m256d newElement = calculate(u[prevIndex], u[currIndex], *p, 
                    y - 7, x, actual_nx, 
                    m_hxrec, m_hyrec, m_tau);
                maxElement = std::max(maxElement, *((double*)&newElement));
            }
            u[prevIndex][y * actual_nx + nx - 1] = 0;

            if (y == first_y + 7) {
                synchronize(flags, thread_count, thread_number);               
            }

            if (y - 7 == sy) {
                u[prevIndex][sy * actual_nx + sx] += tau * tau * f(i + 4, tau);                
            }

            for (int x = 1; x < nx - 1; x += 4) {
                calculate(u[currIndex], u[prevIndex], *p, 
                    y - 4, x, actual_nx, 
                    m_hxrec, m_hyrec, m_tau);
            }
            u[currIndex][y * actual_nx + nx - 1] = 0;

            if (y - 4 == sy) {
                u[currIndex][sy * actual_nx + sx] += tau * tau * f(i + 3, tau);                
            }

            for (int x = 1; x < nx - 1; x += 4) {
                calculate(u[prevIndex], u[currIndex], *p, 
                    y - 3, x, actual_nx, 
                    m_hxrec, m_hyrec, m_tau);
            }
            u[prevIndex][y * actual_nx + nx - 1] = 0;

            if (y - 3 == sy) {
                u[prevIndex][sy * actual_nx + sx] += tau * tau * f(i + 2, tau);                
            }

            for (int x = 1; x < nx - 1; x += 4) {
                calculate(u[currIndex], u[prevIndex], *p, 
                    y - 1, x, actual_nx, 
                    m_hxrec, m_hyrec, m_tau);
            }
            u[currIndex][y * actual_nx + nx - 1] = 0;

            if (y - 1 == sy) {
                u[currIndex][sy * actual_nx + sx] += tau * tau * f(i + 1, tau);                
            }

            for (int x = 1; x < nx - 1; x += 4) {
                calculate(u[prevIndex], u[currIndex], *p, 
                    y, x, actual_nx, 
                    m_hxrec, m_hyrec, m_tau);
            }
            u[prevIndex][y * actual_nx + nx - 1] = 0;

            if (y == sy) {
                u[prevIndex][sy * actual_nx + sx] += tau * tau * f(i, tau);                
            }
        }

        //final calculations
        for (int y = last_y - 6; y < last_y; y++) {
            for (int x = 1; x < nx - 1; x += 4) {
                __m256d newElement = calculate(u[prevIndex], u[currIndex], *p, 
                    y, x, actual_nx, 
                    m_hxrec, m_hyrec, m_tau);
                maxElement = std::max(maxElement, *((double*)&newElement));
            }
            u[prevIndex][y * actual_nx + nx - 1] = 0;

            if (y == sy) {
                u[prevIndex][sy * actual_nx + sx] += tau * tau * f(i + 4, tau);                
            }
        }

        for (int y = last_y - 3; y < last_y; y++) {
            for (int x = 1; x < nx - 1; x += 4) {
                calculate(u[prevIndex], u[currIndex], *p, 
                    y, x, actual_nx, 
                    m_hxrec, m_hyrec, m_tau);
            }
            u[prevIndex][y * actual_nx + nx - 1] = 0;

            if (y == sy) {
                u[prevIndex][sy * actual_nx + sx] += tau * tau * f(i + 3, tau);                
            }
        }
        
        for (int y = last_y - 2; y < last_y; y++) {
            for (int x = 1; x < nx - 1; x += 4) {
                calculate(u[prevIndex], u[currIndex], *p, 
                    y, x, actual_nx, 
                    m_hxrec, m_hyrec, m_tau);
            }
            u[prevIndex][y * actual_nx + nx - 1] = 0;

            if (y == sy) {
                u[prevIndex][sy * actual_nx + sx] += tau * tau * f(i + 2, tau);                
            }
        }

        for (int y = last_y - 1; y < last_y; y++) {
            for (int x = 1; x < nx - 1; x += 4) {
                calculate(u[prevIndex], u[currIndex], *p, 
                    y, x, actual_nx, 
                    m_hxrec, m_hyrec, m_tau);
            }
            u[prevIndex][y * actual_nx + nx - 1] = 0;

            if (y == sy) {
                u[prevIndex][sy * actual_nx + sx] += tau * tau * f(i + 1, tau);                
            }
        }

        std::swap(currIndex, prevIndex);
        (*maxElements)[thread_number] = maxElement;
        synchronize(flags, thread_count, thread_number);               
        if (thread_number == 0) {
            std::cout << i << std::endl;
            std::cout << *std::max_element(maxElements->begin(), maxElements->end()) << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 7) {
        std::cout << "Usage: ./lab1 Nx Ny Nt Sx Sy THREAD_COUNT" << std::endl;
        return EXIT_FAILURE;
    }

    const int nx = std::stoi(argv[1]);
    const int ny = std::stoi(argv[2]);
    const int nt = std::stoi(argv[3]);
    const int sx = std::stoi(argv[4]);
    const int sy = std::stoi(argv[5]);
    const int thread_count = std::stoi(argv[6]);

    const auto start = std::chrono::high_resolution_clock::now();

    constexpr double xa = 0.0;
    constexpr double xb = 4.0;
    constexpr double ya = 0.0;
    constexpr double yb = 4.0;

    const double hx = (xb - xa) / (nx - 1);
    const double hy = (yb - ya) / (ny - 1);

    const double hxrec = 1 / (2 * hx * hx);
    const double hyrec = 1 / (2 * hy * hy);
    const __m256d m_hxrec = _mm256_set_pd(hxrec, hxrec, hxrec, hxrec);
    const __m256d m_hyrec = _mm256_set_pd(hyrec, hyrec, hyrec, hyrec);

    const double tau = (nx <= 1000 && ny <= 1000) ? 0.01 : 0.001;
    const __m256d m_tau = _mm256_set_pd(tau, tau, tau, tau);

    const int actual_nx = nx + 4;
    std::vector<double> u[] {std::vector<double>(actual_nx * ny), std::vector<double>(actual_nx * ny)};
    std::vector<double> p(actual_nx * ny);
    
    std::vector<ThreadData> thread_data(thread_count);
    std::vector<double> maxElements(thread_count, 0);
    std::vector<std::jthread> threads;
    std::vector<std::atomic_int> flags((thread_count + 1)* 256); //each flag is on different cache line

    for (int i = 0; i < thread_count + 1; i++) {
        flags[i * 256].store(0);
    }

    for (int i = 1; i < thread_count; i++) {
        thread_data[i] = ThreadData(i, thread_count, 
            u, &p, m_hxrec, m_hyrec, m_tau, tau,
            nx, ny, actual_nx, nt, sx, sy, &flags, &maxElements);
        threads.emplace_back(calculate_thread, &thread_data[i]);
    }

    thread_data[0] = ThreadData(0, thread_count, 
            u, &p, m_hxrec, m_hyrec, m_tau, tau,
            nx, ny, actual_nx, nt, sx, sy, &flags, &maxElements);
    calculate_thread(&thread_data[0]);

    for (int i = 0; i < thread_count - 1; i++) {
        threads[i].join();
    }

    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << duration.count() / 1000.0 << "s" << std::endl;

    FILE* file = std::fopen("./main.dat", "w");
    for (int y = 0; y < ny; y++) {
        std::fwrite(&u[nt % 2][y * actual_nx], sizeof(double), nx, file);        
    }
    std::fclose(file);

    return EXIT_SUCCESS;
}