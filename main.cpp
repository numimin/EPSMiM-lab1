#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <chrono>
#include <cstdlib>

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

void calculate(std::vector<double>& prev, const std::vector<double>& curr,
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
}

int main(int argc, char* argv[]) {
    if (argc != 6 && argc != 7) {
        std::cout << "Usage: ./lab1 Nx Ny Nt Sx Sy" << std::endl;
        return EXIT_FAILURE;
    }

    const int nx = std::stoi(argv[1]);
    const int ny = std::stoi(argv[2]);
    const int nt = std::stoi(argv[3]);
    const int sx = std::stoi(argv[4]);
    const int sy = std::stoi(argv[5]);
    const int iterations = (argc == 7) ? std::stoi(argv[6]) : 1;

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

    int prevIndex = 0;
    int currIndex = 1;

    const int actual_nx = nx + 4;

    std::vector<double> u[] {std::vector<double>(actual_nx * ny, 0), std::vector<double>(actual_nx * ny, 0)};

    std::vector<double> p(actual_nx * ny);

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            if (x < nx / 2) {
                p[y * actual_nx + x] = 0.1 * 0.1;
            } else {
                p[y * actual_nx + x] = 0.2 * 0.2;
            }
        }
    }

    for (int i = 0; i < nt; i += iterations) {
        int first_iterations = 1;
        for (int iter = 0; iter < iterations - 1; iter++) {
            if (iter % 2 == 0) {
                first_iterations++;
            } else {
                first_iterations += 2;
            }
        }

        const int max_first_iterations = first_iterations;
        for (int iter = iterations - 2; iter >= 0; iter--) {
            for (int y = 1; y < first_iterations + 1; y++) {
                for (int x = 1; x < nx - 1; x += 4) {
                    calculate(u[prevIndex], u[currIndex], p, 
                        y, x, actual_nx, 
                        m_hxrec, m_hyrec, m_tau);
                }

                if (y == sy) {
                    u[prevIndex][sy * actual_nx + sx] += tau * tau * f(i + iter - iterations + 2, tau);                
                }
            }
            
            if ((iter + iterations) % 2 == 0) {
                first_iterations--;
            } else {
                first_iterations -= 2;
            }
            std::swap(currIndex, prevIndex);
        }

        for (int y = max_first_iterations + 1; y < ny - 1; y++) {
            int curr_y_delta = max_first_iterations;
            for (int iter = iterations - 2; iter >= 0; iter--) {
                for (int x = 1; x < nx - 1; x += 4) {
                    calculate(u[prevIndex], u[currIndex], p, 
                        y - curr_y_delta, x, actual_nx, 
                        m_hxrec, m_hyrec, m_tau);
                }

                if (y - curr_y_delta == sy) {
                    u[prevIndex][sy * actual_nx + sx] += tau * tau * f(i + iter + 1, tau);                
                }
                std::swap(currIndex, prevIndex);
                if (iter == iterations - 2) {
                    curr_y_delta -= 1;
                }

                if (iter % 2 == 0) {
                    curr_y_delta -= 1;
                } else {
                    curr_y_delta -= 2;
                }
            }

            for (int x = 1; x < nx - 1; x += 4) {
                calculate(u[prevIndex], u[currIndex], p, 
                    y, x, actual_nx, 
                    m_hxrec, m_hyrec, m_tau);
            }

            if (y == sy) {
                u[prevIndex][sy * actual_nx + sx] += tau * tau * f(i, tau);                
            }
        }

        if (iterations % 2 == 1) {
            std::swap(currIndex, prevIndex);
        }

        /*for (int iter = iterations - 2; iter >= 0; iter--) {
            for (int y = ny - iterations - 1 + iter; y < ny - 1; y++) {
                for (int x = 1; x < nx - 1; x += 4) {
                    calculate(u[prevIndex], u[currIndex], p, 
                        y, x, actual_nx, 
                        m_hxrec, m_hyrec, m_tau);
                }

                if (y == sy) {
                    u[prevIndex][sy * actual_nx + sx] += tau * tau * f(i + 1, tau);                
                }
            }

            std::swap(currIndex, prevIndex);
        }*/
            
        std::cout << i << std::endl;
        double maxElement = 0;
        for (int j = 0; j < nx * ny; j++) {
            if (maxElement < u[prevIndex][j]) {
                maxElement = u[prevIndex][j];
            }
        }
        std::cout << maxElement << std::endl;
    }

    if (iterations % 2 == 1 && (nt / iterations) % 2 == 1) {
        std::swap(prevIndex, currIndex);
    }

    for (int i = (nt / iterations) * iterations; i < nt; i++) {
        for (int y = 1; y < ny - 1; y++) {
            for (int x = 1; x < nx - 1; x += 4) {
                calculate(u[prevIndex], u[currIndex], p, 
                    y, x, actual_nx, 
                    m_hxrec, m_hyrec, m_tau);
            }

            if (y == sy) {
                u[prevIndex][sy * actual_nx + sx] += tau * tau * f(i, tau);                
            }
        }

        std::swap(prevIndex, currIndex);
    }

    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << duration.count() / 1000.0 << "s" << std::endl;

    FILE* file = std::fopen("./main.dat", "w");
    for (int y = 0; y < ny; y++) {
        std::fwrite(&u[currIndex][y * actual_nx], sizeof(double), nx, file);        
    }
    std::fclose(file);

    return EXIT_SUCCESS;
}