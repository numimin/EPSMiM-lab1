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
const __m256d one_hudredth = _mm256_set_pd(0.01, 0.01, 0.01, 0.01);
const __m256d zeros = _mm256_set_pd(0, 0, 0, 0);

__m256d shift(__m256d a, __m256d b) {
    __m128d a0 = *(__m128d*)&a;
    __m128d a1 = *((__m128d*)&a + 1);
    __m128d b0 = *(__m128d*)&b;
    __m256d c;
    __m128d* cPtr = (__m128d*)&c;
    cPtr[0] = _mm_shuffle_pd(a0, a1, 5); 
    cPtr[1] = _mm_shuffle_pd(a1, b0, 1); 
    return c;
}

__m256d shift_r(__m256d a, __m256d b) {
    __m128d a0 = *(__m128d*)&a;
    __m128d a1 = *((__m128d*)&a + 1);
    __m128d b0 = *(__m128d*)&b;
    __m256d c;
    __m128d* cPtr = (__m128d*)&c;
    cPtr[0] = a1; 
    cPtr[1] = b0; 
    return c;
}

void gather(__m256d* p0, __m256d* p1, __m256d a) {
    __m128d p00 = *(__m128d*)p0;
    __m128d p10 = *(__m128d*)p1;
    __m128d p11 = *((__m128d*)p1 + 1);
    __m128d a0 = *(__m128d*)&a;
    __m128d a1 = *((__m128d*)&a + 1);
    __m128d* p0Ptr = (__m128d*)p0;
    __m128d* p1Ptr = (__m128d*)p1;
    p0Ptr[0] = _mm_shuffle_pd(p00, a0, 4);
    p0Ptr[1] = _mm_shuffle_pd(a0, a1, 1);
    p1Ptr[0] = _mm_shuffle_pd(a1, p10, 3);
}

//All input parameters are aligned
//We are aligned + 1(double)
__m256d calculate(__m256d* prev0, __m256d* prev1, 
    __m256d curr0, __m256d curr1, 
    __m256d curr_top0, __m256d curr_top1,
     __m256d curr_down0, __m256d curr_down1,
    __m256d p0, __m256d p1,
    __m256d p_down0, __m256d p_down1,
    __m256d m_hxrec, __m256d m_hyrec, __m256d m_tau) 
{
    __m256d uPrev = shift(*prev0, *prev1);
    __m256d uc = shift(curr0, curr1);
    __m256d ur = shift_r(curr0, curr1);
    __m256d ul = curr0;
    __m256d ut = shift(curr_top0, curr_top1);
    __m256d ud = shift(curr_down0, curr_down1);
    __m256d pc = shift(p0, p1);
    __m256d pdl = p_down0;
    __m256d pd = shift(p_down0, p_down1);
    __m256d pl = p0;

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
    //_mm256_storeu_pd(prevOut, result);
    gather(prev0, prev1, result);

    __m256d minus_result = _mm256_sub_pd(zeros, result);
    __m256d abs_result = _mm256_max_pd(result, minus_result);

    __m256d permuted_result = _mm256_permute2f128_pd(abs_result, abs_result, 1);
    __m256d m1 = _mm256_max_pd(abs_result, permuted_result);
    __m256d m2 = _mm256_permute_pd(m1, 5);
    return _mm256_max_pd(m1, m2);
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

    const int actual_nx = ((nx + 8) / 4) * 4;

    double* u0 = (double*) aligned_alloc(32, actual_nx * ny * sizeof(double));
    double* u1 = (double*) aligned_alloc(32, actual_nx * ny * sizeof(double));
    double* u[] {u0, u1};

    for (int i = 0; i < actual_nx * ny; i++) {
        u0[i] = 0;
        u1[i] = 0;
    }

    double* p = (double*) aligned_alloc(32, actual_nx * ny * sizeof(double));

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            if (x < nx / 2) {
                p[y * actual_nx + x] = 0.1 * 0.1;
            } else {
                p[y * actual_nx + x] = 0.2 * 0.2;
            }
        }
    }

    for (int i = 0; i < nt; i++) {
        double maxElement = 0;
        for (int y = 1; y < ny - 1; y++) {
            int vectorIndex = 0;
            int otherVectorIndex = 1;
            __m256d* prev[] {(__m256d*)&u[prevIndex][y * actual_nx], NULL};
            __m256d curr[] {*(__m256d*)&u[currIndex][y * actual_nx], zeros};
            __m256d curr_top[] {*(__m256d*)&u[currIndex][(y + 1) * actual_nx], zeros};
            __m256d curr_down[] {*(__m256d*)&u[currIndex][(y - 1) * actual_nx], zeros};
            __m256d p_center[] {*(__m256d*)&p[y * actual_nx], zeros};
            __m256d p_down[] {*(__m256d*)&p[(y - 1) * actual_nx], zeros};
            for (int x = 1; x < nx - 1; x += 4) {
                prev[otherVectorIndex] = (__m256d*)&u[prevIndex][y * actual_nx + x + 3];
                curr[otherVectorIndex] = *(__m256d*)&u[currIndex][y * actual_nx + x + 3];
                curr_top[otherVectorIndex] = *(__m256d*)&u[currIndex][(y + 1) * actual_nx + x + 3];
                curr_down[otherVectorIndex] = *(__m256d*)&u[currIndex][(y - 1) * actual_nx + x + 3];
                p_center[otherVectorIndex] = *(__m256d*)&p[y * actual_nx + x + 3];
                p_down[otherVectorIndex] = *(__m256d*)&p[(y - 1) * actual_nx + x + 3];
                __m256d newElement = calculate(
                        prev[vectorIndex], prev[otherVectorIndex],
                        curr[vectorIndex], curr[otherVectorIndex],
                        curr_top[vectorIndex], curr_top[otherVectorIndex],
                        curr_down[vectorIndex], curr_down[otherVectorIndex],
                        p_center[vectorIndex], p_center[otherVectorIndex],
                        p_down[vectorIndex], p_down[otherVectorIndex],
                        m_hxrec, m_hyrec, m_tau);
                maxElement = std::max(maxElement, *((double*)&newElement));
                std::swap(vectorIndex, otherVectorIndex);
            }
            u[prevIndex][y * actual_nx + nx - 1] = 0;
        }

        u[prevIndex][sy * actual_nx + sx] += tau * tau * f(i, tau);

        std::cout << i << std::endl;
        std::cout << maxElement << std::endl;

        std::swap(currIndex, prevIndex);
    }

    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << duration.count() / 1000.0 << "s" << std::endl;

    FILE* file = std::fopen("./main.dat", "w");
    for (int y = 0; y < ny; y++) {
        std::fwrite(&u[currIndex][y * actual_nx], sizeof(double), nx, file);        
    }
    std::fclose(file);

    free(u0);
    free(u1);
    free(p);

    return EXIT_SUCCESS;
}