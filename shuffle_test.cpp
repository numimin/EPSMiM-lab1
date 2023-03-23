#include <cstdlib>
#include <iostream>

#include <smmintrin.h>
#include <immintrin.h>

template<int Shift>
__m256d shift(__m256d a, __m256d b) {
	const auto c = _mm256_permute4x64_pd(a, 57); //9(1200), 57 (1230)
	const auto d = _mm256_permute4x64_pd(b, 0); //0(4444)
	return _mm256_blend_pd(c, d, 8);
}

template<int Shift>
__m256d shift_r(__m256d a, __m256d b) {
	//auto c = _mm256_permute4x64_pd(a, 190);
	//auto d = _mm256_permute4x64_pd(b, 64);
	//return _mm256_blend_pd(c, d, 12);
	auto c _mm256_permute4x64_pd(b, 147); //b = a, a = p0
	auto final0 = _mm256_blend_pd(a, c, 14);
	auto d = _mm256_permute4x64_pd(a, 255); //a = a, b = p1
	auto final1 = _mm256_blend_pd(d, b, 14);
	return final1;
}

struct Pair {
	__m256d p0;
	__m256d p1;
};

template<int Shift>
Pair gather(__m256d* p0, __m256d* p1, __m256d a) {
	/*__m128d p00 = *(__m128d*)p0;
	__m128d p10 = *(__m128d*)p1;
	__m128d p11 = *((__m128d*)p1 + 1);
	__m128d a0 = *(__m128d*)&a;
	__m128d a1 = *((__m128d*)&a + 1);
	__m128d* p0Ptr = (__m128d*)p0;
	__m128d* p1Ptr = (__m128d*)p1;
	p0Ptr[0] = _mm_shuffle_pd(p00, a0, 4);
	p0Ptr[1] = _mm_shuffle_pd(a0, a1, 1);
	p1Ptr[0] = _mm_shuffle_pd(a1, p10, 3);*/
	/*__m256d c = _mm256_permute4x64_pd(a, 39);
	double* cPtr = (double*) &c;
	for (int i = 0; i < 4; i++) {
		std::cout << cPtr[i] << std::endl;
	}*/
	return {
		_mm256_permute4x64_pd(a, Shift),
		_mm256_permute4x64_pd(a, 39)
	};
}

template<int Shift>
void iterate_gather(__m256d* p0, __m256d* p1, __m256d a) {
	std::cout << "Iteration " << Shift << std::endl;
	double* dp0 = (double*) p0;
	double* dp1 = (double*) p1;
	for (int i = 0; i < 4; i++) {
		dp0[i] = 4 + i;
		dp1[i] = 8 + i;
	}
	auto pair = gather<Shift>(p0, p1, a);
	dp0 = (double*) &pair.p0;
	dp1 = (double*) &pair.p1;
	for (int i = 0; i < 4; i++) {
		std::cout << dp0[i] << std::endl;
	}
	for (int i = 0; i < 4; i++) {
		std::cout << dp1[i] << std::endl;
	}
	iterate_gather<Shift - 1>(p0, p1, a);
}

template<>
void iterate_gather<0>(__m256d* p0, __m256d* p1, __m256d a) {
	std::cout << "Iteration " << 0 << std::endl;
	double* dp0 = (double*) p0;
	double* dp1 = (double*) p1;
	for (int i = 0; i < 4; i++) {
		dp0[i] = 4 + i;
		dp1[i] = 8 + i;
	}
	auto pair = gather<0>(p0, p1, a);
	dp0 = (double*) &pair.p0;
	dp1 = (double*) &pair.p1;
	for (int i = 0; i < 4; i++) {
		std::cout << dp0[i] << std::endl;
	}
	for (int i = 0; i < 4; i++) {
		std::cout << dp1[i] << std::endl;
	}
}

template<int Shift>
void iterate(__m256d a, __m256d b) {
	double* c = (double*) aligned_alloc(32, 4 * sizeof(double));
	*((__m256d*)c) = shift_r<Shift>(a, b);
	std::cout << "Iteration " << Shift << std::endl;
	for (int i = 0; i < 4; i++) {
		std::cout << c[i] << std::endl;
	}
	iterate<Shift - 1>(a, b);
}

template<>
void iterate<0>(__m256d a, __m256d b) {
	double* c = (double*) aligned_alloc(32, 4 * sizeof(double));
	*((__m256d*)c) = shift_r<0>(a, b);
	std::cout << "Iteration " << 0 << std::endl;
	for (int i = 0; i < 4; i++) {
		std::cout << c[i] << std::endl;
	}
}

int main() {
	double* a = (double*) aligned_alloc(32, 4 * sizeof(double));
	double* b = (double*) aligned_alloc(32, 4 * sizeof(double));
	for (int i = 0; i < 4; i++) {
		a[i] = i;
		b[i] = 4 + i;
	}
	__m256d va = *(__m256d*)a;
	__m256d vb = *(__m256d*)b;
	iterate<15>(va, vb);

	/*double* a = (double*) aligned_alloc(32, 4 * sizeof(double));
	double* p0 = (double*) aligned_alloc(32, 4 * sizeof(double));
	double* p1 = (double*) aligned_alloc(32, 4 * sizeof(double));

	__m256d va = *(__m256d*)a;
	__m256d* vp0 = (__m256d*)p0;
	__m256d* vp1 = (__m256d*)p1;
	iterate_gather<255>(vp0, vp1, va);*/
}