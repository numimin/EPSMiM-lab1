#include <cstdlib>
#include <iostream>

#include <smmintrin.h>
#include <immintrin.h>

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

int main() {
	/*double* a = (double*) aligned_alloc(32, 4 * sizeof(double));
	double* b = (double*) aligned_alloc(32, 4 * sizeof(double));
	for (int i = 0; i < 4; i++) {
		a[i] = i;
		b[i] = 4 + i;
	}
	__m256d va = *(__m256d*)a;
	__m256d vb = *(__m256d*)b;
	double* c = (double*) aligned_alloc(32, 4 * sizeof(double));
	*((__m256d*)c) = shift(va, vb);
	for (int i = 0; i < 4; i++) {
		std::cout << c[i] << std::endl;
	}*/

	double* a = (double*) aligned_alloc(32, 4 * sizeof(double));
	double* p0 = (double*) aligned_alloc(32, 4 * sizeof(double));
	double* p1 = (double*) aligned_alloc(32, 4 * sizeof(double));
	for (int i = 0; i < 4; i++) {
		a[i] = i;
		p0[i] = 4 + i;
		p1[i] = 8 + i;
	}

	__m256d va = *(__m256d*)a;
	__m256d* vp0 = (__m256d*)p0;
	__m256d* vp1 = (__m256d*)p1;
	gather(vp0, vp1, va);
	for (int i = 0; i < 4; i++) {
		std::cout << p0[i] << std::endl;
	}
	for (int i = 0; i < 4; i++) {
		std::cout << p1[i] << std::endl;
	}
}