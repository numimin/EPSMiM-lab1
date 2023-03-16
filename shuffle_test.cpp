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

int main() {
	double* a = (double*) aligned_alloc(32, 4 * sizeof(double));
	double* b = (double*) aligned_alloc(32, 4 * sizeof(double));
	for (int i = 0; i < 4; i++) {
		a[i] = i;
		b[i] = 4 + i;
	}
	__m256d va = *(__m256d*)a;
	__m256d vb = *(__m256d*)b;
	double* c = (double*) aligned_alloc(32, 4 * sizeof(double));
	*((__m256d*)c) = shift_r(va, vb);
	for (int i = 0; i < 4; i++) {
		std::cout << c[i] << std::endl;
	}
}