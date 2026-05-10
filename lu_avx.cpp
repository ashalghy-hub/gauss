#include <iostream>
#include <chrono>
#include <immintrin.h>
#include <cstdlib>

using namespace std;

const int N = 512;
alignas(32) float m_unaligned[N][N];
alignas(32) float m_aligned[N][N];

void init_matrix(float m[][N]) {
    for (int i = 0; i < N; i++) {
        m[i][i] = 1.0;
        for (int j = i + 1; j < N; j++) m[i][j] = rand() % 10;
        for (int j = 0; j < i; j++) m[i][j] = 0;
    }
    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            for (int j = 0; j < N; j++) m[i][j] += m[k][j];
        }
    }
}

//不对齐版本
void LU_AVX_Unaligned(float m[][N]) {
    for (int k = 0; k < N; ++k) {
        float pivot = m[k][k];
        __m256 vt = _mm256_set1_ps(pivot);
        int j = k + 1;
        for (; j + 8 <= N; j += 8) {
            __m256 va = _mm256_loadu_ps(&m[k][j]); 
            va = _mm256_div_ps(va, vt);
            _mm256_storeu_ps(&m[k][j], va);        
        }
        for (; j < N; ++j) m[k][j] = m[k][j] / pivot;
        m[k][k] = 1.0;

        for (int i = k + 1; i < N; ++i) {
            float factor = m[i][k];
            __m256 vaik = _mm256_set1_ps(factor);
            j = k + 1;
            for (; j + 8 <= N; j += 8) {
                __m256 vakj = _mm256_loadu_ps(&m[k][j]); 
                __m256 vaij = _mm256_loadu_ps(&m[i][j]); 
                __m256 vx = _mm256_mul_ps(vakj, vaik);   
                vaij = _mm256_sub_ps(vaij, vx);          
                _mm256_storeu_ps(&m[i][j], vaij);        
            }
            for (; j < N; ++j) m[i][j] = m[i][j] - factor * m[k][j];
            m[i][k] = 0.0;
        }
    }
}

//严格对齐版本
void LU_AVX_Aligned(float m[][N]) {
    for (int k = 0; k < N; ++k) {
        float pivot = m[k][k];
        __m256 vt = _mm256_set1_ps(pivot);
        
        int j = k + 1;
        while ((j % 8 != 0) && (j < N)) {
            m[k][j] = m[k][j] / pivot;
            j++;
        }
        for (; j + 8 <= N; j += 8) {
            __m256 va = _mm256_load_ps(&m[k][j]); 
            va = _mm256_div_ps(va, vt);
            _mm256_store_ps(&m[k][j], va);        
        }
        for (; j < N; ++j) m[k][j] = m[k][j] / pivot;
        m[k][k] = 1.0;

        for (int i = k + 1; i < N; ++i) {
            float factor = m[i][k];
            __m256 vaik = _mm256_set1_ps(factor);
            
            j = k + 1;
            while ((j % 8 != 0) && (j < N)) {
                m[i][j] = m[i][j] - factor * m[k][j];
                j++;
            }
            for (; j + 8 <= N; j += 8) {
                __m256 vakj = _mm256_load_ps(&m[k][j]); 
                __m256 vaij = _mm256_load_ps(&m[i][j]); 
                __m256 vx = _mm256_mul_ps(vakj, vaik);   
                vaij = _mm256_sub_ps(vaij, vx);          
                _mm256_store_ps(&m[i][j], vaij);        
            }
            for (; j < N; ++j) m[i][j] = m[i][j] - factor * m[k][j];
            m[i][k] = 0.0;
        }
    }
}

int main() {
    system("chcp 65001 > nul");
    int test_runs = 20;

    cout << "手工AVX不对齐版本:" << endl;
    double time_unaligned = 0;
    for (int r = 0; r < test_runs; r++) {
        init_matrix(m_unaligned);
        auto start = chrono::high_resolution_clock::now();
        LU_AVX_Unaligned(m_unaligned);
        auto end = chrono::high_resolution_clock::now();
        time_unaligned += chrono::duration<double, milli>(end - start).count();
    }
    cout << "平均耗时: " << time_unaligned / test_runs << " ms" << endl << endl;

    cout << "手工AVX严格对齐版本:" << endl;
    double time_aligned = 0;
    for (int r = 0; r < test_runs; r++) {
        init_matrix(m_aligned);
        auto start = chrono::high_resolution_clock::now();
        LU_AVX_Aligned(m_aligned);
        auto end = chrono::high_resolution_clock::now();
        time_aligned += chrono::duration<double, milli>(end - start).count();
    }
    cout << "平均耗时: " << time_aligned / test_runs << " ms" << endl;

    return 0;
}