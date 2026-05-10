#include <iostream>
#include <chrono>
#include <cstdlib>

using namespace std;

const int N = 256;
float m[N][N];

void init_matrix() {
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

void LU_Serial() {
    for (int k = 0; k < N; ++k) {
        float pivot = m[k][k];
        for (int j = k + 1; j < N; ++j) {
            m[k][j] = m[k][j] / pivot;
        }
        m[k][k] = 1.0;

        for (int i = k + 1; i < N; ++i) {
            float factor = m[i][k];
            for (int j = k + 1; j < N; ++j) {
                m[i][j] = m[i][j] - factor * m[k][j];
            }
            m[i][k] = 0.0;
        }
    }
}

int main() {
    system("chcp 65001 > nul");
    int test_runs = 20;
    double total_time = 0;

    for (int r = 0; r < test_runs; r++) {
        init_matrix(); // 每次测试前重置矩阵
        auto start = chrono::high_resolution_clock::now();
        
        LU_Serial();
        
        auto end = chrono::high_resolution_clock::now();
        total_time += chrono::duration<double, milli>(end - start).count();
    }
    
    cout << "平均耗时: " << total_time / test_runs << " ms" << endl;
    return 0;
}