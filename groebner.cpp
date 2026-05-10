#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <omp.h>
#include <iomanip>

using namespace std;

const int NUM_INTS = 32;
const int NUM_COLS = NUM_INTS * 32;
const int NUM_ROWS = 1024;

struct BitRow {
    unsigned int data[NUM_INTS] = {0};

    int getLeadingBitPosition() const {
        for (int i = NUM_INTS - 1; i >= 0; --i) {
            if (data[i] != 0) {
                for (int b = 31; b >= 0; --b) {
                    if (data[i] & (1U << b)) return i * 32 + b;
                }
            }
        }
        return -1; 
    }

    bool isZero() const {
        for (int i = 0; i < NUM_INTS; ++i) {
            if (data[i] != 0) return false;
        }
        return true;
    }

    void xorWith(const BitRow& other) {
        #pragma omp simd
        for (int i = 0; i < NUM_INTS; ++i) {
            data[i] ^= other.data[i];
        }
    }
};

void generateSparseRow(BitRow& row) {
    int numBits = rand() % 20 + 5; 
    for (int i = 0; i < numBits; ++i) {
        int pos = rand() % NUM_COLS;
        row.data[pos / 32] |= (1U << (pos % 32));
    }
}

int main() {
    system("chcp 65001 > nul");
    srand(42); 

    cout << NUM_ROWS << " x " << NUM_COLS << endl;

    int test_runs = 100; // 跑 100 次取平均
    double total_time = 0;
    int active_eliminators = 0;

    for (int r = 0; r < test_runs; ++r) {
        // 每次重新初始化数据
        vector<BitRow> eliminatees(NUM_ROWS);
        for (int i = 0; i < NUM_ROWS; ++i) generateSparseRow(eliminatees[i]);
        BitRow eliminators[NUM_COLS];

        auto start = chrono::high_resolution_clock::now();

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < NUM_ROWS; ++i) {
            BitRow& row = eliminatees[i];
            while (!row.isZero()) {
                int lp = row.getLeadingBitPosition();
                if (!eliminators[lp].isZero()) {
                    row.xorWith(eliminators[lp]); 
                } else {
                    bool upgraded = false;
                    #pragma omp critical
                    {
                        if (eliminators[lp].isZero()) {
                            eliminators[lp] = row;
                            upgraded = true;
                        }
                    }
                    if (upgraded) break;
                }
            }
        }

        auto end = chrono::high_resolution_clock::now();
        total_time += chrono::duration<double, milli>(end - start).count();

        if (r == 0) { // 只统计第一次的结果用于验证
            for (int i = 0; i < NUM_COLS; ++i) {
                if (!eliminators[i].isZero()) active_eliminators++;
            }
        }
    }

    cout << "平均耗时: " << fixed << setprecision(4) << total_time / test_runs << " ms" << endl;
    cout << "Total Eliminators : " << active_eliminators << " / " << NUM_COLS << endl;

    return 0;
}