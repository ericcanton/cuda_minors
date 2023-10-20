#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include "device_minors.cuh"

int main()
{
    int m = 2, n = 2;
    int A[m * n] = {0};
    int minors[1] = {1};
    A[0] = 1;
    A[3] = 1;
    std::cout << "A: " << std::endl;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            std::cout << A[idx(i, j, 2)] << " ";
    std::cout << "===" << std::endl;

    // allocate device memory, copy A to device

    minors2xN_device(5, 5, minors, A, n);
    std::cout << "Got this minor from device: " << minors[0] << std::endl;

    thrust::device_vector<int> minors_expected(1, 1.0);
}