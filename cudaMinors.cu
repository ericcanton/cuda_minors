#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include "device_minors.cuh"

int main()
{
    int N = 5;
    int numBlocks = 1;
    while (numBlocks < N)
    {
        numBlocks << 1;
    }
    int numMinors = (5 * 4) / 2;

    // initialize 2 x N matrix with 0..<2N-1
    thrust::device_vector<int> mat(2 * N);
    thrust::sequence(mat.begin(), mat.end());

    thrust::device_vector<int> minors(numMinors);
    thrust::host_vector<int> d_minors_result(numMinors);

    // do the same on the host
    thrust::host_vector<int> h_mat(2 * N);
    thrust::sequence(h_mat.begin(), h_mat.end());

    thrust::host_vector<int> h_minors(numMinors);

    // call the kernel
    minors2byN_kernel<<<numBlocks, numBlocks - 1>>>(minors, mat, N);

    // copy the result back to the host
    thrust::copy(minors.begin(), minors.end(), d_minors_result.begin());

    // do the same on the host
    minors2byN_cpu(h_minors, h_mat, N);

    // print the vectors of minors
    std::cout << "Device: ";
    for (int i = 0; i < numMinors; i++)
    {
        std::cout << d_minors_result[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "CPU: ";
    for (int i = 0; i < numMinors; i++)
    {
        std::cout << h_minors[i] << " ";
    }
    std::cout << std::endl;

    // // check the result
    // for (int i = 0; i < numMinors; i++)
    // {
    //     if (h_minors[i] != d_minors_result[i])
    //     {
    //         std::cout << "Error at index " << i << std::endl;
    //         std::cout << "Host: " << h_minors[i] << std::endl;
    //         std::cout << "Device: " << d_minors_result[i] << std::endl;
    //         return 1;
    //     }
    // }
}