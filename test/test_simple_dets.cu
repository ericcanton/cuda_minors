#include <iostream>
#include "device_minors.cuh"
#include <thrust/device_vector.h>

void test_2x2_identity()
{
    thrust::device_vector<float> A(4, 0);
    // there are 1 minors of size 2x2
    thrust::device_vector<float> minors(1);

    A[0] = 1.0;
    A[3] = 1.0;

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
            std::cout << A[idx(i, j, 2)] << " ";
    }

    minors2xN_device(2, 2, minors, A, 2);

    thrust::device_vector<float> minors_expected(1, 1.0);

    assert(minors == minors_expected);
}

int main()
{
    test_2x2_identity();
}