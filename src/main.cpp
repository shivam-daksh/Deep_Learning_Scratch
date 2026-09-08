#include <iostream>
#include <vector>
#include <stdexcept>

#include "Tensor/TensorStorage.hh"

void printTest(const std::string& name, bool passed)
{
    std::cout << (passed ? "[PASS] " : "[FAIL] ")
              << name << '\n';
}

int main()
{
    try
    {
        // ============================================================
        // TEST 1: Normal element-wise addition
        // [2,3] + [2,3]
        // ============================================================

        Tensor A({2, 3});
        Tensor B({2, 3});

        A({0,0}) = 1;
        A({0,1}) = 2;
        A({0,2}) = 3;
        A({1,0}) = 4;
        A({1,1}) = 5;
        A({1,2}) = 6;

        B({0,0}) = 10;
        B({0,1}) = 20;
        B({0,2}) = 30;
        B({1,0}) = 40;
        B({1,1}) = 50;
        B({1,2}) = 60;

        Tensor C = A + B;

        bool test1 =
            C({0,0}) == 11 &&
            C({0,1}) == 22 &&
            C({0,2}) == 33 &&
            C({1,0}) == 44 &&
            C({1,1}) == 55 &&
            C({1,2}) == 66;

        printTest("Same-shape addition [2,3] + [2,3]", test1);


        // ============================================================
        // TEST 2: Both tensors broadcast
        // [2,1] + [1,3]
        //
        // A:
        // 10
        // 20
        //
        // B:
        // 1  2  3
        //
        // Result:
        // 11 12 13
        // 21 22 23
        // ============================================================

        Tensor D({2, 1});
        Tensor E({1, 3});

        D({0,0}) = 10;
        D({1,0}) = 20;

        E({0,0}) = 1;
        E({0,1}) = 2;
        E({0,2}) = 3;

        Tensor F = D + E;

        bool test2 =
            F({0,0}) == 11 &&
            F({0,1}) == 12 &&
            F({0,2}) == 13 &&
            F({1,0}) == 21 &&
            F({1,1}) == 22 &&
            F({1,2}) == 23;

        printTest("Two-sided broadcasting [2,1] + [1,3]", test2);


        // ============================================================
        // TEST 3: Lower-rank broadcasting
        // [2,3] + [3]
        //
        // A:
        // 1 2 3
        // 4 5 6
        //
        // B:
        // 10 20 30
        //
        // Result:
        // 11 22 33
        // 14 25 36
        // ============================================================

        Tensor G({2, 3});
        Tensor H({3});

        G({0,0}) = 1;
        G({0,1}) = 2;
        G({0,2}) = 3;
        G({1,0}) = 4;
        G({1,1}) = 5;
        G({1,2}) = 6;

        H({0}) = 10;
        H({1}) = 20;
        H({2}) = 30;

        Tensor I = G + H;

        bool test3 =
            I({0,0}) == 11 &&
            I({0,1}) == 22 &&
            I({0,2}) == 33 &&
            I({1,0}) == 14 &&
            I({1,1}) == 25 &&
            I({1,2}) == 36;

        printTest("Lower-rank broadcasting [2,3] + [3]", test3);


        // ============================================================
        // TEST 4: Higher dimensional broadcasting
        // [2,3,4] + [3,4]
        //
        // We don't need to fill every element manually.
        // Just test selected positions.
        // ============================================================

        Tensor J({2, 3, 4});
        Tensor K({3, 4});

        // Fill J with 1
        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 3; ++j)
                for (size_t k = 0; k < 4; ++k)
                    J({i,j,k}) = 1;

        // Fill K with 10
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 4; ++j)
                K({i,j}) = 10;

        Tensor L = J + K;

        bool test4 =
            L({0,0,0}) == 11 &&
            L({0,2,3}) == 11 &&
            L({1,0,0}) == 11 &&
            L({1,2,3}) == 11;

        printTest("Higher-rank broadcasting [2,3,4] + [3,4]", test4);


        // ============================================================
        // TEST 5: Broadcasting subtraction
        // [2,1] - [1,3]
        // ============================================================

        Tensor M({2,1});
        Tensor N({1,3});

        M({0,0}) = 10;
        M({1,0}) = 20;

        N({0,0}) = 1;
        N({0,1}) = 2;
        N({0,2}) = 3;

        Tensor O = M - N;

        bool test5 =
            O({0,0}) == 9 &&
            O({0,1}) == 8 &&
            O({0,2}) == 7 &&
            O({1,0}) == 19 &&
            O({1,1}) == 18 &&
            O({1,2}) == 17;

        printTest("Broadcasting subtraction [2,1] - [1,3]", test5);


        // ============================================================
        // TEST 6: Incompatible shapes
        // [2,3] + [2,4]
        //
        // This should throw an exception.
        // ============================================================

        Tensor P({2,3});
        Tensor Q({2,4});

        bool test6 = false;

        try
        {
            Tensor R = P + Q;
        }
        catch (const std::invalid_argument&)
        {
            test6 = true;
        }

        printTest("Reject incompatible shapes [2,3] + [2,4]", test6);


        // ============================================================
        // TEST 7: Scalar + tensor
        // ============================================================

        Tensor scalar(std::vector<size_t>{});   // scalar
        Tensor S({2,3});

        scalar(std::vector<size_t>{}) = 5;

        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 3; ++j)
                S({i,j}) = static_cast<double>(i * 3 + j);

        Tensor T = scalar + S;

        bool test7 =
            T({0,0}) == 5 &&
            T({0,1}) == 6 &&
            T({0,2}) == 7 &&
            T({1,0}) == 8 &&
            T({1,1}) == 9 &&
            T({1,2}) == 10;

        printTest("Scalar + [2,3]", test7);


        std::cout << "\nAll tests completed.\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "\nUnexpected exception: "
                  << e.what() << '\n';

        return 1;
    }

    return 0;
}