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
        // ============================================================

        Tensor J({2, 3, 4});
        Tensor K({3, 4});

        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 3; ++j)
                for (size_t k = 0; k < 4; ++k)
                    J({i,j,k}) = 1;

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

        Tensor scalar(std::vector<size_t>{});
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


        // ============================================================
        // TEST 8: Same-shape element-wise multiplication
        // ============================================================

        Tensor U({2,3});
        Tensor V({2,3});

        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 3; ++j)
            {
                U({i,j}) = static_cast<double>(i * 3 + j + 1);
                V({i,j}) = 2;
            }

        Tensor W = U * V;

        bool test8 =
            W({0,0}) == 2 &&
            W({0,1}) == 4 &&
            W({0,2}) == 6 &&
            W({1,0}) == 8 &&
            W({1,1}) == 10 &&
            W({1,2}) == 12;

        printTest("Same-shape multiplication [2,3] * [2,3]", test8);


        // ============================================================
        // TEST 9: Broadcasting multiplication
        // [2,3] * [3]
        // ============================================================

        Tensor X({2,3});
        Tensor Y({3});

        X({0,0}) = 1;
        X({0,1}) = 2;
        X({0,2}) = 3;
        X({1,0}) = 4;
        X({1,1}) = 5;
        X({1,2}) = 6;

        Y({0}) = 10;
        Y({1}) = 20;
        Y({2}) = 30;

        Tensor Z = X * Y;

        bool test9 =
            Z({0,0}) == 10 &&
            Z({0,1}) == 40 &&
            Z({0,2}) == 90 &&
            Z({1,0}) == 40 &&
            Z({1,1}) == 100 &&
            Z({1,2}) == 180;

        printTest("Broadcasting multiplication [2,3] * [3]", test9);


        // ============================================================
        // TEST 10: Same-shape division
        // ============================================================

        Tensor AA({2,3});
        Tensor AB({2,3});

        AA({0,0}) = 10;
        AA({0,1}) = 20;
        AA({0,2}) = 30;
        AA({1,0}) = 40;
        AA({1,1}) = 50;
        AA({1,2}) = 60;

        AB({0,0}) = 2;
        AB({0,1}) = 4;
        AB({0,2}) = 5;
        AB({1,0}) = 8;
        AB({1,1}) = 10;
        AB({1,2}) = 20;

        Tensor AC = AA / AB;

        bool test10 =
            AC({0,0}) == 5 &&
            AC({0,1}) == 5 &&
            AC({0,2}) == 6 &&
            AC({1,0}) == 5 &&
            AC({1,1}) == 5 &&
            AC({1,2}) == 3;

        printTest("Same-shape division [2,3] / [2,3]", test10);


        // ============================================================
        // TEST 11: Broadcasting division
        // [2,3] / [3]
        // ============================================================

        Tensor AD({2,3});
        Tensor AE({3});

        AD({0,0}) = 10;
        AD({0,1}) = 40;
        AD({0,2}) = 90;
        AD({1,0}) = 20;
        AD({1,1}) = 60;
        AD({1,2}) = 120;

        AE({0}) = 10;
        AE({1}) = 20;
        AE({2}) = 30;

        Tensor AF = AD / AE;

        bool test11 =
            AF({0,0}) == 1 &&
            AF({0,1}) == 2 &&
            AF({0,2}) == 3 &&
            AF({1,0}) == 2 &&
            AF({1,1}) == 3 &&
            AF({1,2}) == 4;

        printTest("Broadcasting division [2,3] / [3]", test11);


        // ============================================================
        // TEST 12: Division by zero
        // ============================================================

        Tensor AG({3});
        Tensor AH({3});

        AG({0}) = 10;
        AG({1}) = 20;
        AG({2}) = 30;

        AH({0}) = 2;
        AH({1}) = 0;
        AH({2}) = 5;

        bool test12 = false;

        try
        {
            Tensor AI = AG / AH;
        }
        catch (const std::domain_error&)
        {
            test12 = true;
        }

        printTest("Division by zero throws exception", test12);


        // ============================================================
        // TEST 13: Sum all elements along axis 0
        //
        // [2,3]
        //
        // 1 2 3
        // 4 5 6
        //
        // sum(0) -> [5,7,9]
        // ============================================================

        Tensor AJ({2,3});

        AJ({0,0}) = 1;
        AJ({0,1}) = 2;
        AJ({0,2}) = 3;
        AJ({1,0}) = 4;
        AJ({1,1}) = 5;
        AJ({1,2}) = 6;

        Tensor AK = AJ.sum(0);

        bool test13 =
            AK({0}) == 5 &&
            AK({1}) == 7 &&
            AK({2}) == 9;

        printTest("Sum [2,3] along axis 0", test13);


        // ============================================================
        // TEST 14: Sum along axis 1
        //
        // 1 2 3
        // 4 5 6
        //
        // sum(1) -> [6,15]
        // ============================================================

        Tensor AL = AJ.sum(1);

        bool test14 =
            AL({0}) == 6 &&
            AL({1}) == 15;

        printTest("Sum [2,3] along axis 1", test14);


        // ============================================================
        // TEST 15: 3D sum along axis 0
        // [2,3,4] -> [3,4]
        // ============================================================

        Tensor AM({2,3,4});

        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 3; ++j)
                for (size_t k = 0; k < 4; ++k)
                    AM({i,j,k}) =
                        static_cast<double>(i * 12 + j * 4 + k + 1);

        Tensor AN = AM.sum(0);

        bool test15 =
            AN({0,0}) == 14 &&
            AN({0,1}) == 16 &&
            AN({0,2}) == 18 &&
            AN({0,3}) == 20 &&
            AN({2,0}) == 30 &&
            AN({2,3}) == 36;

        printTest("3D sum [2,3,4] along axis 0", test15);


        // ============================================================
        // TEST 16: 3D sum along axis 1
        // [2,3,4] -> [2,4]
        // ============================================================

        Tensor AO = AM.sum(1);

        bool test16 =
            AO({0,0}) == 15 &&
            AO({0,1}) == 18 &&
            AO({0,2}) == 21 &&
            AO({0,3}) == 24 &&
            AO({1,0}) == 51 &&
            AO({1,3}) == 60;

        printTest("3D sum [2,3,4] along axis 1", test16);


        // ============================================================
        // TEST 17: 3D sum along axis 2
        // [2,3,4] -> [2,3]
        // ============================================================

        Tensor AP = AM.sum(2);

        bool test17 =
            AP({0,0}) == 10 &&
            AP({0,1}) == 26 &&
            AP({0,2}) == 42 &&
            AP({1,0}) == 58 &&
            AP({1,1}) == 74 &&
            AP({1,2}) == 90;

        printTest("3D sum [2,3,4] along axis 2", test17);


        // ============================================================
        // TEST 18: Invalid reduction axis
        // ============================================================

        bool test18 = false;

        try
        {
            Tensor AQ({2,3,4});
            Tensor AR = AQ.sum(3);
        }
        catch (const std::out_of_range&)
        {
            test18 = true;
        }

        printTest("Reject invalid reduction axis", test18);


        // ============================================================
        // TEST 19: Zero-sized reduction axis
        // ============================================================

        bool test19 = false;

        try
        {
            Tensor AS({2,0,4});
            Tensor AT = AS.sum(1);
        }
        catch (const std::invalid_argument&)
        {
            test19 = true;
        }

        printTest("Reject reduction over zero-sized axis", test19);


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