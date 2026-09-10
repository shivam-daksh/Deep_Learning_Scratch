#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>

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
                U({i,j}) = static_cast<double>(i * 3 + j + 1);

        V({0,0}) = 10;
        V({0,1}) = 20;
        V({0,2}) = 30;
        V({1,0}) = 40;
        V({1,1}) = 50;
        V({1,2}) = 60;

        Tensor W = U * V;

        bool test8 =
            W({0,0}) == 10 &&
            W({0,1}) == 40 &&
            W({0,2}) == 90 &&
            W({1,0}) == 160 &&
            W({1,1}) == 250 &&
            W({1,2}) == 360;

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
        AD({0,1}) = 20;
        AD({0,2}) = 30;
        AD({1,0}) = 40;
        AD({1,1}) = 50;
        AD({1,2}) = 60;

        AE({0}) = 2;
        AE({1}) = 5;
        AE({2}) = 10;

        Tensor AF = AD / AE;

        bool test11 =
            AF({0,0}) == 5 &&
            AF({0,1}) == 4 &&
            AF({0,2}) == 3 &&
            AF({1,0}) == 20 &&
            AF({1,1}) == 10 &&
            AF({1,2}) == 6;

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

        printTest("Division by zero throws", test12);


        // ============================================================
        // TEST 13: Sum of all elements along axis 0
        // [2,3]
        //
        // [1 2 3]
        // [4 5 6]
        //
        // -> [5 7 9]
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
        // [1 2 3]
        // [4 5 6]
        //
        // -> [6, 15]
        // ============================================================

        Tensor AL = AJ.sum(1);

        bool test14 =
            AL({0}) == 6 &&
            AL({1}) == 15;

        printTest("Sum [2,3] along axis 1", test14);


        // ============================================================
        // TEST 15: Higher-dimensional reduction
        // [2,3,4]
        // ============================================================

        Tensor AM({2,3,4});

        size_t value = 0;

        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 3; ++j)
                for (size_t k = 0; k < 4; ++k)
                    AM({i,j,k}) = static_cast<double>(value++);

        Tensor AN = AM.sum(1);

        bool test15 =
            AN({0,0}) == 12 &&
            AN({0,1}) == 15 &&
            AN({0,2}) == 18 &&
            AN({0,3}) == 21 &&
            AN({1,0}) == 48 &&
            AN({1,1}) == 51 &&
            AN({1,2}) == 54 &&
            AN({1,3}) == 57;

        printTest("Higher-dimensional sum [2,3,4] along axis 1", test15);


        // ============================================================
        // TEST 16: Invalid reduction axis
        // ============================================================

        bool test16 = false;

        try
        {
            Tensor AO({2,3,4});
            Tensor AP = AO.sum(3);
        }
        catch (const std::out_of_range&)
        {
            test16 = true;
        }

        printTest("Reject invalid reduction axis", test16);


        // ============================================================
        // TEST 17: Reduction of zero-sized axis
        // ============================================================

        bool test17 = false;

        try
        {
            Tensor AQ({2,0,4});
            Tensor AR = AQ.sum(1);
        }
        catch (const std::invalid_argument&)
        {
            test17 = true;
        }

        printTest("Reject reduction of zero-sized axis", test17);


        // ============================================================
        // TEST 18: Mean along axis 0
        // [2,3]
        //
        // [1 2 3]
        // [4 5 6]
        //
        // -> [2.5 3.5 4.5]
        // ============================================================

        Tensor AS({2,3});

        AS({0,0}) = 1;
        AS({0,1}) = 2;
        AS({0,2}) = 3;
        AS({1,0}) = 4;
        AS({1,1}) = 5;
        AS({1,2}) = 6;

        Tensor AT = AS.mean(0);

        bool test18 =
            AT({0}) == 2.5 &&
            AT({1}) == 3.5 &&
            AT({2}) == 4.5;

        printTest("Mean [2,3] along axis 0", test18);


        // ============================================================
        // TEST 19: Mean along axis 1
        //
        // [1 2 3]
        // [4 5 6]
        //
        // -> [2, 5]
        // ============================================================

        Tensor AU = AS.mean(1);

        bool test19 =
            AU({0}) == 2.0 &&
            AU({1}) == 5.0;

        printTest("Mean [2,3] along axis 1", test19);


        // ============================================================
        // TEST 20: Mean along axis 0 with keepdim
        // -> shape [1, 3]
        // ============================================================

        Tensor AV = AS.mean(0, true);

        bool test20 =
            AV({0,0}) == 2.5 &&
            AV({0,1}) == 3.5 &&
            AV({0,2}) == 4.5;

        printTest("Mean [2,3] along axis 0 (keepdim)", test20);


        // ============================================================
        // TEST 21: Invalid mean axis
        // ============================================================

        bool test21 = false;

        try
        {
            Tensor AW({2,3,4});
            Tensor AX = AW.mean(3);
        }
        catch (const std::out_of_range&)
        {
            test21 = true;
        }

        printTest("Reject invalid mean axis", test21);


        // ============================================================
        // TEST 22: Mean of zero-sized axis
        // ============================================================

        bool test22 = false;

        try
        {
            Tensor AY({2,0,4});
            Tensor AZ = AY.mean(1);
        }
        catch (const std::invalid_argument&)
        {
            test22 = true;
        }

        printTest("Reject mean of zero-sized axis", test22);


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