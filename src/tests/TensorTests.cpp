#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>

#include "Tensor/Tensor.hh"


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

        Tensor<double> A({2, 3});
        Tensor<double> B({2, 3});

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

        Tensor<double> C = A + B;

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

        Tensor<double> D({2, 1});
        Tensor<double> E({1, 3});

        D({0,0}) = 10;
        D({1,0}) = 20;

        E({0,0}) = 1;
        E({0,1}) = 2;
        E({0,2}) = 3;

        Tensor<double> F = D + E;

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

        Tensor<double> G({2, 3});
        Tensor<double> H({3});

        G({0,0}) = 1;
        G({0,1}) = 2;
        G({0,2}) = 3;
        G({1,0}) = 4;
        G({1,1}) = 5;
        G({1,2}) = 6;

        H({0}) = 10;
        H({1}) = 20;
        H({2}) = 30;

        Tensor<double> I = G + H;

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

        Tensor<double> J({2, 3, 4});
        Tensor<double> K({3, 4});

        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 3; ++j)
                for (size_t k = 0; k < 4; ++k)
                    J({i,j,k}) = 1;

        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 4; ++j)
                K({i,j}) = 10;

        Tensor<double> L = J + K;

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

        Tensor<double> M({2,1});
        Tensor<double> N({1,3});

        M({0,0}) = 10;
        M({1,0}) = 20;

        N({0,0}) = 1;
        N({0,1}) = 2;
        N({0,2}) = 3;

        Tensor<double> O = M - N;

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

        Tensor<double> P({2,3});
        Tensor<double> Q({2,4});

        bool test6 = false;

        try
        {
            Tensor<double> R = P + Q;
        }
        catch (const std::invalid_argument&)
        {
            test6 = true;
        }

        printTest("Reject incompatible shapes [2,3] + [2,4]", test6);


        // ============================================================
        // TEST 7: Scalar + tensor
        // ============================================================

        Tensor<double> scalar(std::vector<size_t>{});
        Tensor<double> S({2,3});

        scalar(std::vector<size_t>{}) = 5;

        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 3; ++j)
                S({i,j}) = static_cast<double>(i * 3 + j);

        Tensor<double> T = scalar + S;

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

        Tensor<double> U({2,3});
        Tensor<double> V({2,3});

        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 3; ++j)
                U({i,j}) = static_cast<double>(i * 3 + j + 1);

        V({0,0}) = 10;
        V({0,1}) = 20;
        V({0,2}) = 30;
        V({1,0}) = 40;
        V({1,1}) = 50;
        V({1,2}) = 60;

        Tensor<double> W = U * V;

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

        Tensor<double> X({2,3});
        Tensor<double> Y({3});

        X({0,0}) = 1;
        X({0,1}) = 2;
        X({0,2}) = 3;
        X({1,0}) = 4;
        X({1,1}) = 5;
        X({1,2}) = 6;

        Y({0}) = 10;
        Y({1}) = 20;
        Y({2}) = 30;

        Tensor<double> Z = X * Y;

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

        Tensor<double> AA({2,3});
        Tensor<double> AB({2,3});

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

        Tensor<double> AC = AA / AB;

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

        Tensor<double> AD({2,3});
        Tensor<double> AE({3});

        AD({0,0}) = 10;
        AD({0,1}) = 20;
        AD({0,2}) = 30;
        AD({1,0}) = 40;
        AD({1,1}) = 50;
        AD({1,2}) = 60;

        AE({0}) = 2;
        AE({1}) = 5;
        AE({2}) = 10;

        Tensor<double> AF = AD / AE;

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

        Tensor<double> AG({3});
        Tensor<double> AH({3});

        AG({0}) = 10;
        AG({1}) = 20;
        AG({2}) = 30;

        AH({0}) = 2;
        AH({1}) = 0;
        AH({2}) = 5;

        bool test12 = false;

        try
        {
            Tensor<double> AI = AG / AH;
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

        Tensor<double> AJ({2,3});

        AJ({0,0}) = 1;
        AJ({0,1}) = 2;
        AJ({0,2}) = 3;
        AJ({1,0}) = 4;
        AJ({1,1}) = 5;
        AJ({1,2}) = 6;

        Tensor<double> AK = AJ.sum(0);

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

        Tensor<double> AL = AJ.sum(1);

        bool test14 =
            AL({0}) == 6 &&
            AL({1}) == 15;

        printTest("Sum [2,3] along axis 1", test14);


        // ============================================================
        // TEST 15: Higher-dimensional reduction
        // [2,3,4]
        // ============================================================

        Tensor<double> AM({2,3,4});

        size_t value = 0;

        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 3; ++j)
                for (size_t k = 0; k < 4; ++k)
                    AM({i,j,k}) = static_cast<double>(value++);

        Tensor<double> AN = AM.sum(1);

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
            Tensor<double> AO({2,3,4});
            Tensor<double> AP = AO.sum(3);
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
            Tensor<double> AQ({2,0,4});
            Tensor<double> AR = AQ.sum(1);
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

        Tensor<double> AS({2,3});

        AS({0,0}) = 1;
        AS({0,1}) = 2;
        AS({0,2}) = 3;
        AS({1,0}) = 4;
        AS({1,1}) = 5;
        AS({1,2}) = 6;

        Tensor<double> AT = AS.mean(0);

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

        Tensor<double> AU = AS.mean(1);

        bool test19 =
            AU({0}) == 2.0 &&
            AU({1}) == 5.0;

        printTest("Mean [2,3] along axis 1", test19);


        // ============================================================
        // TEST 20: Mean along axis 0 with keepdim
        // -> shape [1, 3]
        // ============================================================

        Tensor<double> AV = AS.mean(0, true);

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
            Tensor<double> AW({2,3,4});
            Tensor<double> AX = AW.mean(3);
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
            Tensor<double> AY({2,0,4});
            Tensor<double> AZ = AY.mean(1);
        }
        catch (const std::invalid_argument&)
        {
            test22 = true;
        }

        printTest("Reject mean of zero-sized axis", test22);


        // ============================================================
        // TEST 23: 2D Matrix Multiplication
        // [2, 3] x [3, 2] -> [2, 2]
        // ============================================================

        Tensor<double> BA({2, 3});
        Tensor<double> BB({3, 2});

        BA({0,0}) = 1; BA({0,1}) = 2; BA({0,2}) = 3;
        BA({1,0}) = 4; BA({1,1}) = 5; BA({1,2}) = 6;

        BB({0,0}) = 7; BB({0,1}) = 8;
        BB({1,0}) = 9; BB({1,1}) = 10;
        BB({2,0}) = 11; BB({2,1}) = 12;

        Tensor<double> BC = BA.matmul(BB);

        bool test23 =
            BC({0,0}) == 58 &&
            BC({0,1}) == 64 &&
            BC({1,0}) == 139 &&
            BC({1,1}) == 154;

        printTest("2D Matrix Multiplication [2,3] x [3,2]", test23);


        // ============================================================
        // TEST 24: Broadcasted Batch Matrix Multiplication
        // [2, 2, 3] x [3, 2] -> [2, 2, 2]
        // ============================================================

        Tensor<double> BD({2, 2, 3});
        Tensor<double> BE({3, 2});

        BE({0,0}) = 7; BE({0,1}) = 8;
        BE({1,0}) = 9; BE({1,1}) = 10;
        BE({2,0}) = 11; BE({2,1}) = 12;

        BD({0,0,0}) = 1; BD({0,0,1}) = 2; BD({0,0,2}) = 3;
        BD({0,1,0}) = 4; BD({0,1,1}) = 5; BD({0,1,2}) = 6;

        BD({1,0,0}) = 2; BD({1,0,1}) = 4; BD({1,0,2}) = 6;
        BD({1,1,0}) = 8; BD({1,1,1}) = 10; BD({1,1,2}) = 12;

        Tensor<double> BF = BD.matmul(BE);

        bool test24 =
            BF({0,0,0}) == 58 &&
            BF({0,0,1}) == 64 &&
            BF({0,1,0}) == 139 &&
            BF({0,1,1}) == 154 &&
            BF({1,0,0}) == 116 &&
            BF({1,0,1}) == 128 &&
            BF({1,1,0}) == 278 &&
            BF({1,1,1}) == 308;

        printTest("Broadcasted Batch Matmul [2,2,3] x [3,2]", test24);


        // ============================================================
        // TEST 25: Invalid Matrix Multiplication (dimension mismatch)
        // [2, 3] x [4, 2]
        // ============================================================

        bool test25 = false;
        try
        {
            Tensor<double> BG({2, 3});
            Tensor<double> BH({4, 2});
            Tensor<double> BI = BG.matmul(BH);
        }
        catch (const std::invalid_argument&)
        {
            test25 = true;
        }

        printTest("Reject invalid matmul (inner dimension mismatch)", test25);


        // ============================================================
        // TEST 26: Matmul with 1D tensors (Vector Promotion)
        // [3] x [3]
        // ============================================================

        bool test26 = true;
        try
        {
            Tensor<double> BJ({3});
            Tensor<double> BK({3});
            Tensor<double> BL = BJ.matmul(BK);
        }
        catch (const std::invalid_argument&)
        {
            test26 = false; // It should no longer throw!
        }

        printTest("Accept matmul with 1D vectors", test26);


        // ============================================================
        // TEST 27: 3D x 3D Batch Matmul (Same Batch Size)
        // [2, 2, 3] x [2, 3, 2] -> [2, 2, 2]
        // ============================================================
        
        Tensor<double> BM({2, 2, 3});
        Tensor<double> BN({2, 3, 2});

        // Batch 0
        BM({0,0,0}) = 1; BM({0,0,1}) = 2; BM({0,0,2}) = 3;
        BM({0,1,0}) = 4; BM({0,1,1}) = 5; BM({0,1,2}) = 6;

        BN({0,0,0}) = 7; BN({0,0,1}) = 8;
        BN({0,1,0}) = 9; BN({0,1,1}) = 10;
        BN({0,2,0}) = 11; BN({0,2,1}) = 12;

        // Batch 1
        BM({1,0,0}) = 1; BM({1,0,1}) = 0; BM({1,0,2}) = 0;
        BM({1,1,0}) = 0; BM({1,1,1}) = 1; BM({1,1,2}) = 0;

        BN({1,0,0}) = 1; BN({1,0,1}) = 2;
        BN({1,1,0}) = 3; BN({1,1,1}) = 4;
        BN({1,2,0}) = 5; BN({1,2,1}) = 6;

        Tensor<double> BO = BM.matmul(BN);

        bool test27 =
            // Batch 0
            BO({0,0,0}) == 58 &&
            BO({0,0,1}) == 64 &&
            BO({0,1,0}) == 139 &&
            BO({0,1,1}) == 154 &&
            // Batch 1
            BO({1,0,0}) == 1 &&
            BO({1,0,1}) == 2 &&
            BO({1,1,0}) == 3 &&
            BO({1,1,1}) == 4;

        printTest("3D x 3D Batch Matmul [2,2,3] x [2,3,2]", test27);


        // ============================================================
        // TEST 28: 2D x 3D Broadcasted Batch Matmul
        // [2, 3] x [2, 3, 2] -> [2, 2, 2]
        // ============================================================
        
        Tensor<double> BP({2, 3});
        Tensor<double> BQ({2, 3, 2});

        BP({0,0}) = 1; BP({0,1}) = 0; BP({0,2}) = 0;
        BP({1,0}) = 0; BP({1,1}) = 1; BP({1,2}) = 0;

        // Batch 0
        BQ({0,0,0}) = 7; BQ({0,0,1}) = 8;
        BQ({0,1,0}) = 9; BQ({0,1,1}) = 10;
        BQ({0,2,0}) = 11; BQ({0,2,1}) = 12;

        // Batch 1
        BQ({1,0,0}) = 1; BQ({1,0,1}) = 2;
        BQ({1,1,0}) = 3; BQ({1,1,1}) = 4;
        BQ({1,2,0}) = 5; BQ({1,2,1}) = 6;

        Tensor<double> BR = BP.matmul(BQ);

        bool test28 =
            // Batch 0
            BR({0,0,0}) == 7 &&
            BR({0,0,1}) == 8 &&
            BR({0,1,0}) == 9 &&
            BR({0,1,1}) == 10 &&
            // Batch 1
            BR({1,0,0}) == 1 &&
            BR({1,0,1}) == 2 &&
            BR({1,1,0}) == 3 &&
            BR({1,1,1}) == 4;

        printTest("2D x 3D Broadcast Matmul [2,3] x [2,3,2]", test28);


        // ============================================================
        // TEST 29: Incompatible batch dimensions
        // [2, 2, 3] x [3, 3, 2]
        // ============================================================
        
        bool test29 = false;
        try
        {
            Tensor<double> BS({2, 2, 3});
            Tensor<double> BT({3, 3, 2});
            Tensor<double> BU = BS.matmul(BT);
        }
        catch (const std::invalid_argument&)
        {
            test29 = true;
        }

        printTest("Reject matmul with incompatible batch dims", test29);


        // ============================================================
        // TEST 30: Complex Batch Broadcasting
        // [2, 1, 2, 3] x [4, 3, 2] -> [2, 4, 2, 2]
        // ============================================================
        
        Tensor<double> BV({2, 1, 2, 3});
        Tensor<double> BW({4, 3, 2});

        for (size_t b1 = 0; b1 < 2; ++b1)
            for (size_t b2 = 0; b2 < 1; ++b2)
                for (size_t i = 0; i < 2; ++i)
                    for (size_t j = 0; j < 3; ++j)
                        BV({b1, b2, i, j}) = 1.0;

        for (size_t b1 = 0; b1 < 4; ++b1)
            for (size_t i = 0; i < 3; ++i)
                for (size_t j = 0; j < 2; ++j)
                    BW({b1, i, j}) = 2.0;

        Tensor<double> BX = BV.matmul(BW);

        bool test30 = true;
        try {
            for (size_t b1 = 0; b1 < 2; ++b1) {
                for (size_t b2 = 0; b2 < 4; ++b2) {
                    for (size_t i = 0; i < 2; ++i) {
                        for (size_t j = 0; j < 2; ++j) {
                            if (BX({b1, b2, i, j}) != 6.0) {
                                test30 = false;
                            }
                        }
                    }
                }
            }
            
            // Check out of bounds to confirm shape
            bool outOfBoundsThrown = false;
            try {
                BX({2, 0, 0, 0});
            } catch (const std::out_of_range&) {
                outOfBoundsThrown = true;
            }
            if (!outOfBoundsThrown) test30 = false;

            outOfBoundsThrown = false;
            try {
                BX({0, 4, 0, 0});
            } catch (const std::out_of_range&) {
                outOfBoundsThrown = true;
            }
            if (!outOfBoundsThrown) test30 = false;
            
        } catch (...) {
            test30 = false;
        }

        printTest("Complex Broadcast Matmul [2,1,2,3] x [4,3,2]", test30);

        // ============================================================
        // TEST 31: Tensor Fill
        // ============================================================
        
        Tensor<double> BY({2, 3});
        BY.fill(7.5);

        bool test31 = true;
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                if (BY({i, j}) != 7.5) {
                    test31 = false;
                }
            }
        }
        printTest("Tensor Fill", test31);

        // ============================================================
        // TEST 32: 1D Matmul cases
        // ============================================================
        bool test32 = true;
        try {
            Tensor<double> V1({3}); V1({0}) = 1; V1({1}) = 2; V1({2}) = 3;
            Tensor<double> V2({3}); V2({0}) = 4; V2({1}) = 5; V2({2}) = 6;
            
            // 1D x 1D (dot product: 1*4 + 2*5 + 3*6 = 32)
            Tensor<double> r1 = V1.matmul(V2);
            if (r1.getShape().size() != 0 || r1({}) != 32.0) test32 = false;

            // 2D x 1D ([2, 3] x [3])
            Tensor<double> M2({2, 3});
            M2({0,0}) = 1; M2({0,1}) = 2; M2({0,2}) = 3;
            M2({1,0}) = 4; M2({1,1}) = 5; M2({1,2}) = 6;
            Tensor<double> r2 = M2.matmul(V1);
            if (r2.getShape().size() != 1 || r2.getShape()[0] != 2) test32 = false;
            if (r2({0}) != 14.0 || r2({1}) != 32.0) test32 = false;
            
        } catch (...) {
            test32 = false;
        }
        printTest("1D Matmul (Vector Promotion)", test32);

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