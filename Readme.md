# SGD with 0.01 lr and 100 epochs
```
👻 shivam@MacBook-Pro ~/build 🌿(main) ❯ ./layer_test 
Starting training on 10000 samples...
Epoch 0 | Average Loss: 0.0787247
Epoch 10 | Average Loss: 0.00315721
Epoch 20 | Average Loss: 0.00143211
Epoch 30 | Average Loss: 0.000892299
Epoch 40 | Average Loss: 0.000637102
Epoch 50 | Average Loss: 0.000490055
Epoch 60 | Average Loss: 0.000395478
Epoch 70 | Average Loss: 0.000330064
Epoch 80 | Average Loss: 0.000282301
Epoch 90 | Average Loss: 0.000245944
Epoch 99 | Average Loss: 0.000220026
Training complete in 110.274 seconds!

================ TESTING PHASE ================
Prediction     Target         Error
-----------------------------------------------
0.00160059     0              0.00160059
0.999875       1              -0.000124784
0.000330936    0              0.000330936
0.997506       1              -0.00249399
0.000134853    0              0.000134853
0.988965       1              -0.0110347
0.000686207    0              0.000686207
0.996628       1              -0.00337156
0.00380966     0              0.00380966
0.999997       1              -2.84397e-06

Final Test MSE: 0.000218008
===============================================
```
# Momentum with 0.01 lr, 0.2 beta and 100 epochs
```
👻 shivam@MacBook-Pro ~/build 🌿(main) ❯ ./layer_test 
Starting training on 10000 samples...
Epoch 0 | Average Loss: 0.0724687
Epoch 10 | Average Loss: 0.00250136
Epoch 20 | Average Loss: 0.00112746
Epoch 30 | Average Loss: 0.000702001
Epoch 40 | Average Loss: 0.000500838
Epoch 50 | Average Loss: 0.000385074
Epoch 60 | Average Loss: 0.000310819
Epoch 70 | Average Loss: 0.000259461
Epoch 80 | Average Loss: 0.000221968
Epoch 90 | Average Loss: 0.000193414
Epoch 99 | Average Loss: 0.000173057
Training complete in 116.08 seconds!

================ TESTING PHASE ================
Prediction     Target         Error
-----------------------------------------------
0.00113704     0              0.00113704
0.999922       1              -7.80849e-05
0.00021651     0              0.00021651
0.998155       1              -0.00184516
8.47518e-05    0              8.47518e-05
0.991233       1              -0.00876667
0.000462619    0              0.000462619
0.997299       1              -0.00270051
0.00277375     0              0.00277375
0.999999       1              -1.43899e-06

Final Test MSE: 0.000171441
===============================================
```

# Momentum with 0.01 lr, 0.8 beta and 100 epochs
👻 shivam@MacBook-Pro ~/build 🌿(main) ❯ ./layer_test 
Starting training on 10000 samples...
Epoch 0 | Average Loss: 0.0295013
Epoch 10 | Average Loss: 0.000471152
Epoch 20 | Average Loss: 0.000213514
Epoch 30 | Average Loss: 0.000134071
Epoch 40 | Average Loss: 9.63626e-05
Epoch 50 | Average Loss: 7.46404e-05
Epoch 60 | Average Loss: 6.06162e-05
Epoch 70 | Average Loss: 5.0855e-05
Epoch 80 | Average Loss: 4.36929e-05
Epoch 90 | Average Loss: 3.82275e-05
Epoch 99 | Average Loss: 3.4315e-05
Training complete in 107.592 seconds!

================ TESTING PHASE ================
Prediction     Target         Error
-----------------------------------------------
0.000124768    0              0.000124768
0.999995       1              -5.45203e-06
1.35778e-05    0              1.35778e-05
0.999699       1              -0.000300607
5.13817e-06    0              5.13817e-06
0.997346       1              -0.00265373
3.78765e-05    0              3.78765e-05
0.999579       1              -0.000420564
0.000421267    0              0.000421267
1              1              -1.49544e-08

Final Test MSE: 3.38386e-05
===============================================
👻 shivam@MacBook-Pro ~/build 🌿(main) ❯ 


# ADAM with lr : 0.01, beta1 : 0.3, beta2 : 0.3, epochs : 100
```
👻 shivam@MacBook-Pro ~/build 🌿(main) ❯ ./layer_test
Starting training on 10000 samples...
Epoch 0 | Average Loss: 0.0106844
Epoch 10 | Average Loss: 1.68038e-08
Epoch 20 | Average Loss: 1.71585e-10
Epoch 30 | Average Loss: 9.08615e-12
Epoch 40 | Average Loss: 3.12102e-12
Epoch 50 | Average Loss: 1.68744e-12
Epoch 60 | Average Loss: 1.1249e-12
Epoch 70 | Average Loss: 8.3923e-13
Epoch 80 | Average Loss: 6.67701e-13
Epoch 90 | Average Loss: 5.53153e-13
Epoch 99 | Average Loss: 4.78534e-13
Training complete in 109.61 seconds!

================ TESTING PHASE ================
Prediction     Target         Error
-----------------------------------------------
6.02392e-16    0              6.02392e-16
1              1              0
1.38572e-19    0              1.38572e-19
1              1              -9.01501e-14
1.20424e-20    0              1.20424e-20
1              1              -2.99691e-11
4.45836e-18    0              4.45836e-18
1              1              -4.26326e-14
2.92165e-14    0              2.92165e-14
1              1              0

Final Test MSE: 4.57705e-13
===============================================
👻 shivam@MacBook-Pro ~/build 🌿(main) ❯ 
```

# ADAM with lr : 0.01, beta1 : 0.3, beta2 : 0.3, epochs : 100, batch_size = 32
```
👻 shivam@MacBook-Pro ~/build 🌿(main) ❯ ./layer_test 
Starting Mini-Batch training on 10000 samples with Batch Size 32...
Epoch 0 | Average Loss: 0.0284136
Epoch 10 | Average Loss: 6.2698e-07
Epoch 20 | Average Loss: 8.15822e-09
Epoch 30 | Average Loss: 6.28171e-10
Epoch 40 | Average Loss: 1.70553e-10
Epoch 50 | Average Loss: 8.33797e-11
Epoch 60 | Average Loss: 5.17258e-11
Epoch 70 | Average Loss: 3.64591e-11
Epoch 80 | Average Loss: 2.7838e-11
Epoch 90 | Average Loss: 2.2397e-11
Epoch 99 | Average Loss: 1.89999e-11
Training complete in 243.666 seconds!

================ TESTING PHASE ================
Prediction     Target         Error
-----------------------------------------------
1.27264e-12    0              1.27264e-12
1              1              -8.88178e-16
3.17568e-15    0              3.17568e-15
1              1              -1.63115e-09
9.08409e-16    0              9.08409e-16
1              1              -2.19672e-08
3.73262e-14    0              3.73262e-14
1              1              -1.26033e-12
3.12645e-11    0              3.12645e-11
1              1              0

Final Test MSE: 1.79516e-11
===============================================
👻 shivam@MacBook-Pro ~/build 🌿(main) ❯ 
```

## With Raw Pointer and Heap optimization Optimization

```
👻 shivam@MacBook-Pro ~/build 🌿(main) ❯ ./layer_test 
Starting Mini-Batch training on 10000 samples with Batch Size 32...
Epoch 0 | Average Loss: 0.0284136
Epoch 10 | Average Loss: 6.2698e-07
Epoch 20 | Average Loss: 8.15822e-09
Epoch 30 | Average Loss: 6.28171e-10
Epoch 40 | Average Loss: 1.70553e-10
Epoch 50 | Average Loss: 8.33797e-11
Epoch 60 | Average Loss: 5.17258e-11
Epoch 70 | Average Loss: 3.64591e-11
Epoch 80 | Average Loss: 2.7838e-11
Epoch 90 | Average Loss: 2.2397e-11
Epoch 99 | Average Loss: 1.89999e-11
Training complete in 0.501936 seconds!

================ TESTING PHASE ================
Prediction     Target         Error
-----------------------------------------------
1.27264e-12    0              1.27264e-12
1              1              -8.88178e-16
3.17568e-15    0              3.17568e-15
1              1              -1.63115e-09
9.08409e-16    0              9.08409e-16
1              1              -2.19672e-08
3.73262e-14    0              3.73262e-14
1              1              -1.26033e-12
3.12645e-11    0              3.12645e-11
1              1              0

Final Test MSE: 1.79516e-11
===============================================
```