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