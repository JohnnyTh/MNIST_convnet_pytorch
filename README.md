# MNIST_convnet_pytorch

![Mnist](https://user-images.githubusercontent.com/39649806/65372651-8c090400-dc7b-11e9-9240-ee00cec2fac8.png)
An implementation of typical convnet architechture using pytorch for classificayion of MNIST dataset.

The dataset includes 60,000 train and 10,000 images of numbers in range 0 to 9 along with their labels [Link to more info](http://yann.lecun.com/exdb/mnist/).
The implemented network uses the architercure similar to classical LeNet-5 but with some improvements (dropout, learning rate decay, max pooling, etc.).
Additionally, the project implements data augumentation to exted the size of train dataset by transformed images generated from the train dataset.

The network is represented by the following layers:
1. CONV2d(5x5x1, s=1, p='same') -> ReLU
2. Max.Pool(2x2, s=2)
3. CONV2d(3x3x32, s=1, p='same') -> ReLU
4. Max.Pool(2x2, s=2)
5. Dropout() -> FC(256) -> ReLU -> Dropout()
6. FC(128) -> ReLU -> Dropout()
7. FC(10)

The trained model gives ~0.4 - 0.45% test set error after 35-40 epochs.
![model_final_3](https://user-images.githubusercontent.com/39649806/65372812-798fca00-dc7d-11e9-8f75-347ddff00e8c.png)

### Future work: ###
As can be seen from the examples of mislabeled pictures, trained model still has some difficulties correctly
figuring out labels for images where numbers are distorted/partly "erased"/have unusual shapes.
#### Mislabeled images:
![mislabeled_imgs_2_2](https://user-images.githubusercontent.com/39649806/65372956-67169000-dc7f-11e9-9f28-7cb0874a1254.png)

Suggestions on reducing the test set error:
- Stronger regularization. Test set error of the trained model is higher than the train set error suggesting the model overfits to the training set.
- More data augumentation. It could be useful to use data augumentation to generate more images similar to those misclassified by the model.
- Try different optimization algorithms.
- Play with learning rate decay.

### Requirements: ###
Python =>3.6.8
numpy == 1.16.5
torch == 1.1.0
torchvision == 0.3.0
matplotlib == 3.0.3
requests == 2.21.0
