# 這章進入NeuralNetwork, 
- 介紹 Deep Learning 大致流程(由perceptron發展而來)
- 介紹五種activation function 及適用時機
- 最後用 MINIST dataset 及已經train好的model(Network的參數) implement inference 過程

## directories:
- ActivationFunction:裡面放我實作的5種 activation functions：
    * Identity:顧名思義是單位映射，通常用在**regression問題**
    * Sigmoid:$\frac{1}{1+e^x}$通常用在**二元分類**
    * Softmax:$\frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$，通常用在**多元分類**
    * ReLu:若輸入>0則直接輸出，若<=0則輸出0，
    * Step:>0輸出1，<=0輸出0
- _pycahe_: 由load_mnist() 產生的檔案，應該是pickle file用的

## files

### pythonScript
inference.py: 我寫的用已訓練好的model做inference範例，用minist資料集中的測試資料對此100張資料算出它的正確率(正確次數 / 總資料數(100張28x28的灰階影像))

functions.py 作者寫好的，裡面放sigmoid, softmax等函數

mnist.py 包含用來載入MINIST資料集的functions (作者寫好的)

mnist_show.py 顯示MINIST資料集的input圖片，看看裡面訓練資料圖片長怎樣

neuralnet_mnist.py 用來inference的資料，他利用已經train好的參數(sample_weight.pkl)來推論這個模型的準確度

### pickle files:
- mnist.pkl: 放minist資料集的pickle file, 可較快載入資料，load_mnist()會用到

- sample.weight.pkl 放train好參數的pkl檔