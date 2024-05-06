### files:

- miniBatchLearning.py：這章的主角，結合之前所學的，對minist資料集做一次model training(只用forward propagation)

- Gradient_descent.py：能算各種形狀參數的gradient的 function

- functions.py : 作者寫好的activation functions(自己寫也很容易)

- mnist.py : 作者寫好的可以讀Minist 資料集資料的functions
	裡面的load_mnist(normalize=True, flatten=True, one_hot_label=False)
	會回傳(train Data, train Label, test Data, test Label )

- CrossEntropyError.py: 我寫的交叉商誤差function

- plot3Dfunction.py : 我寫的，描繪出了f(x0,x1) = x0^2 + x1^2的圖形

- Numerical_diffferentiation:展示了f(x)= 0.01x^2 + 0.1x 的圖形及 對點x=5,10微分的結果，是介紹如何用python寫出微分的部分

-TwoLayerNet.py：我寫的神經網路 class，有init, predict(inference), loss計算等functions

