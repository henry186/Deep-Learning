# 這章在介紹backward propagation
backward propagation是利用連鎖律拆解成每層的偏微分相乘得到最後所需的Gradient : $\frac{dL}{dx}$ 的方法，
其中L是Loss function，x是參數。

作者用「計算圖｣展示出這章每個layers做backward propagation的計算過程，
### directory
dataset: 放mnist資料集

### file
