import numpy as np
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad

# def numerical_gradient(f,x):        #計算梯度(多變量微分)的function
#     h = 1e-4
#     grad = np.zeros_like(x)

#     for i in range(x.size):         #有x.size個參數要算
#         temp = x[i]
#         x[i] = temp + h
#         fxh1 = f(x) 

#         x[i] = temp - h
#         fxh2 = f(x)
        
#         grad[i] = (fxh1 - fxh2) / (2*h)
#         x[i] = temp
#     return grad

def gradient_descent(f, init_x, lr = 0.01, step_num = 100):# learning rate預設0.01 step num預設100
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f,x)  
        x -= lr * grad          
    return x

def func(x):
    return x[0]**2 + x[1]**2


# init_x = np.array([-3.0, 4.0])
# gradient_descent(func,init_x, lr =0.0489768,step_num=100)   #GD方法 回傳找到使函數最小/極小的參數(最小應為x0 = 0, x1 = 0)