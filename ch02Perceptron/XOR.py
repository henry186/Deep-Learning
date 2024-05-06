def NAND(x1,x2):    
    w1, w2, theta = -0.5, -0.5, -0.7
    tmp = w1 * x1 + w2 * x2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1
    
def AND(x1,x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = w1 * x1 + w2 * x2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1
    
def OR(x1,x2):
    w1, w2, theta = 0.5, 0.5, 0.4
    tmp = w1 * x1 + w2 * x2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1
    
def XOR(x1,x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    return AND(s1,s2)

print(XOR(0,0))
print(XOR(0,1))
print(XOR(1,0))
print(XOR(1,1))