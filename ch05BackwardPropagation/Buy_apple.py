from MulLayer import MulLayer
from AddLayer import AddLayer

# declare parameters
apple = 100
orange = 150
orange_num = 3
apple_num = 2
tax = 1.1

# create layers
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()
mul_orange_layer = MulLayer()
Untaxed_price_layer = AddLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
total_price = Untaxed_price_layer.forward(apple_price, orange_price)
taxed_price = mul_tax_layer.forward(total_price, tax)

print(taxed_price)

# backward
dtaxed_price = 1
dtotal_price, dtax = mul_tax_layer.backward(dtaxed_price)
dapple_price, dorange_price = Untaxed_price_layer.backward(dtotal_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)


print(dapple, round(dapple_num, 2), round(dorange, 2), dorange_num, dtax)
