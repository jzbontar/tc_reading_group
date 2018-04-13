import torch
import tensor_comprehensions as tc

lang = '''
def scale2(float(N) x) -> (y) {
    y(n) = x(n) * 2
}
'''

scale2 = tc.define(lang, name='scale2')

x = torch.Tensor(4).cuda().uniform_()
y = scale2(x)
print(x)
print(y)
