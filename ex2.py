import torch
import tensor_comprehensions as tc

lang = '''
def scale2(float(N) x) -> (y) {
    y(n) = x(n) * 2
}

def scale2_backward(float(N) x, float(N) grad_output) -> (grad_input) {
    grad_input(n) = grad_output(n) * 2
}
'''

scale2 = tc.define(lang, training=True, name='scale2', backward='scale2_backward')

x = torch.autograd.Variable(torch.Tensor(8).cuda().uniform_(), requires_grad=True)
y = scale2(x)
print(y)
y.mean().backward()
print(x.grad)
