import torch
import tensor_comprehensions as tc

lang = '''
def scale(float(N) x) -> (y) {{
    y(n) = x(n) * {s}
}}

def scale_backward(float(N) x, float(N) grad_output) -> (grad_input) {{
    grad_input(n) = grad_output(n) * {s}
}}
'''

scale2 = tc.define(lang, training=True, name='scale', backward='scale_backward', constants={'s': 2})
scale4 = tc.define(lang, training=True, name='scale', backward='scale_backward', constants={'s': 4})

x = torch.autograd.Variable(torch.Tensor(4).cuda().uniform_(), requires_grad=True)
y = scale4(x)
print(y)
y.mean().backward()
print(x.grad)
