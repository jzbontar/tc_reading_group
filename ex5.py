import tensor_comprehensions as tc
import torch

lang = '''
def conv2d(float(B, C1, H, W) input, float(C2, C1, KH, KW) weight) -> (output) {
    output(b, c2, h, w) +=! input(b, c1, h + kh, w + kw) * weight(c2, c1, kh, kw)
}
'''

B, C1, C2, H, W, KH, KW = 4, 6, 8, 32, 32, 3, 3
conv2d_pth = tc.define(lang, name='conv2d')
conv2d_nn = torch.nn.Conv2d(C1, C2, (KH, KW), bias=False).cuda()
input = torch.autograd.Variable(torch.Tensor(B, C1, H, W).cuda())
weight = conv2d_nn.weight
output_nn = conv2d_nn(input)
output_pth = conv2d_pth(input.data, weight.data)
print((output_nn - output_pth).abs().max())
