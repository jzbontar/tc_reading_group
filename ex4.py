import torch
import tensor_comprehensions as tc
import timeit

lang = '''
def mm(float(N, M) x, float(M, K) y) -> (z) {
    z(n, k) +=! x(n, m) * y(m, k)
}
'''

n, m, k = 55, 66, 77
mm = tc.define(lang, name='mm')

x = torch.Tensor(n, m).cuda().uniform_()
y = torch.Tensor(m, k).cuda().uniform_()
mm.autotune(x, y, options='mlp', generations=5, pop_size=10, number_elits=1, gpus='0,1')
# mm.autotune(x, y, options='mlp', generations=25, pop_size=100, number_elits=10, gpus='0,1')

import numpy as np
print(np.allclose(mm(x, y).data.cpu().numpy(), torch.mm(x, y).cpu().numpy()))

print(timeit.repeat('mm(x, y)', 'from __main__ import mm, x, y', repeat=3, number=10000))
print(timeit.repeat('torch.mm(x, y)', 'from __main__ import torch, x, y', repeat=3, number=10000))
