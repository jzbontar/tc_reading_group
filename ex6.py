import torch
import tensor_comprehensions
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image

lang = '''
def ad_vol(float(C, N, M) left, float(C, N, M) right) -> (vol) {
    vol(d, i, j) +=! fabs(left(c, i, j) - right(c, i, j - d)) / C where d in 0:16
}

def census_transform(float(C, N, M) x) -> (out) {
    out(ii, jj, c, i, j) = float(x(c, i + ii, j + jj) < x(c, i + 3, j + 4)) / C where ii in 0:7, jj in 0:9
}

def census_vol(float(NN, MM, C, N, M) left, float(NN, MM, C, N, M) right) -> (vol) {
    vol(d, i, j) +=! float(left(nn, mm, c, i, j) != right(nn, mm, c, i, j - d)) where d in 0:16
}
'''

tc_ad_vol = tensor_comprehensions.define(lang, name='ad_vol')
tc_census_transform = tensor_comprehensions.define(lang, name='census_transform')
tc_census_vol = tensor_comprehensions.define(lang, name='census_vol')

def load_image(fname):
    return to_tensor(Image.open(fname)).cuda()

def save_disp(vol, fname, max_disp):
    _, d = torch.min(vol, dim=0, keepdim=True)
    to_pil_image((d.float() / max_disp).data.cpu()).save(fname)
    
x0 = load_image('data/imL.png')
x1 = load_image('data/imR.png')

# ad_vol = tc_ad_vol(x0, x1)
# save_disp(ad_vol, 'ad.png', 16)

x0c = tc_census_transform(x0)
x1c = tc_census_transform(x1)
census_vol = tc_census_vol(x0c, x1c)
save_disp(census_vol, 'census.png', 16)
