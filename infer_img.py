import cv2
import warnings
import torch
from torch import Tensor
from build_models import build_model
from tools import Tools

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

model, infer = build_model('RAFT+PerVFI')

i1 = cv2.imread('archive/133.png', cv2.IMREAD_UNCHANGED)
i3 = cv2.imread('archive/253.png', cv2.IMREAD_UNCHANGED)
t1 = Tools.toTensor(i1).to('cuda')
t3 = Tools.toTensor(i3).to('cuda')
print(t1.size(), t3.size())
t2 = infer(t1, t3)
print(t2.size())
i2 = Tools.toArray(t2.to('cpu'))
cv2.imwrite('archive/res.png', i2)
