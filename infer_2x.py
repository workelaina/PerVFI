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

video = 'archive/Elysia1.mp4'
capture = cv2.VideoCapture(video)
nb_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
nb_frames = int(nb_frames + 0.5)
ori_fps = capture.get(cv2.CAP_PROP_FPS)
if ori_fps < 0.1:
    raise RuntimeError("The frame rate is %s!" % str(ori_fps))
# duration = nb_frames / ori_fps
print('nb_frames:', nb_frames)
print('ori_fps:', ori_fps)
print()

for i in range(nb_frames):
    print('batch', i)
    succ, img = capture.read()
    if not succ:
        raise RuntimeError(i)
    t3 = Tools.toTensor(img).to('cuda')
    if i:
        t2 = infer(t1, t3)
        i2 = Tools.toArray(t2.to('cpu'))
        cv2.imwrite('result/%d.png' % (i*2-1), i2)
    cv2.imwrite('result/%d.png' % (i*2), img)
    t1 = t3
