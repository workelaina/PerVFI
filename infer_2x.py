import cv2
import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument(
    "--method",
    "-m",
    type=str,
    default="raft+pervfi",
    choices=[
        "raft+pervfi",
        "raft+pervfi-vb",
        "gma+pervfi",
        "gma+pervfi-vb",
        "gmflow+pervfi",
        "gmflow+pervfi-vb",
    ],
    help="different model types",
)
args = parser.parse_args()

method_name: str = args.method
model, infer = build_model(method_name)

video = 'archive/4k.mp4'
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

pth = 'result/' + method_name.split('+')[0] + '/%d.png'

for i in range(nb_frames):
    print('batch', i)
    succ, img = capture.read()
    if not succ:
        raise RuntimeError(i)
    t3 = Tools.toTensor(img).cuda()
    if i:
        with torch.no_grad():
            t2 = infer(t1, t3)
        i2 = Tools.toArray(t2.cpu())
        cv2.imwrite(pth % (i*2-1), i2)
    cv2.imwrite(pth % (i*2), img)
    t1 = t3
