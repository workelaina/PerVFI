import argparse
import os
import os.path as osp
import warnings

import torch
from tqdm import tqdm

from tools import IOBuffer, Tools

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

############ Arguments ############

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
parser.add_argument(
    "--xx", "-x", type=int, default=2, help="X 2/4/8 interpolation"
)
args = parser.parse_args()

############ Preliminary ############
SCALE = args.xx
input_frames = 2
dstDir = "test"
tmpDir = "tmp"
RESCALE = None
videos = sorted(os.listdir(dstDir))
print("Running VFI method : ", args.method)
print("Testing on Dataset: ", dstDir)
print("TMP (temporary) Dir: ", tmpDir)

from build_models import build_model
print("Building VFI model...")
model, infer = build_model(args.method, device=DEVICE)
print("Done")


def inferRGB(*inputs):
    inputs = [x.to(DEVICE) for x in inputs]
    for x in inputs:
        print(x.size())
    outputs = []
    for time in range(SCALE - 1):
        t = (time + 1) / SCALE
        tenOut = infer(*inputs, time=t)
        outputs.append(tenOut.cpu())
    return outputs

print(videos)
for vid_name in tqdm(videos):
    sequences = [
        x for x in os.listdir(osp.join(dstDir, vid_name)) if ".jpg" in x or ".png" in x
    ]
    sequences.sort(key=lambda x: int(x[:-4]))  # NOTE: This might cause some BUG!
    sequences = [osp.join(dstDir, vid_name, x) for x in sequences]
    print(sequences)

    ############ build buffer with multi-threads ############
    inputSeq = sequences
    IO = IOBuffer(RESCALE, inp_num=input_frames)

    os.makedirs(osp.join(tmpDir, vid_name), exist_ok=True)
    IO.start(inputSeq, osp.join(tmpDir, vid_name))

    ############ interpolation & write distorted frames ############
    inps = IO.reader.get()  # e.g., [I1 I3]
    IO.writer.put(Tools.toArray(inps[0]))
    while True:
        outs = inferRGB(*inps)  # e.g., [I2]
        for o in Tools.toArray(outs + [inps[-input_frames // 2]]):
            IO.writer.put(o)
        inps = IO.reader.get()
        if inps is None:
            break
    IO.stop()
