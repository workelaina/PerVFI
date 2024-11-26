# pervfi

```sh
sudo mkdir -p /mnt/v1
sudo mount /dev/vdc /mnt/v1

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all

cd /mnt/v1
git clone https://github.com/workelaina/PerVFI.git
cd PerVFI
conda env create -f environment.yaml
conda activate pervfi

git pull
python infer_2x.py -m raft+pervfi
# python infer_video.py -m raft+pervfi --xx 2

scp -r ubuntu@pervfi.mil:/mnt/v1/PerVFI/result ./
```
