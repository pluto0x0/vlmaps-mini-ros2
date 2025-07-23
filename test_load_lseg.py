import os
from pathlib import Path
import cv2
import torchvision.transforms as transforms
import torch
import gdown

from lseg.utils.lseg_utils import get_lseg_feat
from lseg.models.lseg_encnet import LSegEncNet


device = "cuda" if torch.cuda.is_available() else "cpu"

def _init_lseg():
    crop_size = 192  # 480
    base_size = 400  # 520
    lseg_model = LSegEncNet("", arch_option=0, block_depth=0, activation="lrelu", crop_size=crop_size)
    model_state_dict = lseg_model.state_dict()
    checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
    checkpoint_path = checkpoint_dir / "demo_e200.ckpt"
    os.makedirs(checkpoint_dir, exist_ok=True)
    if not checkpoint_path.exists():
        print("Downloading LSeg checkpoint...")
        # the checkpoint is from official LSeg github repo
        # https://github.com/isl-org/lang-seg
        checkpoint_url = "https://drive.google.com/u/0/uc?id=1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb"
        gdown.download(checkpoint_url, output=str(checkpoint_path))

    pretrained_state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    pretrained_state_dict = {k.lstrip("net."): v for k, v in pretrained_state_dict["state_dict"].items()}
    model_state_dict.update(pretrained_state_dict)
    lseg_model.load_state_dict(pretrained_state_dict)

    lseg_model.eval()
    lseg_model = lseg_model.to(device)

    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    lseg_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return lseg_model, lseg_transform, crop_size, base_size, norm_mean, norm_std


lseg_model, lseg_transform, crop_size, base_size, norm_mean, norm_std = _init_lseg()
img = '/home/yzf/图片/It.jpg'
bgr = cv2.imread(str(img))
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
pix_feats = get_lseg_feat(
    lseg_model, rgb, ["example"], lseg_transform, device, crop_size, base_size, norm_mean, norm_std
)
print(pix_feats.shape)
