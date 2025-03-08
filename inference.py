import torch
from PIL import Image

from CSD.model import CSD_CLIP
from CSD.utils import convert_state_dict
from CSD.loss_utils import transforms_branch0


model = CSD_CLIP()

model_path = "pretrainedmodels/pytorch_model.bin"

checkpoint = torch.load(model_path, map_location="cpu")
state_dict = convert_state_dict(checkpoint["model_state_dict"])
msg = model.load_state_dict(state_dict, strict=False)
print(f"=> loaded checkpoint with msg {msg}")

img_path = '/home/naumov/code/test.png'
img = Image.open(img_path).convert('RGB')
img = transforms_branch0(img)
img = img.unsqueeze(0).to('cuda')


out = model(img)
print(out)