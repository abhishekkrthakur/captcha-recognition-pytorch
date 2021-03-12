import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.nn import functional as F
import os
from sklearn import preprocessing
import time
import albumentations

class CaptchaModel(nn.Module):

    def __init__(self, num_chars):
        super(CaptchaModel, self).__init__()

        self.conv_1 = nn.Conv2d(3, 256, kernel_size=(3, 6), padding=(1, 1))
        self.pool_1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv_2 = nn.Conv2d(256, 64, kernel_size=(3, 6), padding=(1, 1))
        self.pool_2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.linear_1 = nn.Linear(1152, 64)
        self.drop_1 = nn.Dropout(0.2)
        self.lstm = nn.GRU(64, 32, bidirectional=True, num_layers=2, dropout=0.25, batch_first=True)
        self.output = nn.Linear(64, num_chars + 1)

    def forward(self, images, targets=None):
        bs, _, _, _ = images.size()
        x = F.relu(self.conv_1(images))
        x = self.pool_1(x)
        x = F.relu(self.conv_2(x))
        x = self.pool_2(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(bs, x.size(1), -1)
        x = F.relu(self.linear_1(x))
        x = self.drop_1(x)
        x, _ = self.lstm(x)
        x = self.output(x)
        x = x.permute(1, 0, 2)

        if targets is not None:
            log_probs = F.log_softmax(x, 2)
            input_lengths = torch.full(
                size=(bs,), fill_value=log_probs.size(0), dtype=torch.int32
            )
            target_lengths = torch.full(
                size=(bs,), fill_value=targets.size(1), dtype=torch.int32
            )
            loss = nn.CTCLoss(blank=0)(
                log_probs, targets, input_lengths, target_lengths
            )
            return x, loss

        return x, None

model = CaptchaModel(33)
device = torch.device("cuda")
model.to(device)

PATH = "./your_model_state_dict.pth" 
model.load_state_dict(torch.load(PATH))

def remove_duplicates(s):
    chars = list(s)
    prev = None
    k = 0
 
    for c in s:
        if prev != c:
            chars[k] = c
            prev = c
            k = k + 1
    
    return ''.join(chars[:k])

def return_model_answer(img_path):
    """
    input: image_path
    output: captcha as a string
    """
  
    model.eval()
    start = time.time()

    img = img_path
    image = Image.open(img_path).convert("RGB")
    image = image.resize(
                (70, 25), resample=Image.BILINEAR
            )
    image = np.array(image)

    # Some ImageNet data tricks
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    aug = albumentations.Compose(
        [
            albumentations.Normalize(
                mean, std, max_pixel_value=255.0, always_apply=True
            )
        ]
    )
    image = aug(image=image)["image"]
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = image[None,:,:,:] #add batch dimension as 1
    img = torch.from_numpy(image)
    if str(device) == "cuda":
        img= img.cuda()
    img = img.float()
    
    preds, _ = model(img) # tensor with each class output from net
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    captcha = preds[0]
    #your custom characters that appear in the captcha
    decode = {
    0:"",
    1:"1",
    2:"2",
    3:"3",
    4:"4",
    5:"5",
    6:"6",
    7:"7",
    8:"8",
    9:"a",
    10:"b",
    11:"c",
    12:"d",
    13:"e",
    14:"f",
    15:"g",
    16:"h",
    17:"i",
    18:"j",
    19:"k",
    20:"l",
    21:"m",
    22:"n",
    23:"o",
    24:"p",
    25:"q",
    26:"r",
    27:"s",
    28:"t",
    29:"u",
    30:"v",
    31:"w",
    32:"x",
    33:"y",
    }
    answer= ""
    for i in captcha:
        answer += decode[i]

    end = time.time()
    answer = remove_duplicates(answer)
    print(answer)
    print(f"model time: {(end - start)*1000:.2f}ms")
    print("\n")
    return answer

rootdir = 'input/'
right = 0
cnt = 0
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        cnt+=1
        answer = file[:5] # captchas here are always length 5
        input_data = return_model_answer(f"input/{file}")
        if input_data == answer:
            right+=1
        

print(f"right captchas {right} in {cnt}")
print(f"accuracy:{(right/cnt)*100:.2f}%"
