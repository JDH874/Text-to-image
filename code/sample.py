import torch
import os
from PIL import Image
import clip
import os.path as osp
import os, sys
import torchvision.utils as vutils
sys.path.insert(0, '../')

from lib.utils import load_model_weights,mkdir_p
from models.GALIP import NetG, CLIP_TXT_ENCODER
from lib.datasets1 import get_caption_test
#%%

device = 'cpu' # 'cpu' # 'cuda:0'
CLIP_text = "ViT-B/32"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model = clip_model.eval()

#%%

text_encoder = CLIP_TXT_ENCODER(clip_model).to(device)
netG = NetG(64, 100, 512, 256, 3, False, clip_model).to(device)
path = './saved_models/bird/test/state_epoch_580.pth'
checkpoint = torch.load(path, map_location=torch.device('cpu'))
netG = load_model_weights(netG, checkpoint['model']['netG'], multi_gpus=False)

#%%

batch_size = 16
noise = torch.randn((batch_size, 100)).to(device)

#%%

captions = [
'this a large dull brown bird with large wings, long neck, and a long gray bill.',
'this is a large brown and grey bird with a long neck and a large white beak.',    
]

#%%

mkdir_p('./samples')

#%%

# generate from text
with torch.no_grad():
    for i in range(len(captions)):
        caption = captions[i]
        print("caption: ", caption)
        tokenized_text = clip.tokenize([caption]).to(device)
       
        caps, tokens, token_a1, token_a2, token_a3, a1, a2, a3 = get_caption_test(caption, clip_model)
        
        tokenized_text1 = clip.tokenize(a1).to(device)
        a1, _ = text_encoder(tokenized_text1)
        a1 = a1.repeat(batch_size,1)
        tokenized_text2 = clip.tokenize(a2).to(device)
        a2, _ = text_encoder(tokenized_text2)
        a2 = a2.repeat(batch_size, 1)
        tokenized_text3 = clip.tokenize(a3).to(device)
        a3, _ = text_encoder(tokenized_text3)
        a3 = a3.repeat(batch_size, 1)
        
        
        sent_emb, word_emb = text_encoder(tokenized_text)
        sent_emb = sent_emb.repeat(batch_size,1)
        
        
        
        fake_imgs = netG(noise,sent_emb,a1,a2,a3,word_emb,eval=True).float()
        
        name = f'{captions[i].replace(" ", "-")}'
        vutils.save_image(fake_imgs.data, './samples/%s.png'%(name), nrow=8, value_range=(-1, 1), normalize=True)
        