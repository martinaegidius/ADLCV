import matplotlib.pyplot as plt

# torch
import torch
import torch.nn.functional as F

# custom imports
from ddpm import Diffusion
from model import Classifier, UNet
from dataset.helpers import *
from util import show, set_seed, CLASS_LABELS
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed()

# Load model
diffusion = Diffusion(img_size=16, T=500, beta_start=1e-4, beta_end=0.02, diff_type='DDPM-cg', device=device)

classifier = Classifier(
    img_size=16, c_in=3, labels=5,
    time_dim=256,channels=32, device=device
)
classifier.to(device)
classifier.eval()
classifier.load_state_dict(torch.load('weights/classifier/model.pth', map_location=device))

unet = UNet(device=device)
unet.eval()
unet.to(device)
unet.load_state_dict(torch.load('weights/DDPM/model.pth', map_location=device))
diffusion.classifier = classifier 

# Sample
y = torch.tensor([0,1,2,3,4], device=device)
time_steps = [200,100,50]
x_new, intermediates = diffusion.p_sample_loop(unet, 5, timesteps_to_save=time_steps,y=y,classifier_scale=0.1)
#imgs = [im_normalize(tens2image(x_gen.cpu())) for x_gen in x_new]
#show(imgs, fig_titles=CLASS_LABELS, title='Classifier Guidance t=0', save_path='assets/cg_samples_0.png')




time_steps = time_steps+[0]
for i, interm in enumerate(intermediates):
    imgs = [im_normalize(tens2image(x_gen.cpu())) for x_gen in interm]
    show(imgs, fig_titles=CLASS_LABELS, title=f'Classifier Guidance t={time_steps[i]}', save_path=f'assets/cg_samples_{time_steps[i]}.png')

