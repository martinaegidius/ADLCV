import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT)
        self.vgg_relu4_4 = vgg.features[:27]

    def forward(self, input):
        return self.vgg_relu4_4(input)
    

class VFILoss(nn.Module):
    """Video frame interpolation loss
    Example of losses_dict:
    {
        'rec_loss' : weight of reconstruction loss loss (e.g. 0.5)
        'feature_loss': weight of feature loss (e.g. 0.5)
    }
    """
    def __init__(self, losses_dict, device='cuda'):
        super(VFILoss, self).__init__()
        self.device = device
        self.vgg = VGG()
        self.vgg.to(device)

        self.losses_dict = losses_dict

        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()


    def forward(self, input, target):
        """
        Input is the prediction (output of the model)
        Target is the target image.
        """
        ###using input as a dictionary of the format
        #    {'output_im': output, 'interp0': interp0, 'interp2': interp2,
        #    'u0': u0, 'v0': v0, 'u2': u2, 'v2': v2, 'w0': w0, 'w2': w2,
        #    'uvm': var_uvm}
        


        total_loss = 0
        for loss_func, weight in self.losses_dict.items():
            if loss_func == 'rec_loss': # Reconstruction loss
                tmp_loss = self.l1_loss(input["output_im"],target) # TASK 3

            elif loss_func == 'bidir_rec_loss': # Bidirectional reconstruction loss, equation 5 paper, is simply sum of L1 of interpolations and target with no weighting scheme
                tmp_loss = self.l1_loss(input["interp0"],target)+self.l1_loss(input["interp2"],target) # TASK 4

            elif loss_func == 'feature_loss': # Feature Loss
                tmp_loss = self.l2_loss(self.vgg(input["output_im"]),self.vgg(target)) # TASK 5
                
            else:
                raise AttributeError('Unknown loss: "' + loss_func + '"')
            total_loss += weight* tmp_loss
        
        return total_loss