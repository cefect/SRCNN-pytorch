

import argparse

import torch, os
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr




def eval_model(
        weights_file,
        image_file,
        scale=3,
        out_dir=os.getcwd(),
        ):
    """apply the trained model to some image"""
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'running on device: {device}')
#===========================================================================
# init
#===========================================================================
    model = SRCNN().to(device)
#===========================================================================
# load the saved parameters onto the model
#===========================================================================
    state_dict = model.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc:storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
    
    model.eval()
#===========================================================================
# load and pre-process image
#===========================================================================
    print(f'loading and pre-processing image from {image_file}')
    image = pil_image.open(image_file).convert('RGB')
    image_width = (image.width // scale) * scale
    image_height = (image.height // scale) * scale
    image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    image = image.resize((image.width // scale, image.height // scale), resample=pil_image.BICUBIC)
    image = image.resize((image.width * scale, image.height * scale), resample=pil_image.BICUBIC)
    image.save(image_file.replace('.', '_bicubic_x{}.'.format(scale)))
#===========================================================================
# prep the image
#===========================================================================
    image = np.array(image).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(image)
    y = ycbcr[..., 0]
    y /= 255.
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)
#===========================================================================
# downsacle with model
#===========================================================================
    print(f'downscaling')
    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)
#===========================================================================
# calc result
#===========================================================================
    psnr = calc_psnr(y, preds)
    print('PSNR: {:.2f}'.format(psnr))
    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
#===========================================================================
# save result
#===========================================================================
 
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    ofp = os.path.join(out_dir, 
        os.path.basename(image_file.replace('.', '_srcnn_x{}.'.format(scale))))
    output.save(ofp)
    print(f'image saved to {ofp}')

"""use the calibrated model to downscale an image"""




if __name__ == '__main__':
    #===========================================================================
    # setup
    #===========================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--outdir', type=str, default=os.getcwd())
    args = parser.parse_args()

    eval_model(args)
    
    
