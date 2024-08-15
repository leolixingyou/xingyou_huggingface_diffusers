# !pip install diffusers
import os
import time
import pathlib
from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir,exist_ok=True)

### batch num is setting in pipline_ddpm.py
foldor = [x for x in os.listdir('./xingyou_huggingface_diffusers/examples/unconditional_image_generation/hand_out_2024_08_07') if 'check' in x]
save_dir = f"/workspace/xingyou_huggingface_diffusers/examples/unconditional_image_generation/runs/{time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time()))}/"
mkdir(save_dir)

for i, fol in enumerate(foldor):
    print(f'here is {i}')
    model_id = pathlib.Path(f"./xingyou_huggingface_diffusers/examples/unconditional_image_generation/hand_out_2024_08_07/{fol}/")

    # load model and scheduler
    ddpm = DDPMPipeline.from_pretrained(model_id).to("cuda")  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference

    # run pipeline in inference (sample random noise and denoise)
    image = ddpm().images

    # save image
    for i in range(len(image)): 
        image[i].save(f'{save_dir}{fol}_{i}.png')