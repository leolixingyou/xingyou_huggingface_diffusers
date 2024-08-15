# !pip install diffusers
import os
import time
import pathlib
from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir,exist_ok=True)


target_num = 100

## no use just for calculate
batch = 16
round_num = target_num // batch if target_num % batch == 0 else (target_num // batch )+1

select_folder = ['34500','31500','23000']

### batch num is setting in pipline_ddpm.py
folder = [x for x in os.listdir('./xingyou_huggingface_diffusers/examples/unconditional_image_generation/hand_out_2024_08_07') if 'check' in x for y in select_folder if y in x]

save_dir = f"/workspace/xingyou_huggingface_diffusers/examples/unconditional_image_generation/runs/{time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time()))}/"
[mkdir(f'{save_dir}{x}/') for x in folder]

for i, fol in enumerate(folder):
    print(f'here is {i}, folder is {fol}')
    count = 0
    for c in range(round_num):
        model_id = pathlib.Path(f"./xingyou_huggingface_diffusers/examples/unconditional_image_generation/hand_out_2024_08_07/{fol}/")

        # load model and scheduler
        ddpm = DDPMPipeline.from_pretrained(model_id).to("cuda")  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference

        # run pipeline in inference (sample random noise and denoise)
        image = ddpm().images

        # save image
        for i in range(len(image)): 
            image[i].save(f'{save_dir}{fol}_{count}.png')
            count+=1
        if count == target_num -1:
            break