import cv2
import torch
import einops
import random
import numpy as np
from pytorch_lightning import seed_everything
import config
import os


def process_core(prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta,
        detected_map, model, H, W, ddim_sampler, mode, map_filter):
    detected_map_list = [detected_map]
    if map_filter['type'] == 'Gaussian':
        detected_map_list += [cv2.GaussianBlur(detected_map, (k, k), 0) for k in map_filter["kernel"]]
    elif map_filter['type'] == 'bilateral':
        detected_map_list += [cv2.bilateralFilter(detected_map, *k) for k in map_filter["kernel"]]

    results, results_img = [], None
    for detected_map in detected_map_list:
        if mode == 'normal':
            detected_map = detected_map[:, :, ::-1]
        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        if mode == 'normal':
            detected_map = detected_map[:, :, ::-1]
        result = [detected_map] + [x_samples[i] for i in range(num_samples)]
        results += result
        if results_img is None:
            results_img = cv2.hconcat(result)
        else:
            results_img = cv2.vconcat([results_img, cv2.hconcat(result)])

    fileno = len([s for s in os.listdir('result_imgs') if s.startswith(mode) and s.endswith('.txt')])
    with open(f'result_imgs/{mode}{fileno}.txt', 'w') as f:
        if map_filter is not None:
            f.write(f'{map_filter["type"]}: {map_filter["kernel"]}\n')
        for param in [prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta]:
            f.write(f'{param}\n')
    cv2.imwrite(f'result_imgs/{mode}{fileno}.jpg', cv2.cvtColor(results_img, cv2.COLOR_RGB2BGR))
    return results
