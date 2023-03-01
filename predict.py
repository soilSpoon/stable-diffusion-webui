# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import json
from cog import BaseModel, BasePredictor, Input, Path
from modules import shared, modelloader, sd_models
import modules.scripts
from modules.processing import StableDiffusionProcessingTxt2Img, process_images
from modules.shared import opts


class Output(BaseModel):
    path: Path
    information: str


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # Setup models
        modelloader.cleanup_models()
        sd_models.setup_model()
        sd_models.load_model()

    def predict(
            self,
            prompt: str = Input(
                description="Input prompt",
                default="best quality, masterpiece, White hair,detailed, red eyes, windy, floating hair, snowy, "
                        "upper body, detailed face, winter, trees, sunshine",
            ),
            negative_prompt: str = Input(
                description="Specify things to not see in the output",
                default="(worst quality, low quality:1.4)",
            ),
            width: int = Input(
                description="Width of output image.",
                choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
                default=512,
            ),
            height: int = Input(
                description="Height of output image.",
                choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
                default=768,
            ),
            steps: int = Input(
                description="Number of denoising steps", ge=1, le=500, default=20
            ),
            cfg_scale: float = Input(
                description="Scale for classifier-free guidance", ge=1, le=20, default=11
            ),
            clip_skip: int = Input(
                description="", ge=1, le=12, default=2
            ),
            sampler_name: str = Input(
                default="Euler a",
                choices=[
                    'Euler a',
                    'Euler',
                    'LMS',
                    'Heun',
                    'DPM2',
                    'DPM2 a',
                    'DPM++ 2S a',
                    'DPM++ 2M',
                    'DPM++ SDE',
                    'DPM fast',
                    'DPM adaptive',
                    'LMS Karras',
                    'DPM2 Karras',
                    'DPM2 a Karras',
                    'DPM++ 2S a Karras',
                    'DPM++ 2M Karras',
                    'DPM++ SDE Karras',
                ],
                description="Choose a sampler.",
            ),
            seed: int = Input(
                description="Random seed. Leave blank to randomize the seed", default=None
            ),
            *args
    ) -> Output:
        """Run a single prediction on the model"""
        override_settings = {"CLIP_stop_at_last_layers": 2}

        p = StableDiffusionProcessingTxt2Img(
            sd_model=shared.sd_model,
            outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
            outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
            prompt=prompt,
            # styles=prompt_styles,
            negative_prompt=negative_prompt,
            seed=seed,
            sampler_name=sampler_name,
            steps=steps,
            cfg_scale=cfg_scale,
            width=width,
            height=height,
            override_settings=override_settings,
        )
        p.scripts = modules.scripts.scripts_txt2img
        p.script_args = args

        processed = process_images(p)
        p.close()

        shared.total_tqdm.clear()

        image = processed.images[0]
        output_path = f"/tmp/out.png"
        image.save(output_path)
        
        return Output(path=Path(output_path), information=json.dumps(processed.js()))
