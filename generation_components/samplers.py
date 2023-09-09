from diffusers import DDIMPipeline

def ddim(steps):

    model_id = "google/ddpm-cifar10-32"

    # load model and scheduler
    ddim = DDIMPipeline.from_pretrained(model_id)

    # run pipeline in inference (sample random noise and denoise)
    image = ddim(num_inference_steps=steps).images[0]

def dpm_solver():
    print("Running DPM Solver")