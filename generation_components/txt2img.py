import torch
from torchvision.transforms import functional as F
from diffusers import DDIMPipeline

def generate_image(model, prompt, guidance_scale, width, height, steps, sampler=None, clip_skip=False, seed=None, negative_prompt=None, output_path="generated_image.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()



    if not negative_prompt:
        negative_prompt = ""
    # Initialize the image with random noise
    image = torch.randn(1, 3, height, width).to(device)
    
    # Set a random seed for reproducibility if seed is provided (fix seed assignment)
    if seed is not None:
        torch.manual_seed(seed)
    
    # Define a default sampler if none is provided
    if sampler is None:
        def default_sampler(gradient):
            return gradient
        
        sampler = default_sampler
    
    for step in range(steps):
        # Generate a random vector for diversity
        z = torch.randn_like(image)
        
        # Calculate the loss with the positive prompt
        positive_loss = model(image, z, prompt, negative_prompt, guidance_scale)

        # Calculate the loss with the negative prompt if provided
        if negative_prompt:
            negative_loss = model(image, z, negative_prompt, None, guidance_scale)
            loss = positive_loss - negative_loss
        else:
            loss = positive_loss

        # Update the image using gradient ascent
        image.grad = None
        loss.backward()
        image.data += sampler(image.grad)

        # Clip the image values to be in the range [0, 1]
        image.data = torch.clamp(image.data, 0, 1)

        # Apply control net and clip skip (uncomment if needed)
        # image.data = clip_skip(image.data)

    # Convert the generated tensor to a PIL image
    generated_image = F.to_pil_image(image[0].cpu())
    
    # Save the generated image
    generated_image.save(output_path)
    
    return generated_image
