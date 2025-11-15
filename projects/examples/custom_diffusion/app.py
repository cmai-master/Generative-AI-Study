"""
Gradio Web UI for Custom Diffusion

Usage:
    python app.py --checkpoint path/to/model.pth
"""

import argparse
import torch
import gradio as gr
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import diffusion models (assuming they're in code/diffusion)
# from code.diffusion import DDPM, DDIMSampler, UNet


class DiffusionApp:
    """Gradio application for Diffusion Models"""

    def __init__(self, checkpoint_path, device='cuda'):
        """
        Initialize app

        Args:
            checkpoint_path: Path to model checkpoint
            device: 'cuda' or 'cpu'
        """
        self.device = device if torch.cuda.is_available() else 'cpu'

        print(f"🚀 Loading model from {checkpoint_path}...")
        self.model = self.load_model(checkpoint_path)

        print(f"✅ Model loaded successfully!")
        print(f"🖥️  Using device: {self.device}")

    def load_model(self, checkpoint_path):
        """
        Load model from checkpoint

        TODO: Replace with actual model loading
        """
        # Example:
        # from code.diffusion import DDPM, UNet
        #
        # unet = UNet(...)
        # ddpm = DDPM(model=unet, ...)
        #
        # checkpoint = torch.load(checkpoint_path)
        # ddpm.load_state_dict(checkpoint['model_state_dict'])
        # ddpm.to(self.device)
        # ddpm.eval()
        #
        # return ddpm

        # Placeholder
        print("⚠️  Using placeholder model. Replace with actual model loading!")
        return None

    @torch.no_grad()
    def generate_samples(self,
                        num_samples=4,
                        num_steps=50,
                        sampler='ddim',
                        guidance_scale=7.5,
                        seed=42):
        """
        Generate samples

        Args:
            num_samples: Number of images to generate
            num_steps: Sampling steps (fewer = faster)
            sampler: 'ddpm' or 'ddim'
            guidance_scale: Guidance scale for conditional generation
            seed: Random seed

        Returns:
            List of PIL Images
        """
        # Set seed
        torch.manual_seed(seed)

        print(f"🎨 Generating {num_samples} samples...")
        print(f"   Sampler: {sampler}")
        print(f"   Steps: {num_steps}")
        print(f"   Seed: {seed}")

        # TODO: Implement actual sampling
        # if sampler == 'ddim':
        #     ddim = DDIMSampler(self.model)
        #     samples = ddim.sample(
        #         batch_size=num_samples,
        #         num_steps=num_steps,
        #         device=self.device
        #     )
        # else:
        #     samples = self.model.sample(
        #         batch_size=num_samples,
        #         device=self.device
        #     )

        # Placeholder: return random images
        import numpy as np
        from PIL import Image

        samples = []
        for i in range(num_samples):
            # Random image (replace with actual samples)
            img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            samples.append(img)

        print(f"✅ Generated {len(samples)} samples!")

        return samples

    def create_interface(self):
        """Create Gradio interface"""

        # Wrapper function for gradio
        def generate_wrapper(num_samples, num_steps, sampler, guidance_scale, seed):
            images = self.generate_samples(
                num_samples=int(num_samples),
                num_steps=int(num_steps),
                sampler=sampler,
                guidance_scale=guidance_scale,
                seed=int(seed)
            )
            return images

        # Create Gradio interface
        interface = gr.Interface(
            fn=generate_wrapper,
            inputs=[
                gr.Slider(1, 16, value=4, step=1, label="Number of Samples"),
                gr.Slider(10, 1000, value=50, step=10, label="Sampling Steps",
                         info="Fewer steps = faster generation"),
                gr.Dropdown(['ddim', 'ddpm'], value='ddim', label="Sampler",
                           info="DDIM is much faster"),
                gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance Scale",
                         info="Higher = more faithful to condition (if applicable)"),
                gr.Number(value=42, label="Random Seed",
                         info="Same seed = same output"),
            ],
            outputs=gr.Gallery(label="Generated Images", columns=4),
            title="🎨 Custom Diffusion Image Generator",
            description="""
            Generate high-quality images using Diffusion Models.

            **Tips:**
            - Use DDIM sampler with 50 steps for fast generation
            - Increase steps (100-200) for higher quality
            - Try different seeds for variety
            """,
            examples=[
                [4, 50, "ddim", 7.5, 42],
                [8, 100, "ddim", 7.5, 123],
                [1, 200, "ddpm", 7.5, 456],
            ],
            theme=gr.themes.Soft(),
            allow_flagging="never"
        )

        return interface

    def launch(self, **kwargs):
        """Launch Gradio app"""
        interface = self.create_interface()

        print("\n" + "="*70)
        print("🚀 Launching Gradio App")
        print("="*70)
        print("📱 Access the UI at: http://localhost:7860")
        print("🌍 Or share publicly with: share=True")
        print("="*70 + "\n")

        interface.launch(**kwargs)


def main():
    parser = argparse.ArgumentParser(description='Gradio UI for Custom Diffusion')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--share', action='store_true',
                        help='Create public link')
    parser.add_argument('--port', type=int, default=7860,
                        help='Port to run on')

    args = parser.parse_args()

    # Create app
    app = DiffusionApp(
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    # Launch
    app.launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0"  # Allow external connections
    )


if __name__ == '__main__':
    main()
