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
        """
        # Import diffusion models
        sys.path.append(str(Path(__file__).parent.parent.parent.parent / "code"))
        from diffusion.ddpm import DDPM
        from diffusion.ddim import DDIMSampler
        from diffusion.unet import UNet

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint['config']

        # Build U-Net
        unet = UNet(
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            model_channels=config.model.model_channels,
            channel_mult=tuple(config.model.channel_mult),
            num_res_blocks=config.model.num_res_blocks,
            attention_resolutions=tuple(config.model.attention_resolutions),
            dropout=config.model.dropout,
            num_heads=config.model.num_heads
        )

        # Build DDPM
        ddpm = DDPM(
            model=unet,
            timesteps=config.diffusion.timesteps,
            schedule_type=config.diffusion.schedule_type,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            objective=config.diffusion.objective
        )

        # Load weights
        ddpm.load_state_dict(checkpoint['model_state_dict'])
        ddpm.to(self.device)
        ddpm.eval()

        # Store config and DDIM sampler
        self.config = config
        self.ddim = DDIMSampler(ddpm)

        return ddpm

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
            guidance_scale: Guidance scale for conditional generation (not used for unconditional)
            seed: Random seed

        Returns:
            List of PIL Images
        """
        # Set seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        print(f"🎨 Generating {num_samples} samples...")
        print(f"   Sampler: {sampler}")
        print(f"   Steps: {num_steps}")
        print(f"   Seed: {seed}")

        # Generate samples
        if sampler == 'ddim':
            samples_tensor = self.ddim.sample_from_batch(
                batch_size=num_samples,
                channels=self.config.model.in_channels,
                image_size=self.config.model.image_size,
                num_steps=num_steps,
                eta=0.0,  # Deterministic
                device=self.device,
                show_progress=True
            )
        else:  # ddpm
            samples_tensor = self.model.sample(
                batch_size=num_samples,
                channels=self.config.model.in_channels,
                image_size=self.config.model.image_size,
                device=self.device,
                show_progress=True
            )

        # Denormalize to [0, 1]
        samples_tensor = (samples_tensor + 1.0) / 2.0
        samples_tensor = torch.clamp(samples_tensor, 0.0, 1.0)

        # Convert to PIL images
        from PIL import Image
        import numpy as np

        samples = []
        for i in range(num_samples):
            # Convert to numpy (C, H, W) -> (H, W, C)
            img_np = samples_tensor[i].cpu().numpy()
            img_np = np.transpose(img_np, (1, 2, 0))

            # Convert to uint8
            img_np = (img_np * 255).astype(np.uint8)

            # Handle grayscale
            if img_np.shape[2] == 1:
                img_np = img_np.squeeze(-1)

            img = Image.fromarray(img_np)
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
