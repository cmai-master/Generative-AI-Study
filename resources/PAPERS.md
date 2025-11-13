# 필수 논문 리스트

Generative Model 학습을 위한 필수 논문 리스트입니다. 중요도에 따라 ⭐ 표시가 되어 있습니다.

---

## 📌 기초 (Foundations)

### Deep Learning Basics
- **"Deep Learning"** Book - Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
- **"ImageNet Classification with Deep Convolutional Neural Networks"** - AlexNet (Krizhevsky et al., 2012) ⭐
- **"Very Deep Convolutional Networks for Large-Scale Image Recognition"** - VGG (Simonyan & Zisserman, 2014)
- **"Deep Residual Learning for Image Recognition"** - ResNet (He et al., 2015) ⭐

### Attention & Transformers
- **"Attention Is All You Need"** - Transformer (Vaswani et al., 2017) ⭐⭐⭐
  - [arXiv](https://arxiv.org/abs/1706.03762)
  - 모든 Generative Model의 기초가 되는 논문

---

## 🎨 Autoencoders & VAE

### Autoencoders
- **"Reducing the Dimensionality of Data with Neural Networks"** (Hinton & Salakhutdinov, 2006)
- **"Extracting and Composing Robust Features with Denoising Autoencoders"** (Vincent et al., 2008)

### Variational Autoencoders
- **"Auto-Encoding Variational Bayes"** - VAE (Kingma & Welling, 2013) ⭐⭐⭐
  - [arXiv](https://arxiv.org/abs/1312.6114)
  - VAE의 원조 논문, 반드시 읽어야 함

- **"β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"** (Higgins et al., 2017) ⭐
  - [arXiv](https://openreview.net/forum?id=Sy2fzU9gl)
  - Disentangled representation learning

- **"Neural Discrete Representation Learning"** - VQ-VAE (van den Oord et al., 2017) ⭐⭐
  - [arXiv](https://arxiv.org/abs/1711.00937)
  - Discrete latent representation

- **"Generating Diverse High-Fidelity Images with VQ-VAE-2"** (Razavi et al., 2019)
  - [arXiv](https://arxiv.org/abs/1906.00446)

---

## 🌊 Normalizing Flows

- **"Variational Inference with Normalizing Flows"** (Rezende & Mohamed, 2015)
  - [arXiv](https://arxiv.org/abs/1505.05770)

- **"NICE: Non-linear Independent Components Estimation"** (Dinh et al., 2014)
  - [arXiv](https://arxiv.org/abs/1410.8516)

- **"Density Estimation using Real NVP"** - RealNVP (Dinh et al., 2016) ⭐
  - [arXiv](https://arxiv.org/abs/1605.08803)

- **"Glow: Generative Flow using Invertible 1x1 Convolutions"** (Kingma & Dhariwal, 2018) ⭐⭐
  - [arXiv](https://arxiv.org/abs/1807.03039)

- **"Neural Ordinary Differential Equations"** (Chen et al., 2018) ⭐
  - [arXiv](https://arxiv.org/abs/1806.07366)
  - NeurIPS 2018 Best Paper

---

## 🎭 Generative Adversarial Networks (GANs)

### GAN Foundations
- **"Generative Adversarial Networks"** - Original GAN (Goodfellow et al., 2014) ⭐⭐⭐
  - [arXiv](https://arxiv.org/abs/1406.2661)
  - GAN의 원조 논문, 필수

- **"Unsupervised Representation Learning with Deep Convolutional GANs"** - DCGAN (Radford et al., 2015) ⭐⭐
  - [arXiv](https://arxiv.org/abs/1511.06434)
  - 안정적인 GAN 학습의 시작

### Conditional GANs
- **"Conditional Generative Adversarial Nets"** - cGAN (Mirza & Osindero, 2014) ⭐
  - [arXiv](https://arxiv.org/abs/1411.1784)

- **"Image-to-Image Translation with Conditional Adversarial Networks"** - Pix2Pix (Isola et al., 2016) ⭐⭐
  - [arXiv](https://arxiv.org/abs/1611.07004)

- **"Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"** - CycleGAN (Zhu et al., 2017) ⭐⭐
  - [arXiv](https://arxiv.org/abs/1703.10593)

### Improved Training
- **"Improved Techniques for Training GANs"** (Salimans et al., 2016) ⭐
  - [arXiv](https://arxiv.org/abs/1606.03498)

- **"Wasserstein GAN"** - WGAN (Arjovsky et al., 2017) ⭐⭐
  - [arXiv](https://arxiv.org/abs/1701.07875)

- **"Improved Training of Wasserstein GANs"** - WGAN-GP (Gulrajani et al., 2017) ⭐⭐
  - [arXiv](https://arxiv.org/abs/1704.00028)

- **"Spectral Normalization for Generative Adversarial Networks"** (Miyato et al., 2018) ⭐
  - [arXiv](https://arxiv.org/abs/1802.05957)

### StyleGAN Series
- **"A Style-Based Generator Architecture for Generative Adversarial Networks"** - StyleGAN (Karras et al., 2018) ⭐⭐⭐
  - [arXiv](https://arxiv.org/abs/1812.04948)
  - 고품질 이미지 생성의 혁명

- **"Analyzing and Improving the Image Quality of StyleGAN"** - StyleGAN2 (Karras et al., 2019) ⭐⭐⭐
  - [arXiv](https://arxiv.org/abs/1912.04958)

- **"Alias-Free Generative Adversarial Networks"** - StyleGAN3 (Karras et al., 2021) ⭐
  - [arXiv](https://arxiv.org/abs/2106.12423)

### Large-Scale GANs
- **"Progressive Growing of GANs for Improved Quality, Stability, and Variation"** (Karras et al., 2017) ⭐
  - [arXiv](https://arxiv.org/abs/1710.10196)

- **"Self-Attention Generative Adversarial Networks"** - SAGAN (Zhang et al., 2018) ⭐
  - [arXiv](https://arxiv.org/abs/1805.08318)

- **"Large Scale GAN Training for High Fidelity Natural Image Synthesis"** - BigGAN (Brock et al., 2018) ⭐⭐
  - [arXiv](https://arxiv.org/abs/1809.11096)

### Applications
- **"Photo-Realistic Single Image Super-Resolution Using a GAN"** - SRGAN (Ledig et al., 2016)
  - [arXiv](https://arxiv.org/abs/1609.04802)

---

## 🔄 Autoregressive Models

- **"Pixel Recurrent Neural Networks"** - PixelRNN (van den Oord et al., 2016) ⭐
  - [arXiv](https://arxiv.org/abs/1601.06759)

- **"Conditional Image Generation with PixelCNN Decoders"** - PixelCNN (van den Oord et al., 2016) ⭐
  - [arXiv](https://arxiv.org/abs/1606.05328)

- **"PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood"** (Salimans et al., 2017)
  - [arXiv](https://arxiv.org/abs/1701.05517)

- **"WaveNet: A Generative Model for Raw Audio"** (van den Oord et al., 2016) ⭐
  - [arXiv](https://arxiv.org/abs/1609.03499)

---

## 🌫️ Diffusion Models

### Foundational Papers
- **"Deep Unsupervised Learning using Nonequilibrium Thermodynamics"** (Sohl-Dickstein et al., 2015)
  - [arXiv](https://arxiv.org/abs/1503.03585)
  - Diffusion Model의 이론적 기초

- **"Denoising Diffusion Probabilistic Models"** - DDPM (Ho et al., 2020) ⭐⭐⭐
  - [arXiv](https://arxiv.org/abs/2006.11239)
  - 현대 Diffusion Model의 시작, 필수 논문

- **"Improved Denoising Diffusion Probabilistic Models"** (Nichol & Dhariwal, 2021) ⭐
  - [arXiv](https://arxiv.org/abs/2102.09672)

### Score-Based Models
- **"Generative Modeling by Estimating Gradients of the Data Distribution"** - NCSN (Song & Ermon, 2019)
  - [arXiv](https://arxiv.org/abs/1907.05600)

- **"Score-Based Generative Modeling through Stochastic Differential Equations"** (Song et al., 2021) ⭐⭐
  - [arXiv](https://arxiv.org/abs/2011.13456)
  - SDE 관점에서의 Diffusion

### Accelerated Sampling
- **"Denoising Diffusion Implicit Models"** - DDIM (Song et al., 2020) ⭐⭐
  - [arXiv](https://arxiv.org/abs/2010.02502)
  - 빠른 샘플링

- **"DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling"** (Lu et al., 2022)
  - [arXiv](https://arxiv.org/abs/2206.00927)

- **"Elucidating the Design Space of Diffusion-Based Generative Models"** - EDM (Karras et al., 2022) ⭐
  - [arXiv](https://arxiv.org/abs/2206.00364)

### Conditional & Guided Diffusion
- **"Diffusion Models Beat GANs on Image Synthesis"** (Dhariwal & Nichol, 2021) ⭐⭐
  - [arXiv](https://arxiv.org/abs/2105.05233)
  - Classifier guidance

- **"Classifier-Free Diffusion Guidance"** (Ho & Salimans, 2022) ⭐⭐⭐
  - [arXiv](https://arxiv.org/abs/2207.12598)
  - 현대 Text-to-Image의 핵심 기술

### Latent Diffusion
- **"High-Resolution Image Synthesis with Latent Diffusion Models"** - Stable Diffusion (Rombach et al., 2022) ⭐⭐⭐
  - [arXiv](https://arxiv.org/abs/2112.10752)
  - Stable Diffusion의 기반, 필수

### Text-to-Image Diffusion
- **"Hierarchical Text-Conditional Image Generation with CLIP Latents"** - DALL-E 2 (Ramesh et al., 2022) ⭐⭐⭐
  - [arXiv](https://arxiv.org/abs/2204.06125)

- **"Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding"** - Imagen (Saharia et al., 2022) ⭐⭐
  - [arXiv](https://arxiv.org/abs/2205.11487)

- **"SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis"** (Podell et al., 2023) ⭐
  - [arXiv](https://arxiv.org/abs/2307.01952)

### Controllable Generation
- **"Adding Conditional Control to Text-to-Image Diffusion Models"** - ControlNet (Zhang et al., 2023) ⭐⭐
  - [arXiv](https://arxiv.org/abs/2302.05543)

- **"T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models"** (Mou et al., 2023)
  - [arXiv](https://arxiv.org/abs/2302.08452)

### Personalization
- **"An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion"** (Gal et al., 2022) ⭐
  - [arXiv](https://arxiv.org/abs/2208.01618)

- **"DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation"** (Ruiz et al., 2022) ⭐⭐
  - [arXiv](https://arxiv.org/abs/2208.12242)

---

## 🤖 Large Language Models (LLMs)

### Transformer-based LLMs
- **"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"** (Devlin et al., 2018) ⭐⭐
  - [arXiv](https://arxiv.org/abs/1810.04805)

- **"Language Models are Unsupervised Multitask Learners"** - GPT-2 (Radford et al., 2019) ⭐⭐
  - [PDF](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

- **"Language Models are Few-Shot Learners"** - GPT-3 (Brown et al., 2020) ⭐⭐⭐
  - [arXiv](https://arxiv.org/abs/2005.14165)

### Instruction Following
- **"Training language models to follow instructions with human feedback"** - InstructGPT (Ouyang et al., 2022) ⭐⭐
  - [arXiv](https://arxiv.org/abs/2203.02155)
  - ChatGPT의 기반

### Open Source LLMs
- **"LLaMA: Open and Efficient Foundation Language Models"** (Touvron et al., 2023) ⭐⭐
  - [arXiv](https://arxiv.org/abs/2302.13971)

- **"Llama 2: Open Foundation and Fine-Tuned Chat Models"** (Touvron et al., 2023) ⭐⭐
  - [arXiv](https://arxiv.org/abs/2307.09288)

### Efficient Fine-tuning
- **"LoRA: Low-Rank Adaptation of Large Language Models"** (Hu et al., 2021) ⭐⭐
  - [arXiv](https://arxiv.org/abs/2106.09685)
  - 효율적인 fine-tuning의 표준

- **"QLoRA: Efficient Finetuning of Quantized LLMs"** (Dettmers et al., 2023) ⭐
  - [arXiv](https://arxiv.org/abs/2305.14314)

### Reasoning
- **"Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"** (Wei et al., 2022) ⭐⭐
  - [arXiv](https://arxiv.org/abs/2201.11903)

- **"Tree of Thoughts: Deliberate Problem Solving with Large Language Models"** (Yao et al., 2023)
  - [arXiv](https://arxiv.org/abs/2305.10601)

---

## 🖼️ Multimodal Models

### Vision-Language Pre-training
- **"Learning Transferable Visual Models From Natural Language Supervision"** - CLIP (Radford et al., 2021) ⭐⭐⭐
  - [arXiv](https://arxiv.org/abs/2103.00020)
  - 현대 멀티모달 모델의 기초

- **"Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision"** - ALIGN (Jia et al., 2021)
  - [arXiv](https://arxiv.org/abs/2102.05918)

- **"BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation"** (Li et al., 2022) ⭐
  - [arXiv](https://arxiv.org/abs/2201.12086)

- **"BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and LLMs"** (Li et al., 2023) ⭐
  - [arXiv](https://arxiv.org/abs/2301.12597)

### Visual Instruction Tuning
- **"Flamingo: a Visual Language Model for Few-Shot Learning"** (Alayrac et al., 2022) ⭐⭐
  - [arXiv](https://arxiv.org/abs/2204.14198)

- **"Visual Instruction Tuning"** - LLaVA (Liu et al., 2023) ⭐⭐
  - [arXiv](https://arxiv.org/abs/2304.08485)

- **"Improved Baselines with Visual Instruction Tuning"** - LLaVA-1.5 (Liu et al., 2023)
  - [arXiv](https://arxiv.org/abs/2310.03744)

### Image Generation (Text-to-Image)
- **"Zero-Shot Text-to-Image Generation"** - DALL-E (Ramesh et al., 2021) ⭐⭐
  - [arXiv](https://arxiv.org/abs/2102.12092)

---

## 🎬 Video Generation

- **"VideoGPT: Video Generation using VQ-VAE and Transformers"** (Yan et al., 2021)
  - [arXiv](https://arxiv.org/abs/2104.10157)

- **"Make-A-Video: Text-to-Video Generation without Text-Video Data"** (Singer et al., 2022) ⭐
  - [arXiv](https://arxiv.org/abs/2209.14792)

- **"Imagen Video: High Definition Video Generation with Diffusion Models"** (Ho et al., 2022)
  - [arXiv](https://arxiv.org/abs/2210.02303)

- **"Video Diffusion Models"** (Ho et al., 2022)
  - [arXiv](https://arxiv.org/abs/2204.03458)

---

## 🎵 Audio & Music Generation

- **"Jukebox: A Generative Model for Music"** (Dhariwal et al., 2020) ⭐
  - [arXiv](https://arxiv.org/abs/2005.00341)

- **"AudioLM: a Language Modeling Approach to Audio Generation"** (Borsos et al., 2022) ⭐
  - [arXiv](https://arxiv.org/abs/2209.03143)

- **"MusicLM: Generating Music From Text"** (Agostinelli et al., 2023) ⭐
  - [arXiv](https://arxiv.org/abs/2301.11325)

---

## 🏗️ 3D Generation

- **"NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"** (Mildenhall et al., 2020) ⭐⭐⭐
  - [arXiv](https://arxiv.org/abs/2003.08934)
  - 3D 생성의 혁명

- **"Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"** - Instant-NGP (Müller et al., 2022) ⭐
  - [arXiv](https://arxiv.org/abs/2201.05989)

- **"DreamFusion: Text-to-3D using 2D Diffusion"** (Poole et al., 2022) ⭐⭐
  - [arXiv](https://arxiv.org/abs/2209.14988)

- **"Magic3D: High-Resolution Text-to-3D Content Creation"** (Lin et al., 2022) ⭐
  - [arXiv](https://arxiv.org/abs/2211.10440)

---

## 📊 Evaluation Metrics

- **"Improved Techniques for Training GANs"** - Inception Score (Salimans et al., 2016)
  - [arXiv](https://arxiv.org/abs/1606.03498)

- **"GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"** - FID (Heusel et al., 2017) ⭐
  - [arXiv](https://arxiv.org/abs/1706.08500)

---

## 📖 읽기 순서 추천

### 초급 (1-3개월)
1. Attention Is All You Need (Transformer)
2. Auto-Encoding Variational Bayes (VAE)
3. Generative Adversarial Networks (GAN)
4. DCGAN
5. Denoising Diffusion Probabilistic Models (DDPM)

### 중급 (4-6개월)
6. StyleGAN / StyleGAN2
7. CLIP
8. High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)
9. Classifier-Free Diffusion Guidance
10. DALL-E 2

### 고급 (7-12개월)
11. Score-Based Generative Modeling through SDEs
12. GPT-3
13. InstructGPT
14. LLaMA 2
15. ControlNet
16. LLaVA
17. NeRF
18. DreamFusion

---

## 🔄 정기적으로 체크할 리소스

- **arXiv**: https://arxiv.org/
  - cs.CV (Computer Vision)
  - cs.LG (Machine Learning)
  - cs.AI (Artificial Intelligence)
  - cs.CL (Computation and Language)

- **Papers with Code**: https://paperswithcode.com/
  - State-of-the-art 모델 추적

- **Hugging Face Papers**: https://huggingface.co/papers
  - Daily trending papers

---

**마지막 업데이트**: 2025-01
