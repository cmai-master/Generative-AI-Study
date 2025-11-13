# Generative AI Study Curriculum

## 📚 커리큘럼 개요

이 커리큘럼은 Generative Model을 체계적으로 학습하기 위한 종합 가이드입니다. 기초 이론부터 최신 기술까지 단계별로 학습할 수 있도록 구성되어 있습니다.

---

## 🎯 학습 목표

- Generative Model의 핵심 개념과 수학적 원리 이해
- 주요 Generative Model 아키텍처 습득 (VAE, GAN, Diffusion, Transformer)
- 실전 프로젝트를 통한 실무 능력 배양
- 최신 연구 트렌드 파악 및 응용 능력 개발

---

## 📋 목차

1. [Phase 1: 수학 및 이론 기초](#phase-1-수학-및-이론-기초)
2. [Phase 2: 딥러닝 기초](#phase-2-딥러닝-기초)
3. [Phase 3: Generative Models - 기본](#phase-3-generative-models---기본)
4. [Phase 4: Generative Models - 고급](#phase-4-generative-models---고급)
5. [Phase 5: 최신 기술 및 응용](#phase-5-최신-기술-및-응용)
6. [Phase 6: 실전 프로젝트](#phase-6-실전-프로젝트)

---

## Phase 1: 수학 및 이론 기초

**예상 기간**: 3-4주

### Week 1-2: 확률론 및 통계

#### 학습 목표
- 확률 분포의 이해
- 베이즈 정리와 응용
- 정보 이론의 기초

#### 세부 주제

##### 1.1 확률의 기초
- [ ] 확률 공간과 확률 변수
- [ ] 결합 확률, 조건부 확률, 주변 확률
- [ ] 베이즈 정리 (Bayes' Theorem)
- [ ] 기댓값, 분산, 공분산

##### 1.2 주요 확률 분포
- [ ] 이산 분포: Bernoulli, Binomial, Categorical, Multinomial
- [ ] 연속 분포: Gaussian (Normal), Uniform, Exponential
- [ ] 다변량 정규 분포 (Multivariate Normal)
- [ ] Mixture Models

##### 1.3 정보 이론
- [ ] Entropy (엔트로피)
- [ ] Cross-Entropy
- [ ] KL Divergence (Kullback-Leibler Divergence)
- [ ] Mutual Information
- [ ] Jensen-Shannon Divergence

#### 실습
- [ ] NumPy로 확률 분포 시뮬레이션
- [ ] KL Divergence 계산 및 시각화
- [ ] 베이즈 정리를 활용한 간단한 추론 문제

#### 참고 자료
- **교재**:
  - "Pattern Recognition and Machine Learning" - Christopher Bishop (Chapter 1-2)
  - "Information Theory, Inference, and Learning Algorithms" - David MacKay
- **온라인 강의**:
  - Khan Academy - Probability and Statistics
  - MIT 6.041 - Probabilistic Systems Analysis

---

### Week 3-4: 선형대수 및 최적화

#### 학습 목표
- 딥러닝에 필요한 선형대수 개념 습득
- 최적화 알고리즘의 이해

#### 세부 주제

##### 1.4 선형대수
- [ ] 벡터와 행렬 연산
- [ ] 고유값과 고유벡터 (Eigenvalues & Eigenvectors)
- [ ] 특이값 분해 (SVD: Singular Value Decomposition)
- [ ] 행렬 분해 기법
- [ ] 선형 변환과 차원 축소

##### 1.5 미적분학
- [ ] 편미분 (Partial Derivatives)
- [ ] 그래디언트 (Gradient)
- [ ] 체인 룰 (Chain Rule)
- [ ] 야코비안 (Jacobian)과 헤시안 (Hessian)

##### 1.6 최적화 이론
- [ ] 경사 하강법 (Gradient Descent)
- [ ] 볼록 최적화 (Convex Optimization)
- [ ] 라그랑주 승수법 (Lagrange Multipliers)
- [ ] 변분 추론 (Variational Inference) 기초

#### 실습
- [ ] NumPy로 행렬 연산 구현
- [ ] 그래디언트 계산 및 시각화
- [ ] 간단한 최적화 문제 풀이

#### 참고 자료
- **교재**:
  - "Linear Algebra and Its Applications" - Gilbert Strang
  - "Convex Optimization" - Stephen Boyd
- **온라인 강의**:
  - MIT 18.06 - Linear Algebra
  - Stanford CS229 - Machine Learning (Optimization part)

---

## Phase 2: 딥러닝 기초

**예상 기간**: 4-5주

### Week 5-6: 신경망 기초

#### 학습 목표
- 신경망의 구조와 작동 원리 이해
- 역전파 알고리즘 습득
- PyTorch/TensorFlow 프레임워크 숙달

#### 세부 주제

##### 2.1 퍼셉트론과 다층 신경망
- [ ] 퍼셉트론 (Perceptron)
- [ ] 다층 퍼셉트론 (MLP: Multi-Layer Perceptron)
- [ ] 활성화 함수: Sigmoid, Tanh, ReLU, LeakyReLU, GELU
- [ ] Universal Approximation Theorem

##### 2.2 역전파와 최적화
- [ ] 역전파 알고리즘 (Backpropagation)
- [ ] 손실 함수 (Loss Functions)
- [ ] 최적화 알고리즘:
  - SGD (Stochastic Gradient Descent)
  - Momentum
  - Adam, AdamW
  - RMSprop
  - Learning Rate Scheduling

##### 2.3 정규화 기법
- [ ] L1/L2 Regularization
- [ ] Dropout
- [ ] Batch Normalization
- [ ] Layer Normalization
- [ ] Early Stopping

#### 실습
- [ ] PyTorch 기본 문법 학습
- [ ] 간단한 MLP 구현 (MNIST 분류)
- [ ] 역전파 알고리즘 직접 구현
- [ ] 다양한 최적화 알고리즘 비교 실험

#### 참고 자료
- **교재**:
  - "Deep Learning" - Ian Goodfellow, Yoshua Bengio (Chapter 6-8)
- **온라인 강의**:
  - Fast.ai - Practical Deep Learning
  - Stanford CS231n (Introduction part)

---

### Week 7-8: 합성곱 신경망 (CNN)

#### 학습 목표
- CNN의 구조와 원리 이해
- 이미지 처리를 위한 딥러닝 기법 습득

#### 세부 주제

##### 2.4 CNN 기초
- [ ] Convolution 연산의 원리
- [ ] Pooling (Max, Average)
- [ ] Stride와 Padding
- [ ] 전치 합성곱 (Transposed Convolution / Deconvolution)

##### 2.5 CNN 아키텍처
- [ ] LeNet
- [ ] AlexNet
- [ ] VGG
- [ ] ResNet (Residual Networks)
- [ ] Inception
- [ ] DenseNet

##### 2.6 고급 CNN 기법
- [ ] 1x1 Convolution
- [ ] Depthwise Separable Convolution
- [ ] Dilated Convolution
- [ ] Attention Mechanisms in CNN

#### 실습
- [ ] CNN으로 이미지 분류 (CIFAR-10)
- [ ] 전이 학습 (Transfer Learning) 실습
- [ ] Feature Visualization
- [ ] Class Activation Mapping (CAM, Grad-CAM)

#### 참고 자료
- **온라인 강의**:
  - Stanford CS231n - Convolutional Neural Networks

---

### Week 9: 순환 신경망 (RNN) 및 Transformer

#### 학습 목표
- 시퀀스 데이터 처리를 위한 신경망 이해
- Transformer 아키텍처 학습

#### 세부 주제

##### 2.7 RNN 기초
- [ ] Vanilla RNN
- [ ] LSTM (Long Short-Term Memory)
- [ ] GRU (Gated Recurrent Unit)
- [ ] Bidirectional RNN
- [ ] Sequence-to-Sequence Models

##### 2.8 Attention Mechanism
- [ ] Attention의 개념
- [ ] Self-Attention
- [ ] Multi-Head Attention
- [ ] Scaled Dot-Product Attention

##### 2.9 Transformer
- [ ] Transformer 아키텍처
- [ ] Positional Encoding
- [ ] Encoder-Decoder 구조
- [ ] BERT, GPT 개요

#### 실습
- [ ] LSTM으로 텍스트 생성
- [ ] Attention Mechanism 구현
- [ ] 간단한 Transformer 모델 구현

#### 참고 자료
- **논문**:
  - "Attention Is All You Need" (Vaswani et al., 2017)
- **온라인 강의**:
  - Stanford CS224n - NLP with Deep Learning

---

## Phase 3: Generative Models - 기본

**예상 기간**: 6-7주

### Week 10-11: Autoencoders

#### 학습 목표
- Autoencoder의 원리와 응용 이해
- 차원 축소 및 특징 학습

#### 세부 주제

##### 3.1 Basic Autoencoder
- [ ] Autoencoder의 구조 (Encoder-Decoder)
- [ ] 재구성 손실 (Reconstruction Loss)
- [ ] Latent Space (잠재 공간)
- [ ] Undercomplete vs Overcomplete Autoencoders

##### 3.2 Autoencoder 변형
- [ ] Sparse Autoencoder
- [ ] Denoising Autoencoder
- [ ] Contractive Autoencoder
- [ ] Stacked Autoencoder

##### 3.3 Convolutional Autoencoder
- [ ] CNN 기반 Encoder/Decoder
- [ ] Image Reconstruction
- [ ] Feature Learning

#### 실습
- [ ] 기본 Autoencoder 구현 (MNIST)
- [ ] Denoising Autoencoder 구현
- [ ] Latent Space Visualization
- [ ] 이미지 압축 및 복원

#### 참고 자료
- **교재**:
  - "Deep Learning" - Ian Goodfellow (Chapter 14)
- **논문**:
  - "Extracting and Composing Robust Features with Denoising Autoencoders" (Vincent et al., 2008)

---

### Week 12-14: Variational Autoencoders (VAE)

#### 학습 목표
- VAE의 이론적 배경 완전 이해
- 변분 추론의 원리 학습
- VAE를 활용한 생성 모델 구현

#### 세부 주제

##### 3.4 VAE 이론
- [ ] 확률적 잠재 변수 모델
- [ ] Evidence Lower Bound (ELBO)
- [ ] KL Divergence와 Reconstruction Loss
- [ ] Reparameterization Trick
- [ ] 변분 추론 (Variational Inference)

##### 3.5 VAE 아키텍처
- [ ] Encoder Network (Recognition Network)
- [ ] Decoder Network (Generative Network)
- [ ] 사전 분포와 사후 분포
- [ ] Gaussian VAE

##### 3.6 VAE 변형 모델
- [ ] β-VAE (Disentangled Representations)
- [ ] Conditional VAE (CVAE)
- [ ] Adversarial Autoencoder
- [ ] Vector Quantized VAE (VQ-VAE)
- [ ] VQ-VAE-2

##### 3.7 VAE 고급 주제
- [ ] Importance Weighted Autoencoder (IWAE)
- [ ] Normalizing Flows for VAE
- [ ] Hierarchical VAE
- [ ] Disentanglement Metrics

#### 실습
- [ ] 기본 VAE 구현 (MNIST/CIFAR-10)
- [ ] Latent Space Interpolation
- [ ] β-VAE 구현 및 Disentanglement 분석
- [ ] Conditional VAE로 조건부 생성
- [ ] VQ-VAE 구현

#### 참고 자료
- **논문**:
  - "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013) ⭐
  - "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" (Higgins et al., 2017)
  - "Neural Discrete Representation Learning" (van den Oord et al., 2017) - VQ-VAE
- **블로그**:
  - Lil'Log - "From Autoencoder to Beta-VAE"

---

### Week 15-16: Normalizing Flows

#### 학습 목표
- Normalizing Flow의 수학적 원리 이해
- 정확한 likelihood 계산 가능한 생성 모델 학습

#### 세부 주제

##### 3.8 Flow 기초 이론
- [ ] Change of Variables 정리
- [ ] Jacobian과 Log-Determinant
- [ ] Invertible Transformations
- [ ] Coupling Layers

##### 3.9 Flow 아키텍처
- [ ] NICE (Non-linear Independent Components Estimation)
- [ ] RealNVP (Real-valued Non-Volume Preserving)
- [ ] Glow
- [ ] Flow++

##### 3.10 고급 Flow 기법
- [ ] Autoregressive Flows
- [ ] Continuous Normalizing Flows (Neural ODE)
- [ ] Residual Flows

#### 실습
- [ ] 간단한 2D Flow 구현 및 시각화
- [ ] RealNVP 구현
- [ ] Glow 모델 구현
- [ ] Density Estimation 실험

#### 참고 자료
- **논문**:
  - "NICE: Non-linear Independent Components Estimation" (Dinh et al., 2014)
  - "Density Estimation using Real NVP" (Dinh et al., 2016)
  - "Glow: Generative Flow using Invertible 1x1 Convolutions" (Kingma & Dhariwal, 2018) ⭐
  - "Neural Ordinary Differential Equations" (Chen et al., 2018)

---

## Phase 4: Generative Models - 고급

**예상 기간**: 8-10주

### Week 17-20: Generative Adversarial Networks (GAN)

#### 학습 목표
- GAN의 핵심 원리와 학습 메커니즘 이해
- GAN 학습의 어려움과 해결 방법 습득
- 다양한 GAN 변형 모델 학습

#### 세부 주제

##### 4.1 GAN 기초 이론
- [ ] Minimax Game Theory
- [ ] Generator와 Discriminator
- [ ] Nash Equilibrium
- [ ] Mode Collapse 문제
- [ ] Vanishing Gradient 문제

##### 4.2 GAN 손실 함수
- [ ] Original GAN Loss
- [ ] Non-Saturating Loss
- [ ] Wasserstein Distance
- [ ] Least Squares GAN (LSGAN)
- [ ] Hinge Loss

##### 4.3 GAN 학습 안정화 기법
- [ ] Spectral Normalization
- [ ] Gradient Penalty
- [ ] Two Time-Scale Update Rule (TTUR)
- [ ] Self-Attention
- [ ] Progressive Growing

##### 4.4 주요 GAN 아키텍처
- [ ] DCGAN (Deep Convolutional GAN)
- [ ] Conditional GAN (cGAN)
- [ ] Pix2Pix
- [ ] CycleGAN
- [ ] WGAN (Wasserstein GAN)
- [ ] WGAN-GP (Gradient Penalty)
- [ ] Progressive GAN
- [ ] StyleGAN, StyleGAN2, StyleGAN3
- [ ] BigGAN
- [ ] Self-Attention GAN (SAGAN)

##### 4.5 GAN 평가 지표
- [ ] Inception Score (IS)
- [ ] Fréchet Inception Distance (FID)
- [ ] Precision and Recall
- [ ] Kernel Inception Distance (KID)
- [ ] Perceptual Path Length (PPL)

##### 4.6 고급 GAN 응용
- [ ] Image-to-Image Translation
- [ ] Style Transfer
- [ ] Super Resolution (SRGAN, ESRGAN)
- [ ] Text-to-Image (AttnGAN, StackGAN)
- [ ] 3D Generation

#### 실습
- [ ] 기본 GAN 구현 (MNIST)
- [ ] DCGAN 구현 (CelebA)
- [ ] Conditional GAN 구현
- [ ] WGAN-GP 구현
- [ ] StyleGAN2 fine-tuning
- [ ] CycleGAN으로 스타일 변환
- [ ] FID Score 계산 구현

#### 참고 자료
- **논문**:
  - "Generative Adversarial Networks" (Goodfellow et al., 2014) ⭐⭐⭐
  - "Unsupervised Representation Learning with DCGAN" (Radford et al., 2015)
  - "Conditional Generative Adversarial Nets" (Mirza & Osindero, 2014)
  - "Wasserstein GAN" (Arjovsky et al., 2017)
  - "Improved Training of Wasserstein GANs" (Gulrajani et al., 2017)
  - "A Style-Based Generator Architecture for GANs" (Karras et al., 2018) - StyleGAN ⭐⭐
  - "Analyzing and Improving StyleGAN" (Karras et al., 2019) - StyleGAN2
- **GitHub**:
  - Official StyleGAN2 Implementation
- **온라인 강의**:
  - Stanford CS236 - Deep Generative Models

---

### Week 21-23: Autoregressive Models

#### 학습 목표
- Autoregressive 모델의 원리 이해
- 시퀀스 생성 모델링 기법 습득

#### 세부 주제

##### 4.7 Autoregressive 기초
- [ ] 조건부 독립성
- [ ] Chain Rule of Probability
- [ ] Maximum Likelihood Estimation
- [ ] Teacher Forcing

##### 4.8 주요 Autoregressive 모델
- [ ] PixelRNN
- [ ] PixelCNN
- [ ] PixelCNN++
- [ ] Gated PixelCNN
- [ ] WaveNet
- [ ] Transformer-XL
- [ ] GPT Series (GPT-1, GPT-2, GPT-3)

##### 4.9 고급 기법
- [ ] Masked Autoregressive Flows (MAF)
- [ ] Inverse Autoregressive Flows (IAF)
- [ ] Parallel WaveNet
- [ ] Fast Autoregressive Generation

#### 실습
- [ ] PixelCNN 구현 (MNIST)
- [ ] Character-level Language Model (RNN/LSTM)
- [ ] GPT-2 fine-tuning
- [ ] Conditional Generation

#### 참고 자료
- **논문**:
  - "Pixel Recurrent Neural Networks" (van den Oord et al., 2016)
  - "Conditional Image Generation with PixelCNN Decoders" (van den Oord et al., 2016)
  - "WaveNet: A Generative Model for Raw Audio" (van den Oord et al., 2016)
  - "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019) - GPT-2

---

### Week 24-26: Diffusion Models

#### 학습 목표
- Diffusion Model의 이론적 배경 완전 이해
- Score-based Models와의 관계 파악
- 최신 고성능 생성 모델 구현

#### 세부 주제

##### 4.10 Diffusion 기초 이론
- [ ] Forward Diffusion Process (노이즈 추가)
- [ ] Reverse Diffusion Process (노이즈 제거)
- [ ] Markov Chain
- [ ] Denoising Score Matching
- [ ] Langevin Dynamics

##### 4.11 수학적 배경
- [ ] Variance Schedule (β schedule)
- [ ] Evidence Lower Bound (ELBO)
- [ ] Variational Inference 관점
- [ ] Reparameterization of Loss
- [ ] Noise Prediction vs. Data Prediction

##### 4.12 주요 Diffusion 모델
- [ ] DDPM (Denoising Diffusion Probabilistic Models)
- [ ] DDIM (Denoising Diffusion Implicit Models)
- [ ] Score-based Generative Models (NCSN)
- [ ] SDE-based Diffusion Models
- [ ] Improved DDPM
- [ ] Classifier Guidance
- [ ] Classifier-Free Guidance

##### 4.13 고급 Diffusion 아키텍처
- [ ] U-Net Architecture for Diffusion
- [ ] Attention in Diffusion Models
- [ ] Latent Diffusion Models (LDM)
- [ ] Stable Diffusion
- [ ] Cascaded Diffusion Models
- [ ] DALL-E 2
- [ ] Imagen

##### 4.14 가속화 기법
- [ ] DDIM Sampling (Fast Sampling)
- [ ] DPM-Solver
- [ ] Consistency Models
- [ ] Progressive Distillation
- [ ] EDM (Elucidating Diffusion Models)

##### 4.15 Conditional Generation
- [ ] Text-to-Image Diffusion
- [ ] CLIP Guidance
- [ ] Cross-Attention Conditioning
- [ ] ControlNet
- [ ] T2I-Adapter

#### 실습
- [ ] DDPM 구현 (MNIST/CIFAR-10)
- [ ] DDIM 샘플링 구현
- [ ] Classifier-Free Guidance 구현
- [ ] Stable Diffusion fine-tuning
- [ ] ControlNet 활용
- [ ] Custom Dataset으로 Diffusion 모델 학습

#### 참고 자료
- **논문**:
  - "Deep Unsupervised Learning using Nonequilibrium Thermodynamics" (Sohl-Dickstein et al., 2015)
  - "Denoising Diffusion Probabilistic Models" (Ho et al., 2020) ⭐⭐⭐
  - "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021)
  - "Denoising Diffusion Implicit Models" (Song et al., 2020)
  - "Score-Based Generative Modeling through SDEs" (Song et al., 2021) ⭐⭐
  - "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., 2022) - Stable Diffusion ⭐⭐⭐
  - "Classifier-Free Diffusion Guidance" (Ho & Salimans, 2022)
  - "Hierarchical Text-Conditional Image Generation with CLIP Latents" (Ramesh et al., 2022) - DALL-E 2
  - "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding" (Saharia et al., 2022) - Imagen
- **블로그**:
  - Lil'Log - "What are Diffusion Models?"
  - Hugging Face - Diffusion Models Course
- **GitHub**:
  - Hugging Face Diffusers Library

---

## Phase 5: 최신 기술 및 응용

**예상 기간**: 6-8주

### Week 27-29: Large Language Models (LLM)

#### 학습 목표
- 대규모 언어 모델의 구조와 학습 방법 이해
- Prompt Engineering 및 Fine-tuning 기법 습득

#### 세부 주제

##### 5.1 Transformer 기반 LLM
- [ ] GPT Architecture (Decoder-only)
- [ ] BERT Architecture (Encoder-only)
- [ ] T5 Architecture (Encoder-Decoder)
- [ ] Scaling Laws
- [ ] Emergent Abilities

##### 5.2 주요 LLM 모델
- [ ] GPT-3, GPT-3.5, GPT-4
- [ ] LLaMA, LLaMA 2
- [ ] Claude
- [ ] PaLM, PaLM 2
- [ ] Gemini
- [ ] Mistral, Mixtral

##### 5.3 학습 기법
- [ ] Pre-training (Unsupervised Learning)
- [ ] Instruction Tuning
- [ ] RLHF (Reinforcement Learning from Human Feedback)
- [ ] PPO (Proximal Policy Optimization)
- [ ] DPO (Direct Preference Optimization)
- [ ] Constitutional AI

##### 5.4 효율적 Fine-tuning
- [ ] LoRA (Low-Rank Adaptation)
- [ ] QLoRA (Quantized LoRA)
- [ ] Prefix Tuning
- [ ] P-Tuning
- [ ] Adapter Layers

##### 5.5 Prompt Engineering
- [ ] Zero-shot Prompting
- [ ] Few-shot Prompting
- [ ] Chain-of-Thought (CoT)
- [ ] Tree of Thoughts
- [ ] ReAct Prompting
- [ ] Self-Consistency

##### 5.6 LLM 응용
- [ ] Text Generation
- [ ] Code Generation (Codex, Code Llama)
- [ ] Conversational AI
- [ ] Retrieval-Augmented Generation (RAG)
- [ ] Function Calling
- [ ] Agents and Tool Use

#### 실습
- [ ] GPT-2 fine-tuning
- [ ] LoRA를 이용한 LLaMA fine-tuning
- [ ] RAG 시스템 구축
- [ ] Prompt Engineering 실습
- [ ] LangChain을 이용한 Agent 구현

#### 참고 자료
- **논문**:
  - "Attention Is All You Need" (Vaswani et al., 2017) ⭐⭐⭐
  - "Language Models are Few-Shot Learners" (Brown et al., 2020) - GPT-3 ⭐⭐
  - "Training language models to follow instructions with human feedback" (Ouyang et al., 2022) - InstructGPT
  - "LLaMA: Open and Efficient Foundation Language Models" (Touvron et al., 2023)
  - "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
  - "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (Wei et al., 2022)
- **온라인 리소스**:
  - Hugging Face Transformers Documentation
  - LangChain Documentation

---

### Week 30-32: Multimodal Generative Models

#### 학습 목표
- 멀티모달 학습의 원리 이해
- 텍스트-이미지, 이미지-텍스트 생성 모델 학습

#### 세부 주제

##### 5.7 Vision-Language Models
- [ ] CLIP (Contrastive Language-Image Pre-training)
- [ ] ALIGN
- [ ] BLIP, BLIP-2
- [ ] Flamingo
- [ ] LLaVA
- [ ] GPT-4V

##### 5.8 Text-to-Image 모델
- [ ] DALL-E
- [ ] DALL-E 2
- [ ] DALL-E 3
- [ ] Stable Diffusion (복습)
- [ ] Imagen (복습)
- [ ] Midjourney (개념)
- [ ] Firefly

##### 5.9 Image-to-Text 모델
- [ ] Image Captioning
- [ ] Visual Question Answering (VQA)
- [ ] Vision-Language Navigation

##### 5.10 고급 멀티모달 기법
- [ ] Cross-Modal Attention
- [ ] Contrastive Learning
- [ ] Vision-Language Pre-training
- [ ] Adapter-based Multimodal Fusion

##### 5.11 기타 모달리티
- [ ] Text-to-Video (Runway, Sora 개념)
- [ ] Text-to-3D (DreamFusion, Magic3D)
- [ ] Text-to-Audio (AudioLM, MusicLM)
- [ ] Multimodal LLM

#### 실습
- [ ] CLIP 사용법 및 Zero-shot Classification
- [ ] Stable Diffusion + CLIP Guidance
- [ ] Image Captioning 모델 구현
- [ ] LLaVA fine-tuning
- [ ] Custom Text-to-Image 모델 학습

#### 참고 자료
- **논문**:
  - "Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., 2021) - CLIP ⭐⭐
  - "DALL-E: Creating Images from Text" (Ramesh et al., 2021)
  - "Flamingo: a Visual Language Model for Few-Shot Learning" (Alayrac et al., 2022)
  - "Visual Instruction Tuning" (Liu et al., 2023) - LLaVA
- **GitHub**:
  - OpenAI CLIP
  - LLaVA

---

### Week 33-34: Advanced Topics

#### 학습 목표
- 최신 연구 트렌드 파악
- 특수 도메인 응용 학습

#### 세부 주제

##### 5.12 3D Generation
- [ ] NeRF (Neural Radiance Fields)
- [ ] Instant-NGP
- [ ] DreamFusion
- [ ] Magic3D
- [ ] GET3D
- [ ] Point-E, Shap-E

##### 5.13 Video Generation
- [ ] VideoGPT
- [ ] NUWA
- [ ] Make-A-Video
- [ ] Phenaki
- [ ] Runway Gen-2
- [ ] Sora (OpenAI)

##### 5.14 Audio/Music Generation
- [ ] WaveNet (복습)
- [ ] Jukebox
- [ ] MusicLM
- [ ] AudioLM
- [ ] Riffusion
- [ ] MusicGen

##### 5.15 Model Editing & Control
- [ ] GAN Inversion
- [ ] StyleGAN Editing
- [ ] Textual Inversion
- [ ] DreamBooth
- [ ] LoRA for Diffusion
- [ ] ControlNet (복습)
- [ ] IP-Adapter

##### 5.16 Efficiency & Compression
- [ ] Knowledge Distillation
- [ ] Quantization
- [ ] Pruning
- [ ] Low-Rank Factorization
- [ ] Mixed Precision Training

##### 5.17 Ethics & Safety
- [ ] Bias in Generative Models
- [ ] Deepfake Detection
- [ ] Watermarking
- [ ] Content Moderation
- [ ] AI Safety

#### 실습
- [ ] NeRF 구현 및 실험
- [ ] DreamBooth로 커스텀 모델 학습
- [ ] ControlNet 활용한 이미지 생성
- [ ] Model Quantization 실습
- [ ] Bias 분석 및 완화 기법

#### 참고 자료
- **논문**:
  - "NeRF: Representing Scenes as Neural Radiance Fields" (Mildenhall et al., 2020)
  - "DreamBooth: Fine Tuning Text-to-Image Diffusion Models" (Ruiz et al., 2022)
  - "Adding Conditional Control to Text-to-Image Diffusion Models" (Zhang et al., 2023) - ControlNet
  - "Sora: Video generation models as world simulators" (OpenAI, 2024)

---

## Phase 6: 실전 프로젝트

**예상 기간**: 6-8주

### Week 35-42: 종합 프로젝트

#### 프로젝트 아이디어

##### Project 1: Custom Image Generation System
**난이도**: 중급
- [ ] 특정 도메인(예: 웹툰, 일러스트)의 데이터셋 수집
- [ ] Stable Diffusion fine-tuning
- [ ] LoRA 학습
- [ ] ControlNet 통합
- [ ] Gradio/Streamlit으로 Web UI 구축

**기술 스택**: Stable Diffusion, LoRA, ControlNet, Gradio

##### Project 2: AI Chatbot with RAG
**난이도**: 중급
- [ ] 특정 도메인 지식베이스 구축
- [ ] Vector Database 구축 (Pinecone, Chroma)
- [ ] LLM fine-tuning (LoRA)
- [ ] RAG 파이프라인 구축
- [ ] 대화형 인터페이스 개발

**기술 스택**: LLaMA 2, LangChain, Vector DB, FastAPI

##### Project 3: Style Transfer & Image Editing
**난이도**: 중급
- [ ] GAN 기반 스타일 변환 시스템
- [ ] 실시간 이미지 편집 기능
- [ ] Multiple Style Mixing
- [ ] User-friendly Interface

**기술 스택**: StyleGAN2, CycleGAN, Streamlit

##### Project 4: Text-to-Image Generation Pipeline
**난이도**: 고급
- [ ] 다중 모델 앙상블 (CLIP + Diffusion)
- [ ] Prompt Engineering Automation
- [ ] Image Quality Enhancement
- [ ] Batch Processing System
- [ ] API 서버 구축

**기술 스택**: Stable Diffusion, CLIP, FastAPI, Docker

##### Project 5: Video Generation System
**난이도**: 고급
- [ ] 이미지 시퀀스 생성
- [ ] Frame Interpolation
- [ ] Temporal Consistency
- [ ] Audio Synchronization
- [ ] Post-processing Pipeline

**기술 스택**: Stable Diffusion, Frame Interpolation Models, FFmpeg

##### Project 6: Multimodal Content Creation Tool
**난이도**: 고급
- [ ] Text → Image → Video Pipeline
- [ ] Image Captioning 역기능
- [ ] Style Consistency 유지
- [ ] Interactive Editing
- [ ] Cloud Deployment

**기술 스택**: LLM, Stable Diffusion, LLaVA, AWS/GCP

##### Project 7: 3D Asset Generation
**난이도**: 고급
- [ ] Text-to-3D Generation
- [ ] NeRF 기반 3D Reconstruction
- [ ] Texture Generation
- [ ] Mesh Optimization
- [ ] Rendering Pipeline

**기술 스택**: NeRF, DreamFusion, Blender API

##### Project 8: Music/Audio Generation System
**난이도**: 중급
- [ ] 장르별 음악 생성
- [ ] 멜로디 변형
- [ ] Audio Style Transfer
- [ ] Interactive Composition Tool

**기술 스택**: MusicGen, AudioLM, Magenta

---

## 📊 평가 및 포트폴리오

### 학습 평가 기준
- [ ] 주요 논문 읽기 및 리뷰 (최소 30편)
- [ ] 모델 구현 (최소 10개)
- [ ] 프로젝트 완성 (최소 2개)
- [ ] 기술 블로그 작성

### 포트폴리오 구성 요소
1. **GitHub Repository**
   - 코드 정리 및 문서화
   - README 작성
   - 실행 가능한 예제

2. **Technical Blog**
   - 학습 내용 정리
   - 논문 리뷰
   - 프로젝트 설명

3. **Demo/Presentation**
   - Hugging Face Spaces
   - YouTube Demo Video
   - 발표 자료

---

## 🛠 개발 환경 설정

### 필수 도구

#### Python 환경
```bash
# Python 3.8+ 권장
python --version

# 가상환경 생성
conda create -n genai python=3.10
conda activate genai
```

#### 주요 라이브러리
```bash
# PyTorch (CUDA 11.8 기준)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 기본 라이브러리
pip install numpy pandas matplotlib seaborn
pip install jupyter notebook ipywidgets

# 딥러닝 유틸리티
pip install tensorboard wandb
pip install torchsummary torchinfo
pip install timm einops

# Generative Models
pip install diffusers transformers accelerate
pip install datasets
pip install gradio streamlit

# 기타
pip install opencv-python pillow
pip install tqdm scikit-learn scipy
```

### 하드웨어 권장 사양
- **최소**: GPU 8GB VRAM (RTX 3060 Ti 이상)
- **권장**: GPU 12GB+ VRAM (RTX 3090, RTX 4090, A5000)
- **이상적**: GPU 24GB+ VRAM (RTX 3090 x2, A100)
- **RAM**: 32GB 이상
- **Storage**: SSD 500GB 이상

### 클라우드 옵션
- **Google Colab** (무료, GPU 제한적)
- **Kaggle Notebooks** (무료, GPU 주당 30시간)
- **Paperspace Gradient** (유료, 합리적 가격)
- **AWS SageMaker** (유료, 강력한 성능)
- **Google Cloud Platform** (유료)
- **Lambda Labs** (유료, GPU 특화)

---

## 📚 핵심 참고 자료

### 교재
1. **"Deep Learning"** - Ian Goodfellow, Yoshua Bengio, Aaron Courville
2. **"Pattern Recognition and Machine Learning"** - Christopher Bishop
3. **"Probabilistic Machine Learning"** - Kevin Murphy
4. **"Understanding Deep Learning"** - Simon J.D. Prince

### 온라인 강의
1. **Stanford CS236**: Deep Generative Models
2. **Stanford CS231n**: Convolutional Neural Networks
3. **Stanford CS224n**: NLP with Deep Learning
4. **Berkeley CS294-158**: Deep Unsupervised Learning
5. **Fast.ai**: Practical Deep Learning
6. **Hugging Face Course**: NLP, Diffusion Models

### 블로그 & 리소스
1. **Lil'Log**: https://lilianweng.github.io/
2. **Distill.pub**: https://distill.pub/
3. **Papers with Code**: https://paperswithcode.com/
4. **Hugging Face**: https://huggingface.co/
5. **Google AI Blog**: https://ai.googleblog.com/
6. **OpenAI Blog**: https://openai.com/blog/

### YouTube 채널
1. **Yannic Kilcher**: 논문 리뷰
2. **Two Minute Papers**: 최신 연구 요약
3. **Arxiv Insights**: 심층 논문 분석
4. **AI Coffee Break**: AI 뉴스 및 논문

---

## 🎯 학습 팁

### 효율적인 학습 전략

1. **이론과 실습의 균형**
   - 논문 읽기 40%
   - 코드 구현 40%
   - 프로젝트 20%

2. **논문 읽기 방법**
   - 첫 읽기: Abstract, Introduction, Conclusion
   - 두 번째: Method, Experiments
   - 세 번째: 수식 및 세부사항
   - 구현하면서 이해도 점검

3. **코드 구현 전략**
   - 간단한 버전부터 시작
   - 기존 구현 참고 (하지만 직접 작성)
   - 디버깅 및 시각화
   - 실험 및 개선

4. **시간 관리**
   - 매일 2-3시간 꾸준히
   - 주말에 집중 학습 세션
   - 정기적인 복습

5. **커뮤니티 활용**
   - Reddit (r/MachineLearning, r/LocalLLaMA)
   - Discord 서버
   - Twitter/X AI 커뮤니티
   - 논문 스터디 그룹

---

## 📝 학습 체크리스트

### Phase 1: 기초 (완료 후 체크)
- [ ] 확률론 및 통계 이해
- [ ] 선형대수 및 최적화 이해
- [ ] 정보 이론 개념 숙지

### Phase 2: 딥러닝 (완료 후 체크)
- [ ] MLP 구현 및 학습
- [ ] CNN 아키텍처 이해
- [ ] RNN/LSTM 이해
- [ ] Transformer 구조 이해

### Phase 3: Generative Models 기본 (완료 후 체크)
- [ ] Autoencoder 구현
- [ ] VAE 이론 및 구현
- [ ] Normalizing Flow 이해

### Phase 4: Generative Models 고급 (완료 후 체크)
- [ ] GAN 구현 및 학습
- [ ] StyleGAN 이해
- [ ] Autoregressive 모델 이해
- [ ] Diffusion Model 이론 및 구현
- [ ] Stable Diffusion 활용

### Phase 5: 최신 기술 (완료 후 체크)
- [ ] LLM 이해 및 fine-tuning
- [ ] Multimodal 모델 활용
- [ ] Advanced Topics 탐구

### Phase 6: 프로젝트 (완료 후 체크)
- [ ] 개인 프로젝트 2개 이상 완성
- [ ] 포트폴리오 구축
- [ ] 블로그 작성

---

## 🚀 다음 단계

### 커리큘럼 완료 후
1. **최신 연구 팔로우**
   - arXiv daily
   - 주요 학회 (NeurIPS, ICML, ICLR, CVPR)

2. **오픈소스 기여**
   - Hugging Face
   - PyTorch
   - 관련 라이브러리

3. **커뮤니티 활동**
   - 기술 블로그 운영
   - 발표 및 세미나
   - 논문 스터디 주도

4. **경력 개발**
   - Kaggle 대회 참여
   - AI 해커톤
   - 연구 인턴십
   - 취업/이직

---

## 📞 리소스 및 지원

### 질문 & 도움
- **Stack Overflow**: 기술적 질문
- **GitHub Issues**: 코드 관련 문제
- **Reddit**: 개념 토론
- **Discord**: 실시간 도움

### 데이터셋
- **ImageNet**: 이미지 분류
- **COCO**: 객체 인식, 캡셔닝
- **CelebA**: 얼굴 이미지
- **FFHQ**: 고품질 얼굴
- **LAION**: 대규모 이미지-텍스트
- **Common Crawl**: 텍스트 데이터

---

## 🎓 마치며

이 커리큘럼은 **체계적이고 깊이 있는 학습**을 위해 설계되었습니다.

**중요한 점**:
- 🐢 **서두르지 말고** 각 단계를 충분히 이해하고 넘어가세요
- 💪 **꾸준함**이 가장 중요합니다
- 🤝 **커뮤니티**와 함께 학습하세요
- 🔬 **실험**을 두려워하지 마세요
- 📝 **기록**하고 공유하세요

**성공적인 학습을 기원합니다!** 🚀✨

---

**최종 업데이트**: 2025-01
**버전**: 1.0
**라이선스**: MIT
