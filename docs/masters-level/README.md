# 석사급 Generative AI 완전 분석

> **대상**: 대학원생, 연구자, 논문 저자, AI 엔지니어
>
> **목표**: 각 생성 모델의 이론적 기반부터 최신 연구까지 **완벽하게** 이해하고 **직접 구현**할 수 있는 수준

---

## 📚 문서 구조

각 모델별로 다음 내용을 다룹니다:

### 1. 수학적 기초
- 확률론적 정의 및 정리
- 핵심 수식의 **완전한 유도** (모든 단계 포함)
- 정리의 **엄밀한 증명**
- 가정(assumption)의 의미와 완화 가능성

### 2. 이론적 분석
- 최적화 이론 관점
- 통계학적 관점 (편향-분산 트레이드오프 등)
- 정보 이론 관점 (상호정보량, 엔트로피 등)
- 계산 복잡도 분석

### 3. 실험 방법론
- 평가 지표의 수학적 의미
- 실험 설계 (Ablation study, Hyperparameter tuning)
- 통계적 유의성 검정
- 재현성 확보 방법

### 4. 구현 세부사항
- 수치적 안정성 (Numerical stability)
- 최적화 트릭 (Gradient clipping, Learning rate scheduling)
- 병렬화 및 효율성
- 디버깅 전략

### 5. 최신 연구
- 주요 논문 분석
- 연구 방향성
- 미해결 문제
- 향후 연구 주제

---

## 📖 모델별 완전 분석

### [1. VAE (Variational Autoencoder)](./vae-complete-theory.md)

**다루는 내용**:
- ✅ 잠재 변수 모델의 이론적 배경
- ✅ ELBO의 완전한 유도 (10+ 단계)
- ✅ Reparameterization Trick의 수학적 증명
- ✅ KL Divergence의 해석적 계산
- ✅ β-VAE와 Disentangled Representation
- ✅ VQ-VAE, NVAE 등 변형 모델
- ✅ 이론적 한계와 개선 방향
- ✅ Likelihood 기반 평가 vs Reconstruction 기반 평가

**핵심 정리**:
- 정리 1.1: Maximum Likelihood Estimation
- 정리 1.2: 사후 분포의 복잡성
- 정리 3.2: ELBO와 KL Divergence의 관계
- 정리 3.4: Diagonal Gaussian KL Divergence
- 정리 5.1: Reparameterization Gradient Estimator의 불편성

**수식 예시**:
```
log p_θ(x) = ELBO(θ, φ; x) + D_KL(q_φ(z|x) || p_θ(z|x))
          ≥ E_q [log p_θ(x|z)] - D_KL(q_φ(z|x) || p(z))
```

---

### [2. GAN (Generative Adversarial Networks)](./gan-complete-theory.md)

**다루는 내용**:
- ✅ Game Theory 관점 (Minimax game, Nash Equilibrium)
- ✅ Optimal Discriminator의 해석적 해
- ✅ Generator의 최적화 목적 함수 유도
- ✅ Mode Collapse의 수학적 분석
- ✅ Wasserstein GAN의 이론적 기반
- ✅ Lipschitz Constraint의 의미와 구현
- ✅ StyleGAN의 아키텍처 분석
- ✅ 평가 지표 (IS, FID, KID) 수학적 정의

**핵심 정리**:
- 정리 1: 최적 Discriminator (D* = p_data / (p_data + p_g))
- 정리 2: Global Optimality (p_g = p_data ⟺ C(G) = -log 4)
- 정리 3: Wasserstein Distance의 Dual Form
- 정리 4: 1-Lipschitz Constraint의 필요충분조건

**수식 예시**:
```
min_G max_D V(D, G) = E_{x~p_data}[log D(x)] + E_{z~p(z)}[log(1-D(G(z)))]

W(p_r, p_g) = inf_{γ∈Π(p_r,p_g)} E_{(x,y)~γ}[||x - y||]
```

---

### [3. Diffusion Models](./diffusion-complete-theory.md)

**다루는 내용**:
- ✅ 확률적 미분 방정식 (SDE) 관점
- ✅ Forward Process의 Markov Chain 분석
- ✅ Reverse Process의 Langevin Dynamics
- ✅ Score Matching의 수학적 원리
- ✅ Denoising Score Matching의 등가성 증명
- ✅ DDPM Loss의 유도 과정
- ✅ DDIM의 Non-Markovian Process
- ✅ Classifier Guidance vs Classifier-Free Guidance
- ✅ Latent Diffusion의 효율성 분석

**핵심 정리**:
- 정리 1: Forward Process의 Closed Form
- 정리 2: Reverse Process SDE
- 정리 3: Score Function의 특성화
- 정리 4: Denoising Score Matching Equivalence
- 정리 5: DDIM ODE의 Deterministic Mapping

**수식 예시**:
```
Forward SDE: dx = f(x,t)dt + g(t)dw
Reverse SDE: dx = [f(x,t) - g(t)²∇_x log p_t(x)]dt + g(t)dw̄

q(x_t|x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t)I)
```

---

### [4. Transformer-based Generative Models](./transformer-generative-theory.md)

**다루는 내용**:
- ✅ Attention Mechanism의 수학적 정의
- ✅ Self-Attention의 계산 복잡도 분석
- ✅ Positional Encoding의 이론적 근거
- ✅ GPT의 Autoregressive Modeling
- ✅ BERT의 Masked Language Modeling
- ✅ T5의 Encoder-Decoder 구조
- ✅ In-Context Learning의 메커니즘
- ✅ Scaling Laws의 수학적 모델링

---

## 🔬 공통 주제

### A. 최적화 이론

**다루는 내용**:
- Stochastic Gradient Descent의 수렴 분석
- Adam Optimizer의 수학적 원리
- Learning Rate Scheduling 전략
- Gradient Clipping과 Exploding Gradient
- Second-Order Optimization (Natural Gradient, K-FAC)

### B. 정규화 기법

**다루는 내용**:
- Batch Normalization의 통계적 의미
- Layer Normalization vs Instance Normalization
- Spectral Normalization의 Lipschitz Constraint
- Dropout의 Bayesian 해석
- Weight Decay vs L2 Regularization

### C. 평가 방법론

**다루는 내용**:
- Inception Score의 수학적 정의와 문제점
- Fréchet Inception Distance (FID)의 유도
- Precision-Recall for Generative Models
- Perceptual Path Length (PPL)
- Likelihood-based Metrics vs Sample Quality

---

## 📊 실험 재현

각 모델의 주요 논문 결과를 재현하는 완전한 코드:

### 1. VAE on MNIST/CelebA
- 원논문: Auto-Encoding Variational Bayes (Kingma & Welling, 2013)
- 재현 목표: 재구성 품질, Latent Space Interpolation
- 코드: `experiments/vae_reproduction/`

### 2. WGAN-GP on CIFAR-10
- 원논문: Improved Training of Wasserstein GANs (Gulrajani et al., 2017)
- 재현 목표: FID < 30, Mode Coverage
- 코드: `experiments/wgan_reproduction/`

### 3. DDPM on ImageNet 64×64
- 원논문: Denoising Diffusion Probabilistic Models (Ho et al., 2020)
- 재현 목표: FID ≈ 3.17
- 코드: `experiments/ddpm_reproduction/`

---

## 📝 학습 가이드

### Phase 1: 수학적 기초 (2-3주)
1. 확률론 복습 (베이즈 정리, 변분 추론)
2. 최적화 이론 (경사 하강법, KKT 조건)
3. 정보 이론 (엔트로피, KL Divergence, Mutual Information)

**추천 교재**:
- Pattern Recognition and Machine Learning (Bishop)
- Convex Optimization (Boyd & Vandenberghe)
- Elements of Information Theory (Cover & Thomas)

### Phase 2: VAE 완전 정복 (2-3주)
1. 논문 정독: Auto-Encoding Variational Bayes
2. 이론 학습: [vae-complete-theory.md](./vae-complete-theory.md)
3. 구현: MNIST → CelebA → Custom Dataset
4. 실험: β-VAE, Disentanglement Metrics

**연습 문제**:
1. ELBO를 3가지 다른 방법으로 유도하시오
2. KL Divergence의 비대칭성이 VAE에 미치는 영향을 분석하시오
3. β=0.5, 1, 4, 10일 때 잠재 공간 변화를 시각화하시오

### Phase 3: GAN 완전 정복 (3-4주)
1. 논문 정독:
   - Generative Adversarial Networks (Goodfellow et al.)
   - WGAN, WGAN-GP
   - StyleGAN, StyleGAN2
2. 이론 학습: [gan-complete-theory.md](./gan-complete-theory.md)
3. 구현: DCGAN → WGAN-GP → StyleGAN
4. 실험: Mode Collapse Analysis, FID Comparison

**연습 문제**:
1. Nash Equilibrium의 존재성을 증명하시오
2. Wasserstein Distance가 JS Divergence보다 나은 이유를 수학적으로 설명하시오
3. Spectral Normalization이 Lipschitz Constraint를 어떻게 강제하는지 증명하시오

### Phase 4: Diffusion Models 완전 정복 (3-4주)
1. 논문 정독:
   - DDPM (Ho et al.)
   - Score-Based SDE (Song et al.)
   - DDIM (Song et al.)
   - Stable Diffusion (Rombach et al.)
2. 이론 학습: [diffusion-complete-theory.md](./diffusion-complete-theory.md)
3. 구현: DDPM → DDIM → Latent Diffusion
4. 실험: Sampling Speed vs Quality Trade-off

**연습 문제**:
1. Forward Process가 Markov Chain임을 증명하시오
2. Reverse Process SDE를 유도하시오
3. DDIM이 왜 DDPM보다 빠른지 수학적으로 설명하시오

---

## 🎓 연구 방향

### 현재 연구 트렌드 (2024-2025)

1. **Unified Framework**
   - VAE, GAN, Diffusion을 통합하는 이론
   - Flow Matching과 Diffusion의 연결

2. **Efficiency**
   - Distillation (1-step generation)
   - Latent Space Optimization
   - Quantization and Pruning

3. **Controllability**
   - Disentangled Representations
   - Compositional Generation
   - Constraint-based Generation

4. **Scalability**
   - Large-scale Training (billions of parameters)
   - Distributed Training Strategies
   - Memory-efficient Architectures

### 미해결 문제

1. **이론적 문제**
   - VAE의 사후 붕괴 (Posterior Collapse) 완전한 해결
   - GAN의 수렴 보장
   - Diffusion의 샘플링 효율성 이론적 한계

2. **실용적 문제**
   - Few-shot Generation
   - Out-of-distribution Generalization
   - Fairness and Bias Mitigation

---

## 📚 필수 논문 리스트 (연대순)

### VAE 계열
1. **Auto-Encoding Variational Bayes** (Kingma & Welling, 2013) ⭐⭐⭐
2. **β-VAE** (Higgins et al., 2017) ⭐⭐
3. **VQ-VAE** (van den Oord et al., 2017) ⭐⭐
4. **VQ-VAE-2** (Razavi et al., 2019) ⭐
5. **NVAE** (Vahdat & Kautz, 2020) ⭐

### GAN 계열
1. **Generative Adversarial Networks** (Goodfellow et al., 2014) ⭐⭐⭐
2. **DCGAN** (Radford et al., 2015) ⭐⭐
3. **WGAN** (Arjovsky et al., 2017) ⭐⭐⭐
4. **WGAN-GP** (Gulrajani et al., 2017) ⭐⭐
5. **Progressive GAN** (Karras et al., 2017) ⭐⭐
6. **StyleGAN** (Karras et al., 2018) ⭐⭐⭐
7. **StyleGAN2** (Karras et al., 2019) ⭐⭐

### Diffusion 계열
1. **DDPM** (Ho et al., 2020) ⭐⭐⭐
2. **Score-Based SDE** (Song et al., 2021) ⭐⭐⭐
3. **DDIM** (Song et al., 2020) ⭐⭐
4. **Stable Diffusion** (Rombach et al., 2022) ⭐⭐⭐
5. **Classifier-Free Guidance** (Ho & Salimans, 2022) ⭐⭐

---

## 🛠 실험 환경

### 권장 사양
- **GPU**: A100 80GB (이상적), RTX 3090/4090 (실용적)
- **RAM**: 64GB+
- **Storage**: 1TB+ SSD

### 소프트웨어
```bash
python >= 3.9
pytorch >= 2.0
cuda >= 11.8
```

### 재현성 체크리스트
- [ ] Random seed 고정
- [ ] Deterministic algorithms 사용
- [ ] 학습 곡선 저장
- [ ] Checkpoint 저장
- [ ] Hyperparameters 기록
- [ ] 실험 환경 명시 (GPU, 라이브러리 버전)

---

## 💬 토론 및 질문

**자주 묻는 질문**:

1. **Q**: VAE의 ELBO를 최대화하면 왜 log p(x)도 증가하나요?
   **A**: ELBO는 log p(x)의 하한이며, 변분 갭이 항상 비음수이므로 ELBO 증가 → log p(x) 증가

2. **Q**: GAN은 왜 likelihood를 직접 최적화하지 않나요?
   **A**: Likelihood는 적분 불가능. 대신 adversarial loss로 분포 매칭

3. **Q**: Diffusion은 왜 VAE/GAN보다 느린가요?
   **A**: 반복적인 denoising 과정 필요 (수백~수천 step)

**토론 주제**:
- VAE vs GAN vs Diffusion: 언제 어떤 모델을 사용해야 하는가?
- Likelihood-based vs Adversarial: 어느 것이 더 나은 평가 방법인가?
- Discrete vs Continuous Latent Space: 각각의 장단점은?

---

**Last Update**: 2025-01
**Maintainer**: Generative AI Study Group
