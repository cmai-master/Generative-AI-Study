# 실전 프로젝트 가이드

Generative AI 학습을 위한 실전 프로젝트 아이디어와 구현 가이드입니다.

---

## 📋 목차

1. [초급 프로젝트](#초급-프로젝트)
2. [중급 프로젝트](#중급-프로젝트)
3. [고급 프로젝트](#고급-프로젝트)
4. [포트폴리오 구성](#포트폴리오-구성)

---

## 🟢 초급 프로젝트

### Project 1: MNIST 숫자 생성기

**목표**: VAE와 GAN의 기초 학습

**난이도**: ⭐

**기간**: 1-2주

**기술 스택**:
- PyTorch
- Matplotlib
- Jupyter Notebook

**구현 단계**:

1. **VAE 구현**
```python
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 400),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
```

2. **GAN 구현**
3. **두 모델 비교**
4. **Latent Space 시각화**

**학습 포인트**:
- VAE의 ELBO 이해
- GAN의 minimax game 이해
- 생성 품질 평가

---

### Project 2: 얼굴 표정 변환

**목표**: Image-to-Image Translation 학습

**난이도**: ⭐⭐

**기간**: 2-3주

**데이터셋**:
- CelebA
- UTKFace

**기술 스택**:
- PyTorch
- Pix2Pix / CycleGAN
- OpenCV

**구현 단계**:

1. **데이터 준비**
   - 얼굴 이미지 수집
   - 전처리 (크롭, 리사이즈)
   - 데이터 증강

2. **Pix2Pix 구현**
   - Generator: U-Net
   - Discriminator: PatchGAN
   - L1 Loss + Adversarial Loss

3. **학습 및 평가**

**학습 포인트**:
- Conditional GAN
- U-Net 아키텍처
- PatchGAN Discriminator

---

### Project 3: 텍스트 생성 모델

**목표**: RNN/LSTM을 이용한 문장 생성

**난이도**: ⭐⭐

**기간**: 2주

**데이터셋**:
- Shakespeare 텍스트
- 한국어 시 데이터

**구현 단계**:

1. **데이터 전처리**
```python
class TextDataset(Dataset):
    def __init__(self, text, seq_length=100):
        self.text = text
        self.seq_length = seq_length

        # 문자 -> 인덱스 매핑
        self.chars = sorted(list(set(text)))
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}

    def __len__(self):
        return len(self.text) - self.seq_length

    def __getitem__(self, idx):
        chunk = self.text[idx:idx + self.seq_length + 1]
        input_seq = [self.char2idx[ch] for ch in chunk[:-1]]
        target_seq = [self.char2idx[ch] for ch in chunk[1:]]
        return torch.tensor(input_seq), torch.tensor(target_seq)
```

2. **LSTM 모델 구현**
3. **Temperature Sampling**
4. **텍스트 생성**

**학습 포인트**:
- Sequence Modeling
- Teacher Forcing
- Temperature Sampling

---

## 🟡 중급 프로젝트

### Project 4: 커스텀 이미지 생성 시스템

**목표**: Stable Diffusion을 활용한 특정 도메인 이미지 생성

**난이도**: ⭐⭐⭐

**기간**: 4-6주

**기술 스택**:
- Stable Diffusion
- LoRA
- DreamBooth
- Gradio

**프로젝트 구조**:
```
custom-image-generator/
├── data/
│   ├── training_images/     # 학습 이미지
│   └── captions.txt          # 캡션
├── models/
│   ├── base/                 # 베이스 모델
│   └── lora/                 # LoRA 가중치
├── scripts/
│   ├── train_lora.py
│   ├── inference.py
│   └── preprocess.py
├── app.py                    # Gradio UI
└── requirements.txt
```

**구현 단계**:

#### 1. 데이터 수집 및 준비
```python
# 이미지 크롤링 (합법적인 소스에서만!)
# 또는 직접 촬영/제작

# 캡션 생성 (BLIP2 사용)
from transformers import Blip2Processor, Blip2ForConditionalGeneration

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

def generate_caption(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors="pt")
    generated_ids = model.generate(**inputs, max_length=50)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return caption
```

#### 2. LoRA 학습
```python
# train_lora.py
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
import torch

# 모델 로드
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)

# LoRA 설정
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,  # LoRA rank
    lora_alpha=32,
    target_modules=["to_q", "to_v"],
    lora_dropout=0.1,
)

# 학습 루프
# ... (세부 구현)
```

#### 3. Gradio UI 구축
```python
# app.py
import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe.load_lora_weights("./models/lora")
pipe = pipe.to("cuda")

def generate_image(prompt, negative_prompt, num_steps, guidance_scale):
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
    ).images[0]
    return image

demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Textbox(label="Negative Prompt"),
        gr.Slider(20, 100, value=50, label="Steps"),
        gr.Slider(1, 20, value=7.5, label="Guidance Scale"),
    ],
    outputs=gr.Image(label="Generated Image"),
    title="Custom Image Generator",
)

demo.launch()
```

#### 4. 배포
- Hugging Face Spaces
- AWS EC2
- Docker 컨테이너

**학습 포인트**:
- Fine-tuning 기법
- LoRA vs Full Fine-tuning
- Prompt Engineering
- UI/UX 디자인

---

### Project 5: AI 챗봇 with RAG

**목표**: 특정 도메인 지식을 가진 대화형 AI 구축

**난이도**: ⭐⭐⭐

**기간**: 4-6주

**기술 스택**:
- LLaMA 2 / Mistral
- LangChain
- ChromaDB / Pinecone
- FastAPI
- Streamlit

**시스템 아키텍처**:
```
User Query → Embedding → Vector Search →
Context Retrieval → LLM (with context) → Response
```

**구현 단계**:

#### 1. 데이터 준비
```python
# document_processor.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, PDFLoader

# 문서 로드
loader = DirectoryLoader('./docs', glob="**/*.pdf", loader_cls=PDFLoader)
documents = loader.load()

# 청크 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(documents)
```

#### 2. Vector Database 구축
```python
# vectorstore.py
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# 임베딩 모델
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector Store 생성
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

#### 3. RAG Chain 구성
```python
# rag_chain.py
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# LLM 로드
llm = LlamaCpp(
    model_path="./models/llama-2-7b-chat.gguf",
    temperature=0.7,
    max_tokens=512,
    n_ctx=2048,
)

# Prompt 템플릿
template = """다음 컨텍스트를 사용하여 질문에 답하세요.
컨텍스트에 답이 없다면, 모른다고 말하세요.

컨텍스트: {context}

질문: {question}

답변:"""

PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": PROMPT}
)
```

#### 4. Streamlit UI
```python
# app.py
import streamlit as st

st.title("도메인 전문 AI 챗봇")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if prompt := st.chat_input("질문을 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 응답
    with st.chat_message("assistant"):
        response = qa_chain.run(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
```

**학습 포인트**:
- RAG 시스템 설계
- Vector Database
- LLM Fine-tuning (선택사항)
- Prompt Engineering

---

### Project 6: 스타일 변환 시스템

**목표**: CycleGAN을 활용한 도메인 간 이미지 변환

**난이도**: ⭐⭐⭐

**기간**: 3-4주

**예시**:
- 사진 → 고흐 스타일
- 낮 → 밤
- 여름 → 겨울
- 말 → 얼룩말

**기술 스택**:
- PyTorch
- CycleGAN
- Gradio

**핵심 구현**:

```python
# cyclegan.py
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # ResNet-based generator
        # ... (구현)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # PatchGAN discriminator
        # ... (구현)

# Cycle Consistency Loss
cycle_loss = nn.L1Loss()(reconstructed_A, real_A)

# Total Loss
loss_G = loss_GAN + lambda_cycle * cycle_loss + lambda_identity * identity_loss
```

**학습 포인트**:
- Unpaired Image Translation
- Cycle Consistency
- Identity Loss

---

## 🔴 고급 프로젝트

### Project 7: Text-to-Image 생성 시스템

**목표**: 처음부터 구축하는 Text-to-Image 모델

**난이도**: ⭐⭐⭐⭐⭐

**기간**: 8-12주

**기술 스택**:
- PyTorch
- CLIP
- Latent Diffusion
- Accelerate (분산 학습)

**시스템 구성**:
1. Text Encoder (CLIP)
2. Image Encoder (VAE)
3. Diffusion Model (U-Net)
4. Sampling Algorithm

**구현 단계**:

#### 1. VAE 학습
```python
# vae.py
class AutoencoderKL(nn.Module):
    def __init__(self, in_channels=3, latent_channels=4):
        super().__init__()
        # Encoder
        self.encoder = Encoder(in_channels, latent_channels)
        # Decoder
        self.decoder = Decoder(latent_channels, in_channels)
        # ... (구현)
```

#### 2. U-Net 구현
```python
# unet.py
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Time embedding
        self.time_mlp = nn.Sequential(...)
        # Cross-attention for text conditioning
        self.cross_attn = CrossAttention(...)
        # ... (구현)
```

#### 3. 학습 파이프라인
```python
# train.py
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="fp16")

# 모델, 옵티마이저, 데이터로더 준비
model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

for epoch in range(num_epochs):
    for batch in train_dataloader:
        # Forward diffusion
        noise = torch.randn_like(batch["latents"])
        timesteps = torch.randint(0, num_train_timesteps, (batch_size,))
        noisy_latents = noise_scheduler.add_noise(batch["latents"], noise, timesteps)

        # Predict noise
        noise_pred = model(noisy_latents, timesteps, batch["text_embeddings"])

        # Loss
        loss = F.mse_loss(noise_pred, noise)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
```

**학습 포인트**:
- 대규모 모델 학습
- 분산 학습
- Mixed Precision Training
- Gradient Accumulation

---

### Project 8: 3D 객체 생성

**목표**: Text-to-3D 생성 시스템

**난이도**: ⭐⭐⭐⭐⭐

**기간**: 10-12주

**기술 스택**:
- PyTorch
- NeRF
- Stable Diffusion
- Three.js (시각화)

**접근 방법**:
1. **DreamFusion 방식**: Score Distillation Sampling
2. **Point-E**: Point Cloud → Mesh
3. **Shap-E**: Direct 3D Generation

**핵심 아이디어 (DreamFusion)**:
```python
# dreamfusion.py
def score_distillation_loss(nerf_render, text_embedding, diffusion_model):
    # NeRF 렌더링
    rendered_image = nerf_render(camera_pose)

    # Diffusion model로 gradient 계산
    with torch.no_grad():
        noise = diffusion_model.add_noise(rendered_image, t)
        noise_pred = diffusion_model(noise, t, text_embedding)

    # SDS loss
    grad = (noise_pred - noise)
    loss = (rendered_image * grad).sum()

    return loss
```

**학습 포인트**:
- NeRF 원리
- Score Distillation
- 3D Representation
- Rendering

---

### Project 9: Video Generation System

**목표**: Text-to-Video 생성

**난이도**: ⭐⭐⭐⭐⭐

**기간**: 10-12주

**접근 방법**:
1. Frame-by-frame generation
2. Temporal consistency module
3. Frame interpolation

**파이프라인**:
```
Text Prompt →
Keyframe Generation (Stable Diffusion) →
Temporal Interpolation →
Frame Smoothing →
Video Output
```

**핵심 구현**:
```python
# video_generator.py
class VideoGenerator:
    def __init__(self):
        self.image_gen = StableDiffusionPipeline.from_pretrained(...)
        self.interpolator = FrameInterpolationModel(...)

    def generate_video(self, prompt, num_frames=60, fps=30):
        # 키프레임 생성
        keyframes = []
        for i in range(0, num_frames, 10):
            frame = self.image_gen(prompt, seed=i).images[0]
            keyframes.append(frame)

        # 프레임 보간
        all_frames = []
        for i in range(len(keyframes) - 1):
            interpolated = self.interpolator(keyframes[i], keyframes[i+1], num_frames=10)
            all_frames.extend(interpolated)

        # 비디오 저장
        save_video(all_frames, fps=fps, output_path="output.mp4")
```

**학습 포인트**:
- Temporal Consistency
- Frame Interpolation
- Video Processing

---

## 📂 포트폴리오 구성

### 1. GitHub Repository

#### README.md 구조
```markdown
# Project Title

## 📌 Overview
- 프로젝트 소개
- 데모 GIF/이미지

## 🎯 Objectives
- 학습 목표

## 🛠 Tech Stack
- 사용 기술

## 📊 Results
- 정량적 결과
- 정성적 평가

## 🚀 Quick Start
\```bash
# 설치 방법
# 실행 방법
\```

## 📁 Project Structure
- 디렉토리 구조

## 📝 Methodology
- 접근 방법
- 핵심 알고리즘

## 🔍 Experiments
- 실험 결과
- Ablation Study

## 📚 References
- 참고 논문
- 참고 자료
```

### 2. Hugging Face Spaces

```python
# app.py for Hugging Face Spaces
import gradio as gr

# 모델 로드
model = load_model()

def generate(input):
    output = model(input)
    return output

demo = gr.Interface(
    fn=generate,
    inputs=...,
    outputs=...,
    title="My Generative AI Project",
    description="...",
    examples=[...],
)

demo.launch()
```

### 3. 기술 블로그

**추천 플랫폼**:
- Medium
- Velog
- GitHub Pages + Jekyll
- Notion

**블로그 포스트 구조**:
1. **Introduction**
   - 동기
   - 문제 정의

2. **Background**
   - 관련 연구
   - 이론적 배경

3. **Methodology**
   - 접근 방법
   - 구현 세부사항

4. **Experiments**
   - 실험 설정
   - 결과 분석

5. **Conclusion**
   - 배운 점
   - 향후 계획

### 4. Demo Video

**구성**:
- 0:00-0:15: 프로젝트 소개
- 0:15-1:00: 시연
- 1:00-1:30: 기술 설명
- 1:30-2:00: 결과 및 향후 계획

**도구**:
- OBS Studio (화면 녹화)
- DaVinci Resolve (편집)
- Loom (간단한 데모)

---

## 📊 프로젝트 평가 기준

### 기술적 깊이
- [ ] 알고리즘 이해도
- [ ] 코드 품질
- [ ] 최적화 노력

### 창의성
- [ ] 새로운 아이디어
- [ ] 문제 해결 접근법
- [ ] 응용 능력

### 완성도
- [ ] 프로젝트 완료 여부
- [ ] 문서화 수준
- [ ] 재현 가능성

### 결과물
- [ ] 정량적 성능
- [ ] 정성적 품질
- [ ] 실용성

---

**프로젝트를 통해 실력을 키우고 포트폴리오를 완성하세요!**

**마지막 업데이트**: 2025-01
