# Transformer 기반 생성 모델 이론

> **난이도**: 석사 수준 (Masters Level)
> **선수 지식**: 선형대수, 확률론, 딥러닝 기초, Transformer 아키텍처
> **목표**: Transformer를 이용한 생성 모델의 수학적 원리를 완전히 이해하고, Autoregressive modeling의 본질을 파악한다

---

## 📚 목차

1. [Autoregressive Modeling 수학적 기초](#1-autoregressive-modeling-수학적-기초)
2. [Attention Mechanism 완전 정복](#2-attention-mechanism-완전-정복)
3. [Transformer 아키텍처](#3-transformer-아키텍처)
4. [GPT: Decoder-only Transformer](#4-gpt-decoder-only-transformer)
5. [Autoregressive Image Generation](#5-autoregressive-image-generation)
6. [BERT와 Masked Language Modeling](#6-bert와-masked-language-modeling)
7. [Vision Transformer for Generation](#7-vision-transformer-for-generation)
8. [Scaling Laws와 Emergent Abilities](#8-scaling-laws와-emergent-abilities)
9. [구현 및 실습](#9-구현-및-실습)

---

## 1. Autoregressive Modeling 수학적 기초

### 1.1 정의

**Autoregressive (AR) 모델**은 현재 값을 과거 값들의 함수로 표현하는 모델입니다.

#### 수학적 정의

시퀀스 **x** = (x₁, x₂, ..., x_T)의 결합 확률 분포:

$$
p(\mathbf{x}) = p(x_1, x_2, \ldots, x_T)
$$

**확률의 Chain Rule**을 적용하면:

$$
p(\mathbf{x}) = p(x_1) \prod_{t=2}^{T} p(x_t \mid x_1, \ldots, x_{t-1})
$$

이를 간단히:

$$
p(\mathbf{x}) = \prod_{t=1}^{T} p(x_t \mid \mathbf{x}_{<t})
$$

여기서 **x_{<t}** = (x₁, ..., x_{t-1})는 t 이전의 모든 토큰.

### 1.2 Autoregressive Property

**핵심 아이디어**: 각 토큰은 **이전 토큰들에만** 의존

$$
x_t \perp \mathbf{x}_{>t} \mid \mathbf{x}_{<t}
$$

**장점:**
- 확률 계산이 명확: exact likelihood
- 샘플링이 간단: 왼쪽→오른쪽으로 순차 생성

**단점:**
- 병렬 생성 불가 (느림)
- Long-range dependency 학습 어려움

### 1.3 Maximum Likelihood Estimation

#### 목적 함수

데이터셋 D = {**x**⁽¹⁾, ..., **x**⁽ᴺ⁾}가 주어졌을 때:

$$
\mathcal{L}(\theta) = \sum_{n=1}^{N} \log p_\theta(\mathbf{x}^{(n)})
$$

Chain rule 적용:

$$
\mathcal{L}(\theta) = \sum_{n=1}^{N} \sum_{t=1}^{T} \log p_\theta(x_t^{(n)} \mid \mathbf{x}_{<t}^{(n)})
$$

**Negative Log-Likelihood (NLL) Loss**:

$$
\mathcal{L}_{NLL} = -\frac{1}{NT} \sum_{n=1}^{N} \sum_{t=1}^{T} \log p_\theta(x_t^{(n)} \mid \mathbf{x}_{<t}^{(n)})
$$

#### Cross-Entropy Loss

분류 문제로 보면 (vocabulary V에 대해):

$$
\mathcal{L}_{CE} = -\sum_{t=1}^{T} \sum_{v \in V} \mathbb{1}[x_t = v] \log p_\theta(x_t = v \mid \mathbf{x}_{<t})
$$

**One-hot encoding**으로 y_t ∈ ℝ^|V|:

$$
\mathcal{L}_{CE} = -\sum_{t=1}^{T} \mathbf{y}_t^\top \log \mathbf{\hat{y}}_t
$$

### 1.4 Teacher Forcing

#### 정의

학습 시, 모델이 **이전 예측**이 아닌 **실제 정답 토큰**을 입력으로 사용:

```
학습 시:
x_t의 예측 ← f(x_1, x_2, ..., x_{t-1})  [실제 정답 사용]

추론 시:
x_t의 예측 ← f(ŷ_1, ŷ_2, ..., ŷ_{t-1})  [모델 예측 사용]
```

**장점:**
- 학습 안정성 ↑
- 학습 속도 ↑ (병렬화 가능)

**단점:**
- Train-test mismatch (exposure bias)
- 추론 시 오류 누적

#### Scheduled Sampling

학습 중 점진적으로 teacher forcing 줄이기:

$$
p(\text{use ground truth at step } t) = \epsilon_t
$$

여기서 ε_t는 epoch에 따라 감소.

### 1.5 Perplexity

**Perplexity (PPL)**는 언어 모델의 표준 평가 지표:

$$
\text{PPL} = \exp\left(-\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t \mid \mathbf{x}_{<t})\right)
$$

**해석:**
- "모델이 얼마나 놀랐는가"
- 낮을수록 좋음
- 예: PPL=10 → 평균 10개 단어 중 하나 정도로 확신

---

## 2. Attention Mechanism 완전 정복

### 2.1 Attention이란?

**Attention**은 입력 시퀀스의 **관련 있는 부분에 집중**하는 메커니즘.

#### Motivation

RNN의 문제:
- 긴 시퀀스에서 초기 정보 손실
- 고정된 크기의 hidden state에 모든 정보 압축

**해결책:** 각 step마다 입력 전체를 보고 관련 정보만 추출!

### 2.2 Scaled Dot-Product Attention

#### 정의

**Query (Q)**, **Key (K)**, **Value (V)** 세 벡터로 계산:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

**차원:**
- Q ∈ ℝ^{n×d_k}: n개 query, 차원 d_k
- K ∈ ℝ^{m×d_k}: m개 key
- V ∈ ℝ^{m×d_v}: m개 value

**결과:** ℝ^{n×d_v}

#### 단계별 계산

1. **Similarity 계산**: QK^T ∈ ℝ^{n×m}
   - 각 query와 모든 key의 내적

2. **Scaling**: QK^T / √d_k
   - 내적 값이 너무 커지는 것 방지
   - Softmax의 gradient vanishing 방지

3. **Attention Weight**: softmax(QK^T / √d_k) ∈ ℝ^{n×m}
   - 각 query에 대한 key들의 확률 분포

4. **Weighted Sum**: Attention × V ∈ ℝ^{n×d_v}
   - Value들을 weight로 가중합

#### 왜 √d_k로 나누나?

**수학적 이유:**

Q, K의 원소가 평균 0, 분산 1이면:

$$
\text{Var}(QK^\top) = \sum_{i=1}^{d_k} \text{Var}(q_i) \text{Var}(k_i) = d_k
$$

따라서 √d_k로 나누면:

$$
\text{Var}\left(\frac{QK^\top}{\sqrt{d_k}}\right) = 1
$$

Softmax 입력의 분산이 일정하게 유지!

### 2.3 Multi-Head Attention

#### 정의

여러 개의 attention을 병렬로 계산:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

여기서:

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

**파라미터:**
- W_i^Q ∈ ℝ^{d_{model}×d_k}
- W_i^K ∈ ℝ^{d_{model}×d_k}
- W_i^V ∈ ℝ^{d_{model}×d_v}
- W^O ∈ ℝ^{hd_v×d_{model}}

#### 왜 Multi-Head인가?

**여러 관점에서 정보 추출:**
- Head 1: 문법 정보
- Head 2: 의미 정보
- Head 3: 위치 정보
- ...

**Ensemble 효과:**
- 여러 attention pattern 학습
- 표현력 ↑

#### 구현

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Multi-Head Attention

        Args:
            d_model: 모델 차원 (512)
            num_heads: Head 개수 (8)
        """
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear layers for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output linear layer
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        """
        (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Args:
            Q, K, V: (batch, num_heads, seq_len, d_k)
            mask: (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)

        Returns:
            output: (batch, num_heads, seq_len, d_k)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        d_k = Q.size(-1)

        # Attention scores: (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask (optional)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Attention weights
        attention_weights = torch.softmax(scores, dim=-1)

        # Weighted sum
        output = torch.matmul(attention_weights, V)

        return output, attention_weights

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q, K, V: (batch, seq_len, d_model)
            mask: attention mask

        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size = Q.size(0)

        # Linear projections
        Q = self.W_q(Q)  # (batch, seq_len, d_model)
        K = self.W_k(K)
        V = self.W_v(V)

        # Split into multiple heads
        Q = self.split_heads(Q)  # (batch, num_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Scaled dot-product attention
        attn_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        # (batch, num_heads, seq_len, d_k)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        # (batch, seq_len, num_heads, d_k)

        attn_output = attn_output.view(batch_size, -1, self.d_model)
        # (batch, seq_len, d_model)

        # Final linear layer
        output = self.W_o(attn_output)

        return output
```

### 2.4 Self-Attention

**Self-Attention**은 Q = K = V인 경우:

$$
\text{SelfAttention}(X) = \text{Attention}(XW^Q, XW^K, XW^V)
$$

**의미:**
- 입력 시퀀스 내부의 관계 모델링
- "문장 내 단어들 간의 관계"

**예시:**

```
문장: "The animal didn't cross the street because it was too tired."

Self-Attention이 학습한 것:
- "it" → "animal" (높은 attention weight)
- "cross" → "street" (높은 attention weight)
```

### 2.5 Causal (Masked) Self-Attention

**생성 모델**에서는 **미래를 볼 수 없음**!

#### Causal Mask

$$
\text{mask}[i, j] = \begin{cases}
1 & \text{if } j \leq i \\
0 & \text{if } j > i
\end{cases}
$$

**구현:**

```python
def create_causal_mask(seq_len):
    """
    Create causal mask for autoregressive generation

    Returns:
        mask: (seq_len, seq_len)
              [[1, 0, 0],
               [1, 1, 0],
               [1, 1, 1]]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask

# Attention에 적용
scores = scores.masked_fill(mask == 0, float('-inf'))
```

**효과:**

```
Position:  1    2    3
Token:    "I"  "am" "happy"

Position 2 ("am") 생성 시:
- "I" 볼 수 있음 ✅
- "am" 볼 수 없음 ❌ (자기 자신)
- "happy" 볼 수 없음 ❌ (미래)
```

---

## 3. Transformer 아키텍처

### 3.1 전체 구조

**Transformer** = Encoder + Decoder

```
Input Sequence
     ↓
   Encoder (6 layers)
     ↓
   Encoding
     ↓
   Decoder (6 layers) ← Output Sequence (shifted)
     ↓
   Output Probabilities
```

### 3.2 Encoder

#### 구조

하나의 Encoder Layer:

```
Input (x)
    ↓
Multi-Head Self-Attention
    ↓
Add & Norm (Residual Connection + LayerNorm)
    ↓
Feed-Forward Network
    ↓
Add & Norm
    ↓
Output
```

#### 수식

**Self-Attention:**
$$
\text{Attn}(X) = \text{MultiHead}(X, X, X)
$$

**Add & Norm:**
$$
\text{Output}_1 = \text{LayerNorm}(X + \text{Attn}(X))
$$

**Feed-Forward:**
$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

**Add & Norm:**
$$
\text{Output}_2 = \text{LayerNorm}(\text{Output}_1 + \text{FFN}(\text{Output}_1))
$$

#### PyTorch 구현

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Transformer Encoder Layer

        Args:
            d_model: 모델 차원 (512)
            num_heads: Attention head 수 (8)
            d_ff: Feed-forward 차원 (2048)
            dropout: Dropout 확률
        """
        super().__init__()

        # Multi-Head Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: attention mask

        Returns:
            output: (batch, seq_len, d_model)
        """
        # Self-Attention + Residual + Norm
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-Forward + Residual + Norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x
```

### 3.3 Decoder

#### 구조

하나의 Decoder Layer:

```
Output (shifted right)
    ↓
Masked Multi-Head Self-Attention (causal)
    ↓
Add & Norm
    ↓
Multi-Head Cross-Attention (attend to encoder)
    ↓
Add & Norm
    ↓
Feed-Forward Network
    ↓
Add & Norm
    ↓
Output
```

**차이점:**
1. **Masked Self-Attention**: Causal mask 사용
2. **Cross-Attention**: Q는 decoder, K/V는 encoder

#### Cross-Attention

$$
\text{CrossAttn}(X_{dec}, X_{enc}) = \text{MultiHead}(X_{dec}W^Q, X_{enc}W^K, X_{enc}W^V)
$$

**의미:**
- Decoder가 Encoder의 정보를 참조
- "번역 시 원문 참고"

### 3.4 Positional Encoding

**문제:** Attention은 순서 정보가 없음!

**해결:** 위치 정보를 encoding에 추가

#### Sinusoidal Positional Encoding

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

**특징:**
- 각 position마다 고유한 encoding
- 상대적 위치도 표현 가능

#### 구현

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            x + pe: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]
```

---

## 4. GPT: Decoder-only Transformer

### 4.1 GPT 아키텍처

**GPT (Generative Pre-trained Transformer)**는 **Decoder-only** 구조:

```
Input Tokens
    ↓
Token Embedding + Positional Encoding
    ↓
Transformer Decoder Layers (12-96층)
    ↓
Language Model Head (Linear + Softmax)
    ↓
Next Token Probabilities
```

**핵심 차이:**
- Encoder 없음
- Cross-Attention 없음
- Causal Self-Attention만 사용

### 4.2 GPT 학습

#### Pre-training: Next Token Prediction

**목적:** 다음 토큰 예측

$$
\mathcal{L}_{LM} = -\sum_{t=1}^{T} \log p_\theta(x_t \mid x_1, \ldots, x_{t-1})
$$

**데이터:** 대규모 텍스트 (인터넷 전체!)

#### Fine-tuning

Pre-trained 모델을 downstream task에 맞게 조정:

```python
# Classification
output = gpt(input_ids)
logits = classification_head(output[:, -1, :])  # 마지막 토큰 사용

# Generation
for _ in range(max_new_tokens):
    logits = gpt(input_ids)
    next_token = torch.argmax(logits[:, -1, :], dim=-1)
    input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
```

### 4.3 GPT-2

**규모:**
- 117M ~ 1.5B 파라미터
- 40GB 텍스트 데이터

**개선점:**
- Layer Normalization 위치 변경 (Pre-norm)
- 더 큰 context (1024 tokens)

#### GPT-2 블록

```python
class GPT2Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # ReLU 대신 GELU
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        # Pre-norm: LayerNorm before attention
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
```

### 4.4 GPT-3

**혁명적 변화:**
- 175B 파라미터
- Few-shot Learning

#### In-Context Learning

**Zero-shot:**
```
Translate to French: Hello
```

**Few-shot:**
```
Translate to French:
Hello → Bonjour
Good morning → Bon matin
Thank you →
```

**메커니즘:** 모델이 예시를 보고 패턴 학습 (gradient update 없이!)

---

## 5. Autoregressive Image Generation

### 5.1 PixelCNN

#### 아이디어

이미지를 1D 시퀀스로 취급:

$$
p(\mathbf{x}) = \prod_{i=1}^{n} p(x_i \mid x_1, \ldots, x_{i-1})
$$

**순서:** 왼쪽 위 → 오른쪽 아래 (raster scan)

#### Masked Convolution

```
Mask:
[1, 1, 1]
[1, 1, 0]  ← 현재 픽셀만 보지 않음
[0, 0, 0]

현재 픽셀 예측 시:
- 위쪽 픽셀들 ✅
- 왼쪽 픽셀들 ✅
- 현재/오른쪽/아래 픽셀들 ❌
```

### 5.2 VQ-VAE + Transformer

#### 2-Stage 생성

**Stage 1: VQ-VAE**
```
Image (256×256) → Encoder → Latent Codes (32×32)
Latent Codes → Decoder → Reconstructed Image
```

**Stage 2: Transformer**
```
Latent Codes (flatten) → Transformer → Next Latent Code
```

**장점:**
- 이미지 직접 생성보다 빠름
- 고해상도 가능

#### Latent Codes

**Vector Quantization:**
$$
z_q = \arg\min_{e_k \in \mathcal{C}} \|z_e - e_k\|
$$

Continuous latent z_e를 discrete code z_q로 양자화.

### 5.3 ViT-VQGAN (Taming Transformers)

#### 개선점

1. **Patch-based Representation**
   - 16×16 패치로 분할
   - 각 패치를 하나의 토큰으로

2. **Bidirectional Transformer (학습)**
   - BERT-like masking

3. **Autoregressive Transformer (생성)**
   - GPT-like generation

#### 구조

```
Image (256×256×3)
    ↓ Encoder
Latent (16×16×256)
    ↓ Quantize
Codes (16×16) → Flatten → (256,)
    ↓
Transformer Decoder
    ↓
Predicted Next Code
```

---

## 6. BERT와 Masked Language Modeling

### 6.1 BERT vs GPT

| | GPT | BERT |
|---|---|---|
| **구조** | Decoder-only | Encoder-only |
| **Attention** | Causal (unidirectional) | Bidirectional |
| **학습** | Next token prediction | Masked token prediction |
| **용도** | Generation | Understanding |

### 6.2 Masked Language Modeling (MLM)

#### 학습 방법

**Masking:**
```
Original: "The cat sat on the mat"
Masked:   "The [MASK] sat on the [MASK]"
Target:   "cat", "mat"
```

**목적 함수:**
$$
\mathcal{L}_{MLM} = -\sum_{i \in \mathcal{M}} \log p_\theta(x_i \mid \mathbf{x}_{\backslash \mathcal{M}})
$$

여기서 M은 mask된 position 집합.

### 6.3 BERT for Generation?

**문제:** BERT는 생성 모델이 아님!

하지만 **iterative refinement**로 사용 가능:

```python
# Iterative decoding
output = [MASK] * seq_len

for iteration in range(num_iterations):
    logits = bert(output)
    probs = softmax(logits)

    # 가장 confident한 토큰만 채우기
    confident_positions = probs.max(dim=-1) > threshold
    output[confident_positions] = argmax(probs[confident_positions])
```

**최근 연구:** Masked Generative Models (MaskGIT)

---

## 7. Vision Transformer for Generation

### 7.1 Image GPT (iGPT)

#### 아이디어

이미지를 **시퀀스**로 취급:

```
Image (32×32×3)
    ↓ Flatten
Sequence (3072,)
    ↓
GPT
    ↓
Next Pixel
```

#### 결과

- 놀랍게도 작동함!
- 하지만 너무 느림 (32×32도 3072 tokens)

### 7.2 Parti (Pathways Autoregressive Text-to-Image)

#### 구조

```
Text Prompt
    ↓
T5 Encoder
    ↓
Text Embeddings
    ↓
ViT-VQGAN Transformer (Cross-Attention)
    ↓
Image Tokens
    ↓
ViT-VQGAN Decoder
    ↓
Generated Image
```

**특징:**
- 20B 파라미터
- Text-to-image generation
- Autoregressive

---

## 8. Scaling Laws와 Emergent Abilities

### 8.1 Scaling Laws

#### Kaplan et al. (2020) 발견

모델 성능은 다음에 따라 **power law**로 향상:

$$
L(N) \propto N^{-\alpha}
$$

- N: 파라미터 수
- L: Loss
- α ≈ 0.076

**의미:**
- 크기가 중요!
- 데이터도 같이 늘려야

#### Chinchilla Scaling Laws

**Compute-optimal:**
- 모델 크기와 데이터 크기를 균형있게

**결과:**
- 70B 모델 + 1.4T tokens > 280B 모델 + 300B tokens

### 8.2 Emergent Abilities

**Emergent Ability**: 큰 모델에서만 나타나는 능력

**예시:**
- Few-shot learning
- Chain-of-thought reasoning
- Instruction following

**언제 나타나나?**
- 보통 10B+ 파라미터부터

---

## 9. 구현 및 실습

### 9.1 Mini GPT 구현

```python
import torch
import torch.nn as nn

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048, max_len=1024):
        super().__init__()

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            GPT2Block(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        # Output head
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        """
        Args:
            input_ids: (batch, seq_len)

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        # Embedding + Positional Encoding
        x = self.token_embedding(input_ids)  # (batch, seq_len, d_model)
        x = self.pos_encoding(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # Language model head
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)

        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0):
        """
        Autoregressive generation

        Args:
            input_ids: (batch, seq_len) 시작 토큰들
            max_new_tokens: 생성할 토큰 수
            temperature: 샘플링 temperature

        Returns:
            generated: (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Forward pass
            logits = self(input_ids)  # (batch, seq_len, vocab_size)

            # 마지막 토큰의 logits
            logits = logits[:, -1, :] / temperature  # (batch, vocab_size)

            # Multinomial sampling
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
```

### 9.2 학습 코드

```python
def train_gpt(model, dataloader, num_epochs=10, lr=3e-4):
    """
    GPT 학습
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0

        for batch in dataloader:
            input_ids = batch['input_ids']  # (batch, seq_len)
            labels = batch['labels']  # (batch, seq_len)

            # Forward
            logits = model(input_ids)  # (batch, seq_len, vocab_size)

            # Compute loss
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        perplexity = torch.exp(torch.tensor(avg_loss))

        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - PPL: {perplexity:.2f}')
```

---

## 📚 참고 자료

### 필수 논문

1. **Attention Is All You Need** (Vaswani et al., 2017)
   - [arXiv](https://arxiv.org/abs/1706.03762)
   - Transformer 원조 논문

2. **GPT-1** (Radford et al., 2018)
   - [PDF](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

3. **BERT** (Devlin et al., 2018)
   - [arXiv](https://arxiv.org/abs/1810.04805)

4. **GPT-2** (Radford et al., 2019)
   - [PDF](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

5. **GPT-3** (Brown et al., 2020)
   - [arXiv](https://arxiv.org/abs/2005.14165)
   - Few-shot learning

6. **Scaling Laws** (Kaplan et al., 2020)
   - [arXiv](https://arxiv.org/abs/2001.08361)

7. **Taming Transformers** (Esser et al., 2021)
   - [arXiv](https://arxiv.org/abs/2012.09841)
   - ViT-VQGAN

---

**최종 업데이트**: 2025-11-15
**라이선스**: MIT
