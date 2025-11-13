# Phase 2: 딥러닝 기초 - 상세 가이드

## Week 5-6: 신경망 기초

### 🎯 학습 목표

이 섹션에서는 딥러닝의 핵심인 **역전파 (Backpropagation)**를 완전히 이해합니다.

**왜 중요한가?**
- Generative Model도 결국 신경망
- 역전파를 이해해야 학습 과정을 이해할 수 있음
- 디버깅과 최적화에 필수

---

### Day 1-2: 신경망의 작동 원리

#### 💡 퍼셉트론에서 딥러닝까지

**단일 퍼셉트론 (Single Perceptron)**

가장 간단한 신경망 유닛:

```
y = σ(w·x + b)

where:
- x: 입력 벡터
- w: 가중치 벡터
- b: 편향(bias)
- σ: 활성화 함수
```

**직관적 이해**:
1. 입력에 가중치를 곱해서 합산 (선형 변환)
2. 편향을 더함 (이동)
3. 활성화 함수 적용 (비선형성 추가)

#### 📐 왜 비선형 활성화 함수가 필요한가?

**없으면 어떻게 되나?**

```python
# 활성화 함수 없이 2층 네트워크
y = W2(W1·x + b1) + b2
  = W2W1·x + W2b1 + b2
  = W'·x + b'  # 결국 선형 변환!
```

**결론**: 활성화 함수 없으면 여러 층을 쌓아도 의미 없음!

**주요 활성화 함수**:

1. **Sigmoid**: σ(x) = 1/(1+e^(-x))
   ```python
   def sigmoid(x):
       return 1 / (1 + np.exp(-x))
   ```
   - 출력: (0, 1)
   - 문제: Vanishing Gradient
   - 사용처: 이진 분류 출력층

2. **Tanh**: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
   ```python
   def tanh(x):
       return np.tanh(x)
   ```
   - 출력: (-1, 1)
   - Sigmoid보다 나음 (zero-centered)
   - 여전히 Vanishing Gradient 문제

3. **ReLU**: f(x) = max(0, x)
   ```python
   def relu(x):
       return np.maximum(0, x)
   ```
   - 현대 딥러닝의 표준
   - 계산 빠름
   - Vanishing Gradient 완화
   - 문제: Dying ReLU (음수 영역에서 그래디언트 0)

4. **LeakyReLU**: f(x) = max(αx, x) where α=0.01
   ```python
   def leaky_relu(x, alpha=0.01):
       return np.where(x > 0, x, alpha * x)
   ```
   - Dying ReLU 문제 해결
   - 음수 영역에서도 작은 그래디언트

#### 💻 실습: 활성화 함수 비교

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """Sigmoid 활성화 함수"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # 오버플로우 방지

def tanh(x):
    """Tanh 활성화 함수"""
    return np.tanh(x)

def relu(x):
    """ReLU 활성화 함수"""
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU 활성화 함수"""
    return np.where(x > 0, x, alpha * x)

def gelu(x):
    """GELU (Gaussian Error Linear Unit)

    Transformer 등 최신 모델에서 사용
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

# 입력 범위
x = np.linspace(-5, 5, 1000)

# 각 활성화 함수 계산
activations = {
    'Sigmoid': sigmoid(x),
    'Tanh': tanh(x),
    'ReLU': relu(x),
    'Leaky ReLU': leaky_relu(x),
    'GELU': gelu(x)
}

# 그래디언트 계산 (수치 미분)
def numerical_gradient(func, x, h=1e-5):
    return (func(x + h) - func(x - h)) / (2 * h)

gradients = {
    'Sigmoid': numerical_gradient(sigmoid, x),
    'Tanh': numerical_gradient(tanh, x),
    'ReLU': numerical_gradient(relu, x),
    'Leaky ReLU': numerical_gradient(lambda x: leaky_relu(x, 0.01), x),
    'GELU': numerical_gradient(gelu, x)
}

# 시각화
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

colors = ['blue', 'green', 'red', 'purple', 'orange']

for idx, (name, color) in enumerate(zip(activations.keys(), colors)):
    # 활성화 함수
    ax = axes[idx]
    ax.plot(x, activations[name], linewidth=3, color=color, label=name)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Input (x)', fontsize=12)
    ax.set_ylabel('Output f(x)', fontsize=12)
    ax.set_title(f'{name}\nActivation Function', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

    # 그래디언트 (같은 플롯에 점선으로)
    ax2 = ax.twinx()
    ax2.plot(x, gradients[name], linewidth=2, linestyle='--',
             color=color, alpha=0.5, label=f"{name} gradient")
    ax2.set_ylabel("Gradient f'(x)", fontsize=10, alpha=0.7)
    ax2.tick_params(axis='y', labelsize=9, alpha=0.7)

# 마지막 subplot 제거
fig.delaxes(axes[-1])

plt.tight_layout()
plt.savefig('activation_functions.png', dpi=150, bbox_inches='tight')
plt.show()

print("🔍 활성화 함수 분석")
print("=" * 70)
print("\n1. Sigmoid")
print("   장점: 출력이 (0,1), 확률로 해석 가능")
print("   단점: Vanishing Gradient (x가 크거나 작을 때 gradient → 0)")
print("   사용: 이진 분류 출력층")

print("\n2. Tanh")
print("   장점: Zero-centered, Sigmoid보다 gradient 크다")
print("   단점: 여전히 Vanishing Gradient")
print("   사용: RNN (과거에), 일부 GAN")

print("\n3. ReLU")
print("   장점: 계산 빠름, Vanishing Gradient 완화, 실전에서 잘 작동")
print("   단점: Dying ReLU (음수 입력 시 gradient=0)")
print("   사용: CNN, MLP의 표준")

print("\n4. Leaky ReLU")
print("   장점: Dying ReLU 해결, 음수 영역도 학습")
print("   단점: α 하이퍼파라미터 선택")
print("   사용: ReLU 대체")

print("\n5. GELU")
print("   장점: 부드러운 곡선, Transformer에서 좋은 성능")
print("   단점: 계산 비용 약간 높음")
print("   사용: BERT, GPT 등 최신 모델")

# Vanishing Gradient 문제 시연
print("\n\n⚠️ Vanishing Gradient 문제")
print("=" * 70)

x_large = np.array([-10, -5, 0, 5, 10])
for x_val in x_large:
    sig_grad = numerical_gradient(sigmoid, np.array([x_val]))[0]
    relu_grad = numerical_gradient(relu, np.array([x_val]))[0]

    print(f"x = {x_val:3.0f}: Sigmoid' = {sig_grad:.6f}, ReLU' = {relu_grad:.6f}")

print("\n✅ 관찰:")
print("   • Sigmoid: |x|가 커지면 gradient → 0 (Vanishing!)")
print("   • ReLU: x > 0이면 항상 gradient = 1 (안정적!)")
```

**코드 상세 설명**:

1. **활성화 함수 그래프**:
   - y축: 함수 값
   - 기울기가 그래디언트의 크기를 나타냄

2. **Vanishing Gradient 시연**:
   ```python
   x = 10일 때:
   - sigmoid'(10) ≈ 0.00005 (거의 0!)
   - relu'(10) = 1.0 (안정적)
   ```

3. **실전 선택 가이드**:
   ```python
   # 숨겨진 층
   hidden_layer = nn.ReLU()  # 기본 선택

   # 출력층
   # 이진 분류
   output_binary = nn.Sigmoid()
   # 다중 분류
   output_multi = nn.Softmax(dim=1)
   # 회귀
   output_regression = None  # 활성화 함수 없음
   ```

---

### Day 3-5: 역전파 알고리즘 완전 정복

#### 💡 역전파란 무엇인가?

**역전파 (Backpropagation)**: 체인 룰을 사용하여 손실 함수의 그래디언트를 효율적으로 계산하는 알고리즘

**왜 필요한가?**
- 신경망에는 수백만~수십억 개의 파라미터
- 각 파라미터에 대한 그래디언트를 계산해야 함
- 순진한 방법: O(n²) → 역전파: O(n)

#### 📐 체인 룰 (Chain Rule)

**기본 형태**:
```
z = f(g(x))
dz/dx = (dz/dg) × (dg/dx)
```

**다변수 함수**:
```
z = f(x, y)
x = g(t)
y = h(t)

dz/dt = (∂z/∂x)(dx/dt) + (∂z/∂y)(dy/dt)
```

**신경망에서**:
```
L = loss(y_pred, y_true)
y_pred = f(W3, f(W2, f(W1, x)))

∂L/∂W1 = (∂L/∂y_pred) × (∂y_pred/∂h2) × (∂h2/∂h1) × (∂h1/∂W1)
```

#### 💻 실습: 처음부터 역전파 구현

**목표**: Autograd 없이 역전파를 직접 구현하여 원리 이해

```python
import numpy as np

class Layer:
    """신경망 레이어 기본 클래스"""

    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        """순전파: 입력 → 출력"""
        raise NotImplementedError

    def backward(self, output_gradient, learning_rate):
        """역전파: 출력의 그래디언트 → 입력의 그래디언트"""
        raise NotImplementedError


class Linear(Layer):
    """
    완전 연결 층 (Fully Connected Layer)

    순전파: y = Wx + b
    역전파:
        - ∂L/∂W = ∂L/∂y × x^T
        - ∂L/∂b = ∂L/∂y
        - ∂L/∂x = W^T × ∂L/∂y
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        # He 초기화 (ReLU에 적합)
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((output_size, 1))

    def forward(self, input):
        """
        순전파

        Args:
            input: (input_size, batch_size)
        Returns:
            output: (output_size, batch_size)
        """
        self.input = input
        self.output = np.dot(self.weights, input) + self.bias
        return self.output

    def backward(self, output_gradient, learning_rate):
        """
        역전파

        Args:
            output_gradient: ∂L/∂output (output_size, batch_size)
            learning_rate: 학습률
        Returns:
            input_gradient: ∂L/∂input (input_size, batch_size)
        """
        # ∂L/∂W = ∂L/∂y × x^T
        weights_gradient = np.dot(output_gradient, self.input.T)

        # ∂L/∂b = Σ ∂L/∂y (배치 차원으로 합산)
        bias_gradient = np.sum(output_gradient, axis=1, keepdims=True)

        # ∂L/∂x = W^T × ∂L/∂y
        input_gradient = np.dot(self.weights.T, output_gradient)

        # 파라미터 업데이트 (Gradient Descent)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient

        return input_gradient


class ReLU(Layer):
    """
    ReLU 활성화 함수

    순전파: y = max(0, x)
    역전파: ∂L/∂x = ∂L/∂y × 1(x > 0)
    """

    def forward(self, input):
        self.input = input
        self.output = np.maximum(0, input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        """
        역전파

        ReLU의 미분:
        - x > 0이면 1
        - x ≤ 0이면 0
        """
        # Element-wise 곱셈
        return output_gradient * (self.input > 0)


class Sigmoid(Layer):
    """
    Sigmoid 활성화 함수

    순전파: y = 1/(1+e^(-x))
    역전파: ∂L/∂x = ∂L/∂y × y(1-y)
    """

    def forward(self, input):
        self.input = input
        self.output = 1 / (1 + np.exp(-np.clip(input, -500, 500)))
        return self.output

    def backward(self, output_gradient, learning_rate):
        """
        역전파

        sigmoid'(x) = sigmoid(x) × (1 - sigmoid(x))
        """
        sigmoid_derivative = self.output * (1 - self.output)
        return output_gradient * sigmoid_derivative


class MSELoss:
    """
    평균 제곱 오차 (Mean Squared Error)

    L = (1/n) Σ (y_pred - y_true)²
    ∂L/∂y_pred = (2/n) (y_pred - y_true)
    """

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2)

    def backward(self):
        """손실의 그래디언트"""
        n = self.y_pred.shape[1]  # 배치 크기
        return (2 / n) * (self.y_pred - self.y_true)


# 간단한 신경망 구축 및 학습
print("🧠 신경망 학습 예제: XOR 문제")
print("=" * 70)

# XOR 데이터셋
# XOR는 선형 분리 불가능 → 은닉층 필요!
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T  # (2, 4)
y_train = np.array([[0], [1], [1], [0]]).T  # (1, 4)

print("XOR 진리표:")
print("X1 | X2 | Y")
print("---|----|-")
for i in range(4):
    print(f" {X_train[0,i]:.0f} |  {X_train[1,i]:.0f} | {y_train[0,i]:.0f}")

# 네트워크 구조: 2 → 4 → 1
network = [
    Linear(2, 4),
    ReLU(),
    Linear(4, 1),
    Sigmoid()
]

criterion = MSELoss()
learning_rate = 0.1
epochs = 5000

# 학습
losses = []
for epoch in range(epochs):
    # 순전파
    output = X_train
    for layer in network:
        output = layer.forward(output)

    # 손실 계산
    loss = criterion.forward(output, y_train)
    losses.append(loss)

    # 역전파
    gradient = criterion.backward()
    for layer in reversed(network):
        gradient = layer.backward(gradient, learning_rate)

    # 진행 상황 출력
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch+1:5d}: Loss = {loss:.6f}")

# 테스트
print("\n📊 학습 결과:")
print("=" * 70)
output = X_train
for layer in network:
    output = layer.forward(output)

print("\nX1 | X2 | Target | Predicted | Rounded")
print("---|----| -------|-----------|--------")
for i in range(4):
    pred = output[0, i]
    print(f" {X_train[0,i]:.0f} |  {X_train[1,i]:.0f} |   {y_train[0,i]:.0f}    |   {pred:.4f}  |    {round(pred):.0f}")

# 손실 그래프
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(losses, linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.title('Training Loss over Time', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.yscale('log')  # 로그 스케일로 보기
plt.tight_layout()
plt.show()

print("\n✅ 학습 성공!")
print("XOR 문제는 선형 분리 불가능하지만, 은닉층이 있으면 해결 가능!")
```

**코드 단계별 설명**:

1. **Layer 클래스**:
   ```python
   # 모든 층이 구현해야 할 인터페이스
   - forward(input): 순전파
   - backward(output_gradient, lr): 역전파
   ```

2. **Linear 층의 역전파**:
   ```python
   # y = Wx + b에서

   # W에 대한 그래디언트
   ∂L/∂W = ∂L/∂y × ∂y/∂W
          = output_gradient × input^T

   # b에 대한 그래디언트
   ∂L/∂b = ∂L/∂y × ∂y/∂b
          = output_gradient × 1

   # 입력에 대한 그래디언트 (다음 층으로 전달)
   ∂L/∂x = ∂L/∂y × ∂y/∂x
          = output_gradient × W^T
   ```

3. **ReLU의 역전파**:
   ```python
   # ReLU(x) = max(0, x)
   # 미분: x > 0이면 1, 아니면 0

   ∂L/∂x = ∂L/∂y × (x > 0)
   ```

4. **학습 루프**:
   ```python
   for epoch in range(epochs):
       # 1. 순전파: 입력 → 출력
       output = forward(input)

       # 2. 손실 계산
       loss = criterion(output, target)

       # 3. 역전파: 출력 ← 입력
       gradient = backward()

       # 4. 파라미터 업데이트
       update_parameters(gradient)
   ```

**💡 XOR 문제를 통한 인사이트**:

```
XOR 진리표:
0 XOR 0 = 0
0 XOR 1 = 1
1 XOR 0 = 1
1 XOR 1 = 0
```

**선형 분리 불가능**:
- 단일 직선으로는 데이터를 분리할 수 없음
- 하지만 은닉층을 추가하면 해결!

**은닉층이 하는 일**:
- 입력 공간을 새로운 표현으로 변환
- 변환된 공간에서는 선형 분리 가능

```python
# 은닉층의 출력을 시각화하면
# 원래는 선형 분리 불가능했던 데이터가
# 변환된 공간에서는 분리 가능해짐!
```

---

### ⚠️ 역전파의 일반적인 실수와 해결

#### 1. 그래디언트 폭발/소실

**문제**:
```python
# 100층 네트워크
for i in range(100):
    # Sigmoid 사용
    x = sigmoid(Wx + b)
    # 각 층에서 gradient가 0.25씩 곱해짐
    # 100층: 0.25^100 ≈ 0 (소실!)
```

**해결**:
- ReLU 사용
- Residual Connection (ResNet)
- Batch Normalization
- Gradient Clipping

#### 2. 잘못된 차원

**문제**:
```python
# 배치 처리 시 차원 주의
input: (batch_size, features)
weights: (features, hidden)
# 올바른 곱셈: input @ weights
```

**해결**:
- Shape 항상 확인
- Assert문 사용
```python
assert output.shape == expected_shape
```

#### 3. In-place 연산

**문제**:
```python
# PyTorch에서
x += 1  # 잘못됨! Autograd 그래프 망가짐
x = x + 1  # 올바름
```

**해결**:
- In-place 연산 (`+=`, `*=`) 피하기
- 새 텐서 생성

---

이어서 Phase 3 (VAE, GAN, Diffusion)도 이런 식으로 상세하게 작성할까요?
