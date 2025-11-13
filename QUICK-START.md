# 🚀 Quick Start Guide

Generative AI 학습을 시작하는 가장 빠른 방법!

---

## 1️⃣ 저장소 클론

```bash
git clone https://github.com/your-username/Generative-AI-Study.git
cd Generative-AI-Study
```

---

## 2️⃣ Interactive 웹페이지 열기

### 방법 1: 직접 열기
`index.html` 파일을 더블클릭하여 브라우저에서 엽니다.

### 방법 2: 로컬 서버 사용
```bash
# Python 3가 설치되어 있다면
python -m http.server 8000

# 브라우저에서 http://localhost:8000 접속
```

### 방법 3: VS Code Live Server
1. VS Code에서 프로젝트 열기
2. Live Server 확장 설치
3. `index.html` 우클릭 → "Open with Live Server"

---

## 3️⃣ 개발 환경 설정

### Python 환경
```bash
# 가상환경 생성
conda create -n genai python=3.10
conda activate genai

# PyTorch 설치 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 기본 라이브러리
pip install numpy pandas matplotlib jupyter
pip install transformers diffusers datasets accelerate
```

### 설치 확인
```bash
python code/examples/hello_pytorch.py
```

---

## 4️⃣ 학습 시작

### Phase 1부터 시작 (권장)
1. 웹페이지에서 "Phase 1" 클릭
2. Week 1-2 체크리스트 펼치기
3. 각 항목을 학습하며 체크 표시
4. 메모 섹션에 학습 내용 기록

### 빠른 실습부터 시작 (Top-Down)
1. [프로젝트 가이드](./resources/PROJECTS.md) 참고
2. 초급 프로젝트 선택 (예: MNIST VAE)
3. 구현하면서 필요한 이론 학습
4. 웹페이지에서 진행 상황 추적

---

## 5️⃣ 학습 루틴

### 일일 루틴
```
1. 웹페이지에서 오늘의 목표 확인
2. 타이머 시작
3. 이론 학습 (30-60분)
4. 코드 실습 (60-90분)
5. 진행 상황 체크 및 메모 작성
```

### 주간 루틴
```
월요일: 이번 주 계획 수립
화-목: 집중 학습
금요일: 주간 리뷰 및 블로그 작성
주말: 프로젝트 진행
```

---

## 6️⃣ 주요 기능

### 웹페이지 기능
- ✅ **진행률 추적**: 실시간으로 학습 진행 상황 확인
- ⏱️ **타이머**: 학습 시간 측정 및 통계
- 📝 **체크리스트**: 각 Week별 학습 항목 관리
- 💾 **자동 저장**: 브라우저 로컬 스토리지에 자동 저장
- 📊 **대시보드**: 전체 통계 한눈에 보기
- 🏆 **업적**: 마일스톤 달성 추적
- 🌙 **다크 모드**: 눈 편한 야간 모드

### 진행 상황 내보내기
웹페이지 하단의 "진행상황 다운로드" 버튼을 클릭하여 JSON 파일로 저장

---

## 7️⃣ 추천 학습 자료

### 필수 논문 (최우선)
1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer
2. [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - VAE
3. [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) - GAN
4. [DDPM](https://arxiv.org/abs/2006.11239) - Diffusion Models
5. [Stable Diffusion](https://arxiv.org/abs/2112.10752) - Latent Diffusion

### 온라인 강의
- [Stanford CS236: Deep Generative Models](https://deepgenerativemodels.github.io/)
- [Fast.ai - Practical Deep Learning](https://course.fast.ai/)
- [Hugging Face Course](https://huggingface.co/learn/nlp-course)

### 블로그
- [Lil'Log by Lilian Weng](https://lilianweng.github.io/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

---

## 8️⃣ 커뮤니티

### 질문 & 토론
- Reddit: [r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
- Discord: Hugging Face, Stable Diffusion
- Stack Overflow: [pytorch] [machine-learning] 태그

### 프로젝트 공유
- [Hugging Face Spaces](https://huggingface.co/spaces)
- [Papers with Code](https://paperswithcode.com/)
- GitHub

---

## 9️⃣ 문제 해결

### CUDA Out of Memory
```python
# 배치 크기 줄이기
batch_size = 8  # 기존 16 → 8

# Gradient Accumulation
accumulation_steps = 4

# Mixed Precision Training
from torch.cuda.amp import autocast
with autocast():
    output = model(input)
```

### Import Error
```bash
# 환경 확인
conda activate genai

# 패키지 재설치
pip install package-name --upgrade
```

### 더 많은 문제 해결
[환경 설정 가이드](./resources/SETUP-GUIDE.md#문제-해결) 참고

---

## 🔟 다음 단계

### 첫 1주일
- [ ] 개발 환경 설정 완료
- [ ] `hello_pytorch.py` 실행 성공
- [ ] Phase 1 Week 1 시작
- [ ] 첫 논문 읽기 (Attention Is All You Need)

### 첫 1개월
- [ ] Phase 1 완료
- [ ] Phase 2 진행 중
- [ ] 논문 5편 읽기
- [ ] 블로그 포스트 1개 작성

### 첫 3개월
- [ ] Phase 1-3 완료
- [ ] VAE, GAN 구현 경험
- [ ] 논문 15편 읽기
- [ ] 미니 프로젝트 1개 완성

---

## 💡 학습 팁

1. **꾸준함이 핵심**: 매일 조금씩이라도 학습
2. **손으로 직접**: 코드는 복사-붙여넣기 말고 직접 타이핑
3. **기록하기**: 학습 내용을 블로그나 노트에 정리
4. **커뮤니티 활용**: 혼자 고민하지 말고 질문하기
5. **프로젝트 중심**: 이론만 공부하지 말고 프로젝트 진행

---

## 📞 도움이 필요하신가요?

- **문서**: 이 저장소의 `docs/` 폴더 참고
- **이슈**: GitHub Issues에 질문 남기기
- **논의**: GitHub Discussions 활용

---

**준비 되셨나요? 지금 바로 시작하세요! 🚀**

[📊 Interactive 학습 트래커 열기](./index.html) | [📚 상세 커리큘럼 보기](./README.md) | [📖 학습 가이드](./docs/STUDY-GUIDE.md)
