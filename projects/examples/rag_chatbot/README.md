# RAG Chatbot 프로젝트

> Retrieval-Augmented Generation을 활용한 도메인 특화 챗봇
>
> **목표**: LLM + 벡터 DB로 정확하고 신뢰할 수 있는 답변 제공

---

## 📖 프로젝트 개요

이 프로젝트는 RAG (Retrieval-Augmented Generation) 기법을 사용하여 특정 도메인 지식에 대한 질문에 답하는 챗봇 시스템입니다.

**핵심 기술:**
- 🔍 **Retrieval**: 벡터 DB에서 관련 문서 검색
- 🤖 **Augmentation**: 검색된 문서를 컨텍스트로 추가
- 💬 **Generation**: LLM으로 답변 생성

**장점:**
- ✅ Hallucination 감소 (실제 문서 기반)
- ✅ 최신 정보 반영 (DB 업데이트만 하면 됨)
- ✅ 출처 제시 가능
- ✅ 도메인 특화 가능

---

## 🏗️ 시스템 아키텍처

```
User Query
    ↓
Embedding Model → Vector Representation
    ↓
Vector DB Search → Top-K Relevant Documents
    ↓
Prompt Construction (Query + Documents)
    ↓
LLM (GPT-4, LLaMA, etc.) → Answer
    ↓
Post-processing + Source Links
    ↓
Response to User
```

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
pip install -r requirements.txt

# 필수 패키지
# - langchain
# - openai (또는 transformers for local LLM)
# - chromadb (또는 pinecone, weaviate)
# - sentence-transformers
```

### 2. 데이터 준비

```bash
# 문서 디렉토리 구조
knowledge_base/
├── documents/
│   ├── doc1.txt
│   ├── doc2.pdf
│   └── doc3.md
└── processed/
    └── chunks.json

# 문서 처리 및 임베딩
python scripts/build_knowledge_base.py \
    --input_dir knowledge_base/documents \
    --output_db chroma_db \
    --chunk_size 500 \
    --chunk_overlap 50
```

### 3. 챗봇 실행

```bash
# CLI 버전
python chatbot.py --knowledge_base chroma_db

# Web UI (Gradio)
python app.py --knowledge_base chroma_db

# API 서버 (FastAPI)
python api_server.py --knowledge_base chroma_db --port 8000
```

---

## 📁 프로젝트 구조

```
rag_chatbot/
├── knowledge_base/          # 지식 베이스
│   ├── documents/          # 원본 문서
│   └── processed/          # 처리된 데이터
├── vector_db/              # 벡터 데이터베이스
├── configs/
│   └── config.yaml         # 설정 파일
├── models/
│   ├── embedder.py         # Embedding 모델
│   └── llm.py              # LLM wrapper
├── retrieval/
│   ├── vector_store.py     # 벡터 DB 인터페이스
│   └── retriever.py        # 검색 로직
├── generation/
│   ├── prompt_template.py  # 프롬프트 템플릿
│   └── generator.py        # 생성 로직
├── utils/
│   ├── document_loader.py  # 문서 로딩
│   └── chunking.py         # 문서 분할
├── scripts/
│   ├── build_knowledge_base.py
│   └── evaluate_rag.py
├── chatbot.py              # CLI 챗봇
├── app.py                  # Gradio UI
├── api_server.py           # FastAPI 서버
└── README.md
```

---

## ⚙️ 설정

### config.yaml

```yaml
# Embedding Model
embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cuda"

# LLM
llm:
  provider: "openai"  # openai, huggingface, local
  model: "gpt-3.5-turbo"
  temperature: 0.7
  max_tokens: 500

# Vector DB
vector_db:
  type: "chroma"  # chroma, pinecone, weaviate
  persist_directory: "./vector_db"
  collection_name: "knowledge_base"

# Retrieval
retrieval:
  top_k: 3  # 검색할 문서 수
  similarity_threshold: 0.7

# Chunking
chunking:
  chunk_size: 500  # 문자 단위
  chunk_overlap: 50
  separator: "\n\n"
```

---

## 💬 사용 예시

### CLI 챗봇

```
$ python chatbot.py

RAG Chatbot v1.0
Knowledge Base: chroma_db (1250 documents)

User: VAE와 β-VAE의 차이점은 무엇인가요?