# E-Commerce AI Analytics Engine & Decision Support System

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue.svg)](https://www.postgresql.org/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-purple.svg)](https://qdrant.tech/)
[![Ollama](https://img.shields.io/badge/Ollama-Llama_3.2_3B-black.svg)](https://ollama.ai/)

Çok kaynaklı e-ticaret müşteri geri bildirimlerini, ürün puanlarını ve sipariş metriklerini analiz eden; **Akıllı Sorgu Yönlendirme (Intent-Based Dynamic Routing)** ve **Hibrit RAG (Retrieval-Augmented Generation)** mimarisine sahip kurumsal karar destek platformu.

---

## Mimari Genel Bakış (Architecture Pipeline)

Sistem, gelen kullanıcı sorgusunun anlamsal niyetini analiz ederek sorguyu en optimize veri katmanına ve çıkarım motoruna dinamik olarak yönlendirir:

```text
                          ┌───────────────────────┐
                          │   Kullanıcı Sorgusu   │
                          └──────────┬────────────┘
                                     │
                        ┌────────────▼────────────┐
                        │ Dynamic Intent Routing  │
                        └────────────┬────────────┘
         ┌───────────────────────────┼───────────────────────────┐
         │                           │                           │
  [ANALYTICAL]                   [HYBRID]                   [SEMANTIC]
         │                           │                           │
┌────────▼─────────┐   ┌─────────────▼─────────────┐   ┌─────────▼─────────┐
│ PostgreSQL Engine│   │ PostgreSQL SQL Filter     │   │ Qdrant Vector DB  │
│ (Agregasyon,     │   │         +                 │   │ (Kosinüs Benzerlik│
│ Ortalama, Adet)  │   │ Qdrant Vector Search      │   │ MiniLM-L12-v2)    │
└────────┬─────────┘   └─────────────┬─────────────┘   └─────────┬─────────┘
         │                           │                           │
         └───────────────────────────┼───────────────────────────┘
                                     │
                        ┌────────────▼────────────┐
                        │   Context Construction  │
                        └────────────┬────────────┘
                                     │
                        ┌────────────▼────────────┐
                        │   Ollama (Llama 3.2 3B) │
                        │  Stratejik Karar Raporu │
                        └────────────┬────────────┘
                                     │
                        ┌────────────▼────────────┐
                        │ Streamlit Executive Hub │
                        │ (Grafik, Dağılım, Ham)  │
                        └─────────────────────────┘

Öne Çıkan Özellikler
Dinamik Sorgu Yönlendirme (Intent Router):

Analitik Rota (ANALYTICAL): SQL agregasyonları ile kategori bazlı hacim, ortalama puan ve negatif dağılım analizleri.

Hibrit Rota (HYBRID): İlişkisel veritabanı filtreleri (örn: rating <= 2) ile vektörel anlamsal benzerlik aramasının birleşimi.

Semantik Rota (SEMANTIC): Cümle düzeyinde anlamsal yakınlık (paraphrase-multilingual-MiniLM-L12-v2) üzerinden doğrudan kosinüs benzerliği eşleştirmesi.

Yerel LLM Çıkarımı (Local Inference): Ollama altyapısı üzerinde çalışan Llama 3.2 (3B) ile dışa bağımlılık olmadan, düşük sıcaklık (temperature: 0.05) ve kısıtlayıcı sistem talimatları ile yüksek kaliteli kurumsal Türkçe yönetici raporları.

Yönetici Karar Paneli (Executive Dashboard): Streamlit ile tasarlanmış puan dağılım histogramları, kosinüs güven skoru saçılım grafikleri ve genişletilebilir ham kanıt kartları.

 Teknoloji Yığını (Tech Stack)
Backend / API: FastAPI, Uvicorn, Pydantic

Frontend / Dashboard: Streamlit, Plotly

Vektör Veritabanı: Qdrant

İlişkisel Veritabanı: PostgreSQL 16

Embedding Modeli: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

LLM Engine: Ollama / Llama 3.2 (3B)

Konteynerizasyon: Docker & Docker Compose

 Kurulum ve Çalıştırma
1. Gereksinimler
Python 3.10+

Docker Desktop

Ollama (ollama run llama3.2:3b)

2. Depoyu Klonlama ve Sanal Ortam
Bash
git clone [https://github.com/](https://github.com/)<KULLANICI_ADIN>/ecommerce-ai-data-engine.git
cd ecommerce-ai-data-engine

python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
3. Veritabanlarını Başlatma (Docker)
Bash
docker-compose up -d postgres qdrant
4. Servisleri Başlatma
Backend API (Terminal 1):

Bash
uvicorn main:app --reload --port 8000
Frontend Dashboard (Terminal 2):

Bash
streamlit run app.py
Tarayıcınızda http://localhost:8501 adresine giderek sistemi kullanmaya başlayabilirsiniz.

