import re
import requests
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

app = FastAPI(title="E-Commerce AI Analytics Engine")

# Modeller ve İstemciler
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
qdrant_client = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "customer_reviews"
OLLAMA_URL = "http://localhost:11434/api/generate"

def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        database="ecommerce_db",
        user="user",
        password="password",
        port=5432
    )

class QueryRequest(BaseModel):
    query: str
    limit: int = 4

def detect_intent(query: str) -> str:
    q = query.lower()
    analytical_keywords = ["kaç", "ortalama", "toplam", "istatistik", "oran", "yüzde", "dağılım", "kategori"]
    hybrid_keywords = ["düşük puan", "1 yıldız", "2 yıldız", "kötü", "bozuk", "arızalı", "şikayet", "iade"]
    
    is_analytical = any(k in q for k in analytical_keywords)
    is_hybrid = any(k in q for k in hybrid_keywords)
    
    if is_analytical and not is_hybrid:
        return "ANALYTICAL"
    elif is_hybrid:
        return "HYBRID"
    else:
        return "SEMANTIC"

def generate_rag_summary(query: str, context: str) -> str:
    system_instruction = (
        "GÖREV: E-ticaret müşteri yorumlarını analiz eden Türkçe veri uzmanısın.\n"
        "KESİN DİL KURALLARI:\n"
        "1. Yalnızca duru ve doğal Türkçe kelimeler kullan.\n"
        "2. Asla İngilizce kelime kullanma! "
        "'customer' yerine 'müşteri', 'review/comment' yerine 'yorum', 'product' yerine 'ürün', "
        "'feedback' yerine 'geri bildirim', 'issue' yerine 'sorun', 'satisfied' yerine 'memnun' yaz.\n"
        "3. Uydurma heceler veya yabancı ifadeler kesinlikle yasaktır.\n\n"
        "ÇIKTI FORMATI:\n"
        "**Özet Değerlendirme**\n"
        "(Müşterilerin genel memnuniyet veya şikayet durumunu 2-3 cümleyle açıkla)\n\n"
        "**Kritik Bulgular**\n"
        "- (Öne çıkan ilk tespit)\n"
        "- (Öne çıkan ikinci tespit)\n\n"
        "**Aksiyon Önerisi**\n"
        "- (Satıcı veya operasyon ekibinin atması gereken adım)"
    )

    prompt = (
        f"{system_instruction}\n\n"
        f"Kullanıcı Talebi: {query}\n\n"
        f"Analiz Edilecek Ham Yorum Verileri:\n{context}\n\n"
        f"Türkçe Rapor:"
    )

    payload = {
        "model": "llama3.2:3b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.05,     # Dil sapmasını ve uydurmayı engeller
            "top_p": 0.85,
            "repeat_penalty": 1.2
        }
    }

    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=60)
        if res.status_code == 200:
            return res.json().get("response", "").strip()
        else:
            return "Yapay zekâ analiz üretirken bir sorun oluştu."
    except Exception as e:
        return f"Ollama bağlantı hatası: {str(e)}"

@app.post("/query")
def process_query(req: QueryRequest):
    intent = detect_intent(req.query)
    results = []
    data_source = ""
    context_text = ""

    try:
        # 1. ANALYTICAL (PostgreSQL)
        if intent == "ANALYTICAL":
            data_source = "PostgreSQL (Agregasyon Motoru)"
            conn = get_db_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            sql = """
                SELECT 
                    category,
                    COUNT(*) as total_reviews,
                    ROUND(AVG(rating)::numeric, 2) as avg_rating,
                    COUNT(CASE WHEN rating <= 2 THEN 1 END) as negative_reviews
                FROM reviews
                GROUP BY category
                ORDER BY total_reviews DESC;
            """
            cur.execute(sql)
            rows = cur.fetchall()
            cur.close()
            conn.close()

            results = [dict(r) for r in rows]
            context_text = "\n".join([
                f"- Kategori: {r['category']} | Ortalama: {r['avg_rating']} | Toplam: {r['total_reviews']} | Negatif: {r['negative_reviews']}"
                for r in results
            ])

        # 2. HYBRID (SQL Filtre + Qdrant Vektör)
        elif intent == "HYBRID":
            data_source = "PostgreSQL (Filtre) + Qdrant (Kosinüs Vektör)"
            
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT id FROM reviews WHERE rating <= 2;")
            ids = [row[0] for row in cur.fetchall()]
            cur.close()
            conn.close()

            if not ids:
                return {
                    "intent": intent,
                    "data_source": data_source,
                    "ai_summary": "Kriterlere uygun kayıt bulunamadı.",
                    "results": []
                }

            query_vector = embed_model.encode(req.query).tolist()
            
            q_filter = models.Filter(
                must=[
                    models.HasIdCondition(has_id=ids[:250])
                ]
            )

            try:
                search_res = qdrant_client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_vector,
                    query_filter=q_filter,
                    limit=req.limit
                ).points
            except AttributeError:
                search_res = qdrant_client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_vector,
                    query_filter=q_filter,
                    limit=req.limit
                )

            context_parts = []
            for hit in search_res:
                p = hit.payload
                results.append({
                    "product": p.get("product", "Ürün"),
                    "rating": p.get("rating", 0),
                    "comment": p.get("comment", ""),
                    "similarity_score": round(hit.score, 4)
                })
                context_parts.append(f"- Ürün: {p.get('product')} (Puan: {p.get('rating')}/5): {p.get('comment')}")
            
            context_text = "\n".join(context_parts)

        # 3. SEMANTIC (Saf Vektör Arama)
        else:
            data_source = "Qdrant (Kosinüs Vektör Arama)"
            query_vector = embed_model.encode(req.query).tolist()
            
            try:
                search_res = qdrant_client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_vector,
                    limit=req.limit
                ).points
            except AttributeError:
                search_res = qdrant_client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_vector,
                    limit=req.limit
                )

            context_parts = []
            for hit in search_res:
                p = hit.payload
                results.append({
                    "product": p.get("product", "Ürün"),
                    "rating": p.get("rating", 0),
                    "comment": p.get("comment", ""),
                    "similarity_score": round(hit.score, 4)
                })
                context_parts.append(f"- Ürün: {p.get('product')} (Puan: {p.get('rating')}/5): {p.get('comment')}")
                
            context_text = "\n".join(context_parts)

        # RAG Çıkarımı
        ai_summary = generate_rag_summary(req.query, context_text)

        return {
            "intent": intent,
            "data_source": data_source,
            "ai_summary": ai_summary,
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))