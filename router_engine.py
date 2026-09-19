import psycopg2
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range
from sentence_transformers import SentenceTransformer

# 1. Konfigürasyon
PG_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "ecommerce_db",
    "user": "user",
    "password": "password",
}
QDRANT_CLIENT = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "customer_reviews"
SCORE_THRESHOLD = 0.35  # Doğal eşleşmeleri kaçırmamak için optimize edildi

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def classify_intent(query: str) -> str:
    """Kullanıcının sorusunun niyetini belirler: 'ANALYTICAL', 'SEMANTIC' veya 'HYBRID'."""
    q = query.lower()

    analytical_keywords = ["kaç adet", "kaç tane", "ortalama", "oranı", "toplam", "istatistik", "dağılım"]
    hybrid_keywords = ["puanı", "yıldızlı", "yıldızın altı", "1 puan", "2 puan", "düşük puan", "kötü puan"]

    has_analytical = any(k in q for k in analytical_keywords)
    has_hybrid = any(k in q for k in hybrid_keywords)

    if has_hybrid and not has_analytical:
        return "HYBRID"
    elif has_analytical:
        return "ANALYTICAL"
    else:
        return "SEMANTIC"


def clean_semantic_query(query: str) -> str:
    """Hibrit aramalarda filtreleme belirten kelimeleri temizleyip sadece anlamsal özü bırakır."""
    stop_phrases = ["düşük puanlı siparişlerde", "düşük puanlı", "1 puanlı", "2 puanlı", "kötü puanlı"]
    q = query
    for p in stop_phrases:
        q = q.replace(p, "").replace(p.capitalize(), "")
    return q.strip()


def run_analytical_query(query: str):
    """PostgreSQL üzerinden sayısal özet döner."""
    print("\n⚡ [YÖNLENDİRİCİ KARARI: PostgreSQL (SQL Analitiği)]")
    conn = psycopg2.connect(**PG_CONFIG)
    cur = conn.cursor()

    cur.execute("""
        SELECT 
            category,
            COUNT(*) AS total_reviews,
            ROUND(AVG(rating), 2) AS avg_rating,
            COUNT(CASE WHEN rating <= 2 THEN 1 END) AS negative_reviews
        FROM reviews
        GROUP BY category;
    """)
    rows = cur.fetchall()

    print(f"📊 Sonuçlar ({query}):")
    print(f"{'Kategori':<15} | {'Toplam Yorum':<12} | {'Ort. Puan':<10} | {'Negatif (<=2)'}")
    print("-" * 55)
    for r in rows:
        print(f"{r[0]:<15} | {r[1]:<12} | {r[2]:<10} | {r[3]}")

    cur.close()
    conn.close()


def run_semantic_query(query: str, limit: int = 3):
    """Qdrant üzerinden semantik tarama yapar."""
    print("\n⚡ [YÖNLENDİRİCİ KARARI: Qdrant (Semantik Vektör Arama)]")
    query_vector = model.encode(query).tolist()

    search_result = QDRANT_CLIENT.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
    )

    print(f"🔍 '{query}' için Eşik Üstü (>={SCORE_THRESHOLD}) Sonuçlar:")
    matched = 0
    for hit in search_result.points:
        if hit.score >= SCORE_THRESHOLD:
            matched += 1
            p = hit.payload
            print(f"  • [{p['product']}] (Puan: {p['rating']} | Benzerlik: {hit.score:.4f})")
            print(f"    \"{p['comment']}\"")

    if matched == 0:
        print("  ❌ Bu konuyla doğrudan eşleşen bir yorum bulunamadı.")


def run_hybrid_query(query: str, max_rating: int = 2):
    """Qdrant Payload Filtresi ile doğrudan puanı <= max_rating olanlar arasında semantik arar."""
    print(f"\n⚡ [YÖNLENDİRİCİ KARARI: Hibrit Arama (Puan Filtresi <= {max_rating} + Qdrant)]")

    # Arama cümlesindeki "düşük puanlı" kalıbını atıp saf anlama odaklanıyoruz
    semantic_target = clean_semantic_query(query)
    print(f"🎯 Ayıklanan Semantik Konu: '{semantic_target}' (Filtre: rating <= {max_rating})")

    query_vector = model.encode(semantic_target).tolist()

    # Qdrant'ın kendi filtreleme mekanizması
    rating_filter = Filter(
        must=[
            FieldCondition(
                key="rating",
                range=Range(lte=max_rating)
            )
        ]
    )

    search_result = QDRANT_CLIENT.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=rating_filter,
        limit=3,
    )

    matched = 0
    print("🔎 Filtre İçinde Anlamsal Olarak En Çok Öne Çıkan Şikayetler:")
    for hit in search_result.points:
        if hit.score >= SCORE_THRESHOLD:
            matched += 1
            p = hit.payload
            print(f"  • [{p['product']}] (Puan: {p['rating']} | Benzerlik: {hit.score:.4f})")
            print(f"    \"{p['comment']}\"")

    if matched == 0:
        print("  ❌ Filtre kriterine uyan semantik sonuç bulunamadı.")


def voice_engine_ask(user_query: str):
    """Ana Giriş Kapısı."""
    print("\n" + "=" * 60)
    print(f"👤 KULLANICI SORUSU: \"{user_query}\"")
    
    intent = classify_intent(user_query)
    
    if intent == "ANALYTICAL":
        run_analytical_query(user_query)
    elif intent == "HYBRID":
        run_hybrid_query(user_query, max_rating=2)
    else:
        run_semantic_query(user_query)


if __name__ == "__main__":
    # Test Senaryosu 1: Analitik Soru
    voice_engine_ask("Kategorilerin ortalama puanı ve toplam yorum sayısı kaç?")

    # Test Senaryosu 2: Semantik Soru
    voice_engine_ask("ayakkabının kalıbı ve numarası dar")

    # Test Senaryosu 3: Hibrit Soru
    voice_engine_ask("Düşük puanlı siparişlerde paketleme ve ezilme sorunu")