import psycopg2
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# 1. Bağlantı Ayarları
PG_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "ecommerce_db",
    "user": "user",
    "password": "password",
}
QDRANT_CLIENT = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "customer_reviews"

# 2. Embedding Modeli
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def search_semantic_feedback(user_query: str, limit: int = 3):
    """Kullanıcının sorusunu vektöre çevirip Qdrant'ta anlamsal olarak en yakın yorumları bulur."""
    print(f"\n🔍 Semantik Arama Yapılıyor: '{user_query}'")

    query_vector = model.encode(user_query).tolist()

    # Güncel Qdrant istemcisinde arama metodu: query_points
    search_result = QDRANT_CLIENT.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
    )

    print("--- En Yakın Müşteri Yorumları ---")
    for hit in search_result.points:
        p = hit.payload
        score = round(hit.score, 4)
        print(
            f"⭐ Puan: {p['rating']} | Ürün: {p['product']} | Benzerlik Skoru: {score}"
        )
        print(f"   Yorum: \"{p['comment']}\"")


def get_analytical_metrics():
    """PostgreSQL'e bağlanıp kategorilere göre ortalama puan ve yorum sayılarını hesaplar."""
    print("\n📊 Analitik Özet (PostgreSQL SQL Sorgusu):")

    conn = psycopg2.connect(**PG_CONFIG)
    cur = conn.cursor()

    cur.execute("""
        SELECT category, COUNT(*) as total_reviews, ROUND(AVG(rating), 2) as avg_rating
        FROM reviews
        GROUP BY category;
    """)

    rows = cur.fetchall()
    print("Kategori         | Yorum Sayısı | Ortalama Puan")
    print("-----------------------------------------------")
    for r in rows:
        print(f"{r[0]:<16} | {r[1]:<12} | {r[2]}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    # 1. SQL Analitiği Testi
    get_analytical_metrics()

    # 2. Semantik Vektör Arama Testi (Kargo / Hasar problemi)
    search_semantic_feedback("paketim ezilmiş ve teslimat gecikti")

    # 3. Semantik Vektör Arama Testi (Kalıp / Beden şikayeti)
    search_semantic_feedback("ayakkabının bedeni küçük geldi")