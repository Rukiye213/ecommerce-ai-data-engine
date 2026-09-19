import psycopg2
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

# 1. Bağlantı Ayarları
PG_HOST = "localhost"
PG_PORT = 5432
PG_DB = "ecommerce_db"
PG_USER = "user"
PG_PASS = "password"

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "customer_reviews"

# 2. Örnek E-Ticaret Yorum Veri Seti
SAMPLE_REVIEWS = [
    {"id": 1, "product": "Kablosuz Kulaklık", "category": "Elektronik", "rating": 1, "comment": "Kargo 10 günde zor geldi, kutu paramparçaydı ve ambalaj yırtılmıştı."},
    {"id": 2, "product": "Kablosuz Kulaklık", "category": "Elektronik", "rating": 2, "comment": "Yürürken Bluetooth bağlantısı sürekli kopuyor, sol kulaklıktan dip ses geliyor."},
    {"id": 3, "product": "Kablosuz Kulaklık", "category": "Elektronik", "rating": 5, "comment": "Ses kalitesi muazzam, baslar çok doyurucu ve gürültü engelleme kusursuz çalışıyor."},
    {"id": 4, "product": "Koşu Ayakkabısı", "category": "Giyim", "rating": 2, "comment": "Kalıbı inanılmaz dar. 42 giyiyorum ama ayağımı sıktı, en az 1 numara büyük alınmalı."},
    {"id": 5, "product": "Koşu Ayakkabısı", "category": "Giyim", "rating": 5, "comment": "Tabanı bulut gibi çok rahat. Bütün gün ayakta çalışanlar için kesinlikle tavsiye ederim."},
    {"id": 6, "product": "Akıllı Saat", "category": "Elektronik", "rating": 1, "comment": "Şarjı 1 gün bile dayanmıyor, ekran dokunmatiği bazen hiç algılamıyor. İade ettim."},
    {"id": 7, "product": "Pamuklu Tişört", "category": "Giyim", "rating": 2, "comment": "İlk 30 derece yıkamada çekti ve rengi soldu. Kumaş kalitesi beklentimin çok altında."},
    {"id": 8, "product": "Pamuklu Tişört", "category": "Giyim", "rating": 5, "comment": "Kumaşı yumuşacık ve dikişleri çok sağlam. Tam bedeninizi tercih edebilirsiniz."}
]

def setup_postgresql():
    print("🐘 PostgreSQL'e bağlanılıyor ve tablo oluşturuluyor...")
    conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS)
    cur = conn.cursor()
    
    # Tabloyu sıfırla ve yeniden kur
    cur.execute("DROP TABLE IF EXISTS reviews;")
    cur.execute("""
        CREATE TABLE reviews (
            id INT PRIMARY KEY,
            product VARCHAR(100),
            category VARCHAR(50),
            rating INT,
            comment TEXT
        );
    """)
    
    for r in SAMPLE_REVIEWS:
        cur.execute(
            "INSERT INTO reviews (id, product, category, rating, comment) VALUES (%s, %s, %s, %s, %s);",
            (r["id"], r["product"], r["category"], r["rating"], r["comment"])
        )
    conn.commit()
    cur.close()
    conn.close()
    print("✅ PostgreSQL'e veri başarıyla yazıldı.")

def setup_qdrant():
    print("\n🧠 Embedding modeli yükleniyor ve vektörler oluşturuluyor...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    q_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Vektör boyutu (MiniLM modeli için 384'tür)
    vector_size = 384
    
    # Koleksiyon varsa baştan oluştur
    q_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

    points = []
    for r in SAMPLE_REVIEWS:
        # Metni 384 boyutlu sayı dizisine çeviriyoruz
        vector = model.encode(r["comment"]).tolist()
        points.append(
            PointStruct(
                id=r["id"],
                vector=vector,
                payload={
                    "product": r["product"],
                    "category": r["category"],
                    "rating": r["rating"],
                    "comment": r["comment"]
                }
            )
        )
    
    q_client.upsert(collection_name=COLLECTION_NAME, points=points)
    print("✅ Qdrant vektör veritabanına tüm kayıtlar başarıyla eklendi.")

if __name__ == "__main__":
    setup_postgresql()
    setup_qdrant()
    print("\n🎉 Veri boru hattı (ETL) tamamlandı!")