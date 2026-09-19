import psycopg2
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

# 1. Konfigürasyonlar
PG_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "ecommerce_db",
    "user": "user",
    "password": "password",
}
QDRANT_CLIENT = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "customer_reviews"

print("🔄 Embedding modeli yükleniyor...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 2. Gerçek E-Ticaret Yorum Veri Kümesini İndir
print("📥 Hugging Face'ten gerçek e-ticaret yorumları indiriliyor...")
ds = load_dataset("winvoker/turkish-sentiment-analysis-dataset", split="train")

samples_per_rating = 200
balanced_records = []
rating_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

for item in ds:
    comment_text = str(item.get("text", "")).strip()
    label = str(item.get("label", "")).lower()

    if not comment_text or len(comment_text) < 15:
        continue

    text_lower = comment_text.lower()

    # Gerçek E-Ticaret Kategorisi Belirleme
    if any(
        w in text_lower
        for w in [
            "telefon", "kulaklık", "şarj", "batarya", "kablo", "ekran",
            "cihaz", "bilgisayar", "mouse", "klavye", "ses", "bağlantı"
        ]
    ):
        category = "Elektronik"
    elif any(
        w in text_lower
        for w in [
            "krem", "parfüm", "şampuan", "cilt", "koku", "nemlendirici",
            "serum", "saç", "makyaj", "ruj"
        ]
    ):
        category = "Kozmetik"
    elif any(
        w in text_lower
        for w in [
            "beden", "kumaş", "kalıp", "tişört", "ayakkabı", "elbise",
            "kot", "pamuk", "rahat", "spor"
        ]
    ):
        category = "Giyim"
    else:
        category = "Ev & Yaşam"

    # Duygu etiketine göre 1-5 puan dağılımı
    is_negative = "neg" in label or "0" in label
    if is_negative:
        if any(
            w in text_lower
            for w in [
                "çöp", "bozuk", "kırık", "iade", "berbat", "çalışmıyor",
                "rezalet", "arızalı", "sakın"
            ]
        ):
            assigned_rating = 1
        else:
            assigned_rating = 2
    else:
        if any(
            w in text_lower
            for w in [
                "fena değil", "idare eder", "orta", "fiyatına göre", "beklentim"
            ]
        ):
            assigned_rating = 3
        elif any(
            w in text_lower
            for w in ["hızlı", "güzel", "beğendim", "iyi", "başarılı"]
        ):
            assigned_rating = 4
        else:
            assigned_rating = 5

    # Dengeli dağıtım için puan kotası kontrolü
    if rating_counts[assigned_rating] < samples_per_rating:
        balanced_records.append({
            "product": f"Ürün - {category}",
            "category": category,
            "rating": assigned_rating,
            "comment": comment_text[:300],
        })
        rating_counts[assigned_rating] += 1

    if sum(rating_counts.values()) >= 1000:
        break

print(f"📊 Dengeli E-Ticaret Dağılımı: {rating_counts}")

# 3. PostgreSQL Tablosunu Temizle ve Yeni Verileri Ekle
print("🐘 PostgreSQL tablosu güncelleniyor...")
conn = psycopg2.connect(**PG_CONFIG)
cur = conn.cursor()

cur.execute("TRUNCATE TABLE reviews RESTART IDENTITY;")

insert_query = """
INSERT INTO reviews (product, category, rating, comment)
VALUES (%s, %s, %s, %s);
"""

for r in balanced_records:
    cur.execute(
        insert_query, (r["product"], r["category"], r["rating"], r["comment"])
    )

conn.commit()
cur.close()
conn.close()
print("✅ PostgreSQL'e gerçek ürün yorumları başarıyla yazıldı.")

# 4. Qdrant Vektör Tabanını Temizle ve Yeni Vektörleri Bas
print("🔍 Qdrant koleksiyonu sıfırlanıyor ve vektörler üretiliyor...")
QDRANT_CLIENT.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

texts = [r["comment"] for r in balanced_records]
vectors = model.encode(texts, show_progress_bar=True).tolist()

points = [
    PointStruct(
        id=idx,
        vector=vectors[idx],
        payload={
            "product": balanced_records[idx]["product"],
            "category": balanced_records[idx]["category"],
            "rating": balanced_records[idx]["rating"],
            "comment": balanced_records[idx]["comment"],
        },
    )
    for idx in range(len(balanced_records))
]

QDRANT_CLIENT.upsert(collection_name=COLLECTION_NAME, points=points)
print("✅ Qdrant e-ticaret vektör tabanı başarıyla güncellendi.")