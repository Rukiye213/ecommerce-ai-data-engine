import random
import uuid
import psycopg2
from psycopg2.extras import execute_batch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
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
TOTAL_RECORDS = 500
BATCH_SIZE = 64  # Performanslı embedding üretimi için batch boyutu

print("⏳ Embedding modeli yükleniyor...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 2. Sentetik Veri Şablonları
CATEGORIES = {
    "Elektronik": [
        ("Kablosuz Kulaklık", "Kargo çok yavaş geldi, kutu hasarlıydı.", 1),
        ("Kablosuz Kulaklık", "Ses kalitesi ve baslar harika, şarjı 2 gün gidiyor.", 5),
        ("Akıllı Saat", "Şarjı 1 gün bile dayanmıyor, ekran dokunmatiği berbat.", 1),
        ("Akıllı Saat", "Fiyatına göre çok başarılı, bildirimleri anında alıyorum.", 4),
        ("Bluetooth Hoparlör", "Bağlantı sürekli kopuyor, cızırtı yapıyor.", 2),
        ("Bluetooth Hoparlör", "Ses seviyesi gayet tatmin edici, plajda kullanıyorum.", 5),
        ("Oyuncu Faresi", "Tıklama tuşu 2 hafta sonra bozuldu, çift tıklıyor.", 1),
        ("Oyuncu Faresi", "Ergonomisi çok iyi, sensör hassasiyeti başarılı.", 5),
    ],
    "Giyim": [
        ("Koşu Ayakkabısı", "Kalıbı aşırı dar, 42 giymeme rağmen ayağımı sıktı.", 2),
        ("Koşu Ayakkabısı", "Çok hafif ve rahat, taban yastıklaması mükemmel.", 5),
        ("Pamuklu Tişört", "İlk yıkamada 30 derecede çekti ve rengi soldu.", 2),
        ("Pamuklu Tişört", "Kumaş dokusu yumuşak, dikiş kalitesi harika.", 5),
        ("Oversize Sweatshirt", "Kumaşı çok ince, göründüğü gibi içi şardonlu değil.", 2),
        ("Oversize Sweatshirt", "Tam kışlık, rengi ve dokusu fotoğraftakinin aynısı.", 5),
        ("Keten Pantolon", "Beli tam oldu ama paçaları aşırı uzun geldi.", 3),
        ("Keten Pantolon", "Yaz ayları için vazgeçilmez, kumaşı hava alıyor.", 5),
    ],
    "Kozmetik": [
        ("Cilt Serumu", "Yüzümde kızarıklık ve sivilce yaptı, iade edeceğim.", 1),
        ("Cilt Serumu", "Düzenli kullanımda cildim parladı, nemlendirmesi harika.", 5),
        ("Göz Çevresi Kremi", "Milialara sebep oldu, hiç memnun kalmadım.", 1),
        ("Göz Çevresi Kremi", "Koyu halkalarımın görünümü belirgin şekilde azaldı.", 4),
        ("Şampuan", "Saç derimi kuruttu ve kepeklenme yaptı.", 2),
        ("Şampuan", "Saçlarımı yumuşacık yaptı ve dökülmeyi azalttı.", 5),
    ],
    "Ev & Yaşam": [
        ("Termos", "Sıcaklığı en fazla 2 saat tutuyor, sızdırma yaptı.", 2),
        ("Termos", "12 saat sonra bile çay kaynardı, kesinlikle sızdırmıyor.", 5),
        ("Dikey Süpürge", "Çekim gücü halılarda yetersiz kalıyor, pili çabuk bitiyor.", 2),
        ("Dikey Süpürge", "Pratik ve hafif, günlük kırıntı temizliği için mükemmel.", 4),
        ("Bambu Kesme Tahtası", "Islanınca çatladı ve küf kokmaya başladı.", 1),
        ("Bambu Kesme Tahtası", "Çok kaliteli ve dayanıklı, bıçak izi tutmuyor.", 5),
    ],
}

VARIATIONS = [
    "Kesinlikle tavsiye etmiyorum.",
    "Paketleme çok özenliydi.",
    "Kargo firması paketi kapıya fırlatıp gitmiş.",
    "Fiyat/performans açısından rakipsiz.",
    "Beklentimin çok altında kaldı, hayal kırıklığı.",
    "Bir daha asla bu satıcıdan almam.",
    "İndirimdeyken aldım, çok memnunum.",
    "Müşteri hizmetleri yardımcı olmadı, iade süreci uzadı.",
]


def generate_dataset(num_records=500):
    dataset = []
    category_names = list(CATEGORIES.keys())

    for _ in range(num_records):
        cat = random.choice(category_names)
        product, base_comment, base_rating = random.choice(CATEGORIES[cat])

        # Rastgele varyasyon ekleyerek cümleleri çeşitlendiriyoruz
        if random.random() > 0.4:
            comment = f"{base_comment} {random.choice(VARIATIONS)}"
        else:
            comment = base_comment

        # Puanı küçük dalgalanmalarla ayarla (1 ile 5 arasında)
        rating = max(1, min(5, base_rating + random.choice([-1, 0, 0, 1]) if base_rating in [2, 3, 4] else base_rating))

        dataset.append({
            "product": product,
            "category": cat,
            "rating": rating,
            "comment": comment
        })
    return dataset


def run_pipeline():
    print(f"\n📦 {TOTAL_RECORDS} adet e-ticaret yorumu üretiliyor...")
    data = generate_dataset(TOTAL_RECORDS)

    # 1. PostgreSQL Tablosunu Sıfırla ve Toplu Ekle
    print("🐘 PostgreSQL veritabanı hazırlanıyor...")
    conn = psycopg2.connect(**PG_CONFIG)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS reviews;")
    cur.execute("""
        CREATE TABLE reviews (
            id SERIAL PRIMARY KEY,
            product VARCHAR(100),
            category VARCHAR(50),
            rating INT,
            comment TEXT
        );
    """)

    insert_sql = "INSERT INTO reviews (product, category, rating, comment) VALUES (%s, %s, %s, %s) RETURNING id;"
    inserted_ids = []
    for item in data:
        cur.execute(insert_sql, (item["product"], item["category"], item["rating"], item["comment"]))
        inserted_ids.append(cur.fetchone()[0])

    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ PostgreSQL: {len(inserted_ids)} kayıt başarıyla eklendi.")

    # 2. Qdrant Koleksiyonunu Sıfırla ve Batch Olarak Vektörleri Bas
    print("⚡ Qdrant koleksiyonu hazırlanıyor...")
    QDRANT_CLIENT.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    comments = [item["comment"] for item in data]
    print(f"🧠 {len(comments)} yorum için Batch Embedding üretiliyor (Batch Size: {BATCH_SIZE})...")
    embeddings = model.encode(comments, batch_size=BATCH_SIZE, show_progress_bar=True)

    points = []
    for i, (item, pg_id) in enumerate(zip(data, inserted_ids)):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i].tolist(),
                payload={
                    "pg_id": pg_id,
                    "product": item["product"],
                    "category": item["category"],
                    "rating": item["rating"],
                    "comment": item["comment"],
                },
            )
        )

    print("🚀 Qdrant'a vektörler toplu olarak yükleniyor...")
    # 100'erli paketler halinde yükleme
    upload_chunk = 100
    for i in range(0, len(points), upload_chunk):
        QDRANT_CLIENT.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i:i + upload_chunk]
        )

    print(f"\n🎉 İŞLEM TAMAMLANDI: Hem PostgreSQL hem Qdrant {TOTAL_RECORDS} kayıt ile dolduruldu!")


if __name__ == "__main__":
    run_pipeline()