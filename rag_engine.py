import ollama

def generate_rag_response(query: str, search_results: list, intent: str) -> str:
    """
    Retrieval sonuçlarını Ollama modeline besleyerek yönetici içgörü raporu üretir.
    """
    if not search_results:
        return "İlgili sorguya ait veri bulunamadığı için özet üretilemedi."

    context_lines = []
    for idx, r in enumerate(search_results, start=1):
        if intent == "ANALYTICAL":
            context_lines.append(
                f"- Kategori: {r.get('category')}, Toplam Yorum: {r.get('total_reviews')}, "
                f"Ortalama Puan: {r.get('avg_rating')}, Negatif Yorum: {r.get('negative_reviews')}"
            )
        else:
            context_lines.append(
                f"{idx}. Ürün: {r.get('product')}, Puan: {r.get('rating')}, Yorum: {r.get('comment')}"
            )

    context_text = "\n".join(context_lines)

    system_prompt = (
        "Sen kurumsal bir e-ticaret veri analistisin. "
        "Görevin, sağlanan verileri inceleyip Türkçe, net ve kısa bir yönetici özeti sunmaktır. "
        "Aynı cümleleri asla tekrarlama. Sadece verilen verilere dayan."
    )

    user_prompt = f"""
Kullanıcı Sorusu: "{query}"

Veritabanı Bulguları:
{context_text}

Lütfen yukarıdaki verileri analiz ederek SADECE aşağıdaki 3 madde halinde kısa bir yanıt yaz:
- 📌 **Özet Değerlendirme:** (1-2 cümle)
- 💡 **Kritik Bulgular:** (Öne çıkan istatistikler)
- 🚀 **Aksiyon Önerisi:** (1 somut öneri)
"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        options={
            "temperature": 0.3,
            "repeat_penalty": 1.3,   # Tekrara girmeyi kesin olarak engeller
            "num_predict": 250       # Yanıtın maksimum uzunluğunu sınırlar
        }
    )

    return response["message"]["content"]