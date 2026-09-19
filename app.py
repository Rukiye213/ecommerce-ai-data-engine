import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import datetime

BACKEND_URL = "http://127.0.0.1:8000/query"

st.set_page_config(
    page_title="E-Commerce AI Analytics Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS İyileştirmeleri
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }

    .header-card {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px 30px;
        margin-bottom: 24px;
    }
    
    .status-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .pill-sql { background: rgba(56, 189, 248, 0.15); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.3); }
    .pill-hybrid { background: rgba(245, 158, 11, 0.15); color: #F59E0B; border: 1px solid rgba(245, 158, 11, 0.3); }
    .pill-semantic { background: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.3); }

    .report-container {
        background: #0B132B;
        border-left: 4px solid #38BDF8;
        border-radius: 8px;
        padding: 24px;
        color: #E2E8F0;
        line-height: 1.8;
    }

    .kpi-box {
        background: #161F30;
        border: 1px solid #23324A;
        border-radius: 10px;
        padding: 16px 20px;
    }
</style>
""", unsafe_allow_html=True)

if "query_input" not in st.session_state:
    st.session_state["query_input"] = ""
if "history" not in st.session_state:
    st.session_state["history"] = []

def set_prompt(text):
    st.session_state["query_input"] = text

# Sidebar
with st.sidebar:
    st.markdown("### 📊 **E-Commerce AI Hub**")
    st.caption("Hibrit Analitik & RAG Karar Destek Sistemi")
    st.markdown("---")
    
    sample_limit = st.slider("İncelenecek Yorum Adedi", min_value=2, max_value=10, value=4)
    
    st.markdown("---")
    st.markdown("##### 📌 **Örnek Sorgu Senaryoları**")
    st.button("📊 Kategori Puan ve Hacim Analizi", on_click=set_prompt, args=("Kategorilerin ortalama puanı ve toplam yorum sayısı kaç?",), use_container_width=True)
    st.button("⚡ Donanım ve Şarj Şikayetleri", on_click=set_prompt, args=("düşük puanlı siparişlerde şarjı çabuk biten ve arızalı cihazlar",), use_container_width=True)
    st.button("📦 Hasarlı Teslimat ve Paketleme", on_click=set_prompt, args=("düşük puanlı siparişlerde bozuk ve arızalı ürünler",), use_container_width=True)
    st.button("🚚 Hızlı Teslimat Memnuniyeti", on_click=set_prompt, args=("hızlı teslimat ve özenli paketleme yapılan siparişler",), use_container_width=True)

    if st.session_state["history"]:
        st.markdown("---")
        st.markdown("##### 🕒 **Sorgu Geçmişi**")
        for item in reversed(st.session_state["history"][-5:]):
            st.caption(f"• **[{item['intent']}]** {item['query'][:24]}...")

# Ana Başlık Kartı
st.markdown("""
<div class="header-card">
    <div style="font-size: 1.5rem; font-weight: 700; color: #FFFFFF; display: flex; align-items: center; gap: 10px;">
        🛍️ E-Commerce AI Analytics Engine
    </div>
    <div style="font-size: 0.9rem; color: #94A3B8; margin-top: 6px;">
        PostgreSQL (İlişkisel/Filtre) + Qdrant (Kosinüs Vektör Arama) + Llama 3.2 (Yerel RAG Sentezi)
    </div>
</div>
""", unsafe_allow_html=True)

# Arama Alanı
col_input, col_btn = st.columns([5, 1])
with col_input:
    user_query = st.text_input(
        "Sorgu Alanı",
        key="query_input",
        placeholder="Doğal dilde analiz sorusu girin (örn: düşük puanlı ürünlerde arıza ve kalite sorunları)...",
        label_visibility="collapsed"
    )
with col_btn:
    submit = st.button("Analiz Et", type="primary", use_container_width=True)

# Çıktı Bölümü
if submit:
    if not user_query.strip():
        st.warning("Lütfen bir analiz sorusu yazın veya sol menüdeki hazır senaryolardan birine tıklayın.")
    else:
        with st.spinner("Sorgu sınıflandırılıyor, veritabanı taranıyor ve yönetici raporu hazırlanıyor..."):
            try:
                res = requests.post(
                    BACKEND_URL,
                    json={"query": user_query, "limit": sample_limit},
                    timeout=60
                )
                
                if res.status_code == 200:
                    data = res.json()
                    intent = data.get("intent", "UNKNOWN")
                    source = data.get("data_source", "Unknown")
                    summary = data.get("ai_summary", "")
                    results = data.get("results", [])

                    st.session_state["history"].append({
                        "query": user_query,
                        "intent": intent
                    })

                    # Intent Rozeti Seçimi
                    pill_class = "pill-sql" if intent == "ANALYTICAL" else ("pill-hybrid" if intent == "HYBRID" else "pill-semantic")

                    # Üst Metrikler
                    st.markdown("<br>", unsafe_allow_html=True)
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.markdown(f"""
                        <div class="kpi-box">
                            <div style="font-size:0.75rem; color:#94A3B8; text-transform:uppercase;">Sorgu Rotası</div>
                            <div style="margin-top:6px;"><span class="status-pill {pill_class}">{intent}</span></div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"""
                        <div class="kpi-box">
                            <div style="font-size:0.75rem; color:#94A3B8; text-transform:uppercase;">Veri Katmanı</div>
                            <div style="font-size:1.15rem; font-weight:700; color:#F8FAFC; margin-top:4px;">{source.split(' ')[0]}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m3:
                        st.markdown(f"""
                        <div class="kpi-box">
                            <div style="font-size:0.75rem; color:#94A3B8; text-transform:uppercase;">Taranan Kayıt</div>
                            <div style="font-size:1.15rem; font-weight:700; color:#F8FAFC; margin-top:4px;">{len(results)} Adet</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m4:
                        st.markdown(f"""
                        <div class="kpi-box">
                            <div style="font-size:0.75rem; color:#94A3B8; text-transform:uppercase;">Çıkarım Motoru</div>
                            <div style="font-size:1.15rem; font-weight:700; color:#38BDF8; margin-top:4px;">Llama 3.2 (3B)</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Sekmeler
                    tab_report, tab_chart, tab_raw = st.tabs([
                        "📋 Yönetici Değerlendirme Raporu",
                        "📈 Veri Dağılımı ve Grafikler",
                        "🔍 İncelenen Ham Veriler"
                    ])

                    with tab_report:
                        st.markdown(f"""
                        <div class="report-container">
                            <div style="font-size:1.1rem; font-weight:700; color:#38BDF8; margin-bottom:12px;">
                                Yapay Zekâ Stratejik Özet Raporu
                            </div>
                            {summary.replace(chr(10), '<br>')}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.download_button(
                            label="📥 Raporu Markdown Olarak İndir",
                            data=f"# E-Commerce Analytics Export\nSorgu: {user_query}\nRota: {intent}\n\n{summary}",
                            file_name=f"analiz_raporu_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.md",
                            mime="text/markdown"
                        )

                    with tab_chart:
                        if intent == "ANALYTICAL" and results:
                            df = pd.DataFrame(results)
                            c1, c2 = st.columns(2)
                            with c1:
                                fig_bar = px.bar(
                                    df, x="category", y="avg_rating",
                                    color="avg_rating", color_continuous_scale="Blues",
                                    title="Kategori Bazında Ortalama Puanlar", text="avg_rating"
                                )
                                fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                                fig_bar.update_layout(template="plotly_dark", yaxis_range=[0, 5])
                                st.plotly_chart(fig_bar, use_container_width=True)
                            with c2:
                                fig_pie = px.pie(
                                    df, names="category", values="negative_reviews",
                                    hole=0.55, title="Negatif Geri Bildirimlerin Kategori Dağılımı",
                                    color_discrete_sequence=px.colors.sequential.Teal
                                )
                                fig_pie.update_layout(template="plotly_dark")
                                st.plotly_chart(fig_pie, use_container_width=True)

                        elif results:
                            df = pd.DataFrame(results)
                            c1, c2 = st.columns(2)
                            with c1:
                                if "rating" in df.columns:
                                    fig_hist = px.histogram(
                                        df, x="rating", title="Eşleşen Yorumların Puan Dağılımı",
                                        color_discrete_sequence=["#38BDF8"]
                                    )
                                    fig_hist.update_layout(template="plotly_dark", xaxis=dict(dtick=1))
                                    st.plotly_chart(fig_hist, use_container_width=True)
                            with c2:
                                if "similarity_score" in df.columns:
                                    fig_sc = px.scatter(
                                        df, x="product", y="similarity_score",
                                        size="similarity_score", color="similarity_score",
                                        color_continuous_scale="Viridis",
                                        title="Kosinüs Benzerlik Skorları (Güven Düzeyi)"
                                    )
                                    fig_sc.update_layout(template="plotly_dark", yaxis_range=[0, 1])
                                    st.plotly_chart(fig_sc, use_container_width=True)
                        else:
                            st.info("Bu sorgu türü için görselleştirme verisi üretilmedi.")

                    with tab_raw:
                        if intent == "ANALYTICAL":
                            st.dataframe(pd.DataFrame(results), use_container_width=True)
                        else:
                            for idx, row in enumerate(results, start=1):
                                score = row.get("similarity_score", 0.0)
                                rating = row.get("rating", 0)
                                stars = "★" * rating + "☆" * (5 - rating)
                                with st.expander(f"Kayıt #{idx} — {row.get('product', 'Ürün')} | {stars} ({rating}/5) — Benzerlik: {score:.4f}"):
                                    st.write(row.get("comment", ""))

                else:
                    st.error(f"Backend Hatası ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Bağlantı hatası: {str(e)}")