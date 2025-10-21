# --- [WinError 1114] DLL HATASI ÇÖZÜMÜ ---
# Bu komut, torch veya datasets import edilmeden EN ÖNCE çalışmalıdır.
# Sisteme "Çakışan DLL'leri (libiomp5md.dll) görmezden gel" der.
# Bu, özellikle sentence-transformers gibi torch kullanan kütüphanelerin
# Windows'ta neden olduğu yaygın bir sorunu çözer.
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# --- ÇÖZÜM SONU ---

import streamlit as st
import google.generativeai as genai
from datasets import load_dataset  # Hugging Face veri setlerini yüklemek için
from dotenv import load_dotenv # API anahtarını .env dosyasından okumak için
import sys # DLL çözümü için sys.prefix kullanıldı (artık kullanılmıyor ama import kalabilir)
import time # Embedding sırasında gecikme eklemek için (artık kullanılmıyor ama import kalabilir)

# Lokal Embedding (açık kaynak model) için gerekli importlar
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# RAG Pipeline ve Generation için gerekli importlar
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. PROJE KURULUMU VE API ANAHTARI YÜKLEME ---

# .env dosyasını, script'in bulunduğu dizinden tam yolunu bularak yükle
# Bu, uygulamanın farklı ortamlarda çalıştırıldığında .env dosyasını bulmasını garantiler.
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(script_dir, ".env")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        print(f".env dosyası tam yoldan yüklendi: {dotenv_path}")
    else:
        # Hata durumunda aşağıdaki try bloğu bunu yakalayacak.
        print(f"UYARI: .env dosyası şu yolda bulunamadı: {dotenv_path}")
except Exception as e:
    st.error(f".env dosyası yüklenirken beklenmedik hata: {e}")
    st.stop()

# Google API anahtarını ortam değişkenlerinden al (Sadece Generation için)
try:
    api_key = os.environ["GOOGLE_API_KEY"]
    if not api_key:
        raise ValueError("GOOGLE_API_KEY .env dosyasında bulunamadı veya boş.")
    genai.configure(api_key=api_key)
    print("Google API Anahtarı başarıyla yüklendi.")
except (KeyError, ValueError) as e:
    st.error(f"HATA: GOOGLE_API_KEY ortam değişkeni ayarlanmamış. Lütfen .env dosyanızı kontrol edin. Hata: {e}")
    st.stop()

# --- STREAMLIT ARAYÜZ AYARLARI ---
st.set_page_config(page_title="Psikoloji Bilgi Asistanı", page_icon="🧠")
st.title("🧠 Psikoloji Bilgi Asistanı")
st.caption("Bu chatbot, RAG mimarisi kullanılarak geliştirilmiştir ve yalnızca yüklenen veri setindeki bilgilere dayanarak yanıt verir.")


# --- 2. VERİ SETİNİ YÜKLEME ---
# Hugging Face'ten psikoloji danışmanlık veri setini yükle
# Bu veri seti, RAG sistemimizin bilgi kaynağı (knowledge base) olacaktır.
@st.cache_data # Veri setini önbelleğe al, her seferinde indirme
def load_data():
    try:
        dataset_dict = load_dataset("Amod/mental_health_counseling_conversations")
        # Veri setindeki ilk bölümü (split) al, genellikle 'train' olur.
        split_name = list(dataset_dict.keys())[0]
        return dataset_dict[split_name]
    except Exception as e:
        st.error(f"Veri seti yüklenirken bir hata oluştu: {e}")
        st.stop()

dataset = load_data()
st.success(f"Veri seti başarıyla yüklendi! Toplam {len(dataset)} adet konuşma bulundu.")

# Verinin yapısını göstermek için arayüzde bir örnek göster (opsiyonel)
with st.expander("Veri Setinden Bir Örnek Göster"):
    st.text_area("Örnek Soru (Context)", dataset[0]['Context'], height=100)
    st.text_area("Örnek Cevap (Response)", dataset[0]['Response'], height=200)

# --- 3. RAG HAFIZASINI (VEKTÖR VERİTABANI) OLUŞTURMA ---
# Bu fonksiyon, veri setindeki cevapları alır, parçalar, vektörlere dönüştürür
# ve FAISS veritabanına yükler. Lokal embedding modeli kullanılır.
@st.cache_resource # Hesaplanan sonucu (vector_db) önbelleğe al, tekrar hesaplama
def create_vector_db(_dataset):
    with st.spinner("RAG 'hafızası' (Lokal Model) oluşturuluyor... (Bu işlem ilk çalıştırmada 5-10 dakika sürebilir)"):

        # 1. Veriyi Hazırla: Sadece uzman cevaplarını ('Response') al
        documents = [Document(page_content=item['Response']) for item in _dataset if item['Response']] # Boş cevapları atla

        # 2. Metinleri Parçala (Chunking): Uzun metinleri yönetilebilir parçalara böl
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        if not chunks:
             st.error("Veri setinden geçerli metin parçası (chunk) bulunamadı. Lütfen veri setini kontrol edin.")
             st.stop()

        # 3. LOKAL Embedding Modeli'ni Yükle: Metinleri vektörlere dönüştür
        # Açık kaynaklı `all-MiniLM-L6-v2` modeli kullanılır, API gerektirmez.
        try:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
        except Exception as e:
            st.error(f"Lokal embedding modeli yüklenirken hata oluştu ({model_name}): {e}")
            st.stop()

        # 4. FAISS Vektör Veritabanı'nı Oluştur: Vektörleri hızlı arama için indeksle
        st.info(f"Toplam {len(chunks)} metin parçası (chunk) bulundu. Embedding işlemi başlıyor...")
        try:
            vector_db = FAISS.from_documents(chunks, embeddings)
            return vector_db
        except Exception as e:
            st.error(f"FAISS veritabanı oluşturulurken hata: {e}")
            st.stop()


# Hafıza fonksiyonunu çağır ve sonucu al
try:
    vector_db = create_vector_db(dataset)
    st.success("RAG 'hafızası' başarıyla oluşturuldu ve yüklendi!")
except Exception as e:
    # create_vector_db içindeki hatalar zaten st.stop() ile durdurur,
    # ama yine de genel bir hata yakalama ekleyelim.
    st.error(f"RAG hafızası oluşturma sürecinde genel bir hata oluştu: {e}")
    st.stop()


# --- 4. CHATBOT ARAYÜZÜ VE RAG CEVAP ÜRETİMİ ---

st.header("🧠 Chatbot ile Konuşun")

# Streamlit'in session_state özelliğini kullanarak sohbet geçmişini oturumlar arasında tut
if "messages" not in st.session_state:
    st.session_state.messages = []

# Geçmiş mesajları ekranda göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcıdan yeni bir soru al
if user_prompt := st.chat_input("Psikoloji ile ilgili sorunuzu buraya yazın..."):
    # Kullanıcının sorusunu geçmişe ekle ve ekranda göster
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Botun cevabını üretmek için RAG Pipeline'ını çalıştır
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # Cevap alanı için yer tutucu
        full_response = ""

        try:
            # 1. Retrieval: Kullanıcının sorusuna en benzer dokümanları FAISS'ten bul
            # 'k=5' -> en benzer 5 dokümanı getirir. Bu değer ayarlanabilir.
            retriever = vector_db.as_retriever(search_kwargs={"k": 5})

            # 2. Prompt Şablonu: LLM'e (Gemini) nasıl cevap vermesi gerektiğini söyleyen talimat
            template = """Aşağıdaki bağlamı kullanarak soruyu yanıtlamaya çalışın.
                        Eğer bağlamda yeterli bilgi yoksa, genel bilginizi kullanarak kısa bir cevap verin
                        ancak cevabın genel bilgiye dayandığını belirtin. Cevap uydurmaktan kaçının.

                        Bağlam: {context}

                        Soru: {question}

                        Yardımcı Cevap:"""
            prompt_template = ChatPromptTemplate.from_template(template)

            # 3. LLM (Generation Model): Cevabı üretecek olan model (Gemini Pro)
            # 'temperature' üretilen cevabın ne kadar "yaratıcı" olacağını kontrol eder (0 = en deterministik).
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.7)

            # 4. RAG Zinciri (LCEL - LangChain Expression Language):
            # Adım 1, 2 ve 3'ü birbirine bağlayan akış.
            rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()} # Soruyu al, bağlamı bul
                | prompt_template                                         # Talimatı doldur
                | llm                                                     # Cevabı üret
                | StrOutputParser()                                       # Cevabı metne çevir
            )

            # 5. Zinciri Çalıştır: Kullanıcının sorusunu zincire gönder ve cevabı al
            full_response = rag_chain.invoke(user_prompt)
            message_placeholder.markdown(full_response) # Cevabı ekrana yazdır

        except Exception as e:
            # Hata durumunda kullanıcıya bilgi ver
            st.error(f"Cevap üretilirken bir hata oluştu: {e}")
            full_response = "Üzgünüm, şu anda bir sorun yaşıyorum. Lütfen daha sonra tekrar deneyin."
            message_placeholder.markdown(full_response)

    # Botun cevabını (veya hata mesajını) sohbet geçmişine ekle
    st.session_state.messages.append({"role": "assistant", "content": full_response})