# --- [WinError 1114] DLL HATASI ÇÖZÜMÜ ---
# Bu komut, torch veya datasets import edilmeden EN ÖNCE çalışmalıdır.
# Sisteme "Çakışan DLL'leri (libiomp5md.dll) görmezden gel" der.
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# --- ÇÖZÜM SONU ---

import streamlit as st
import google.generativeai as genai
from datasets import load_dataset  # Hugging Face veri setlerini yüklemek için
from dotenv import load_dotenv # API anahtarını .env dosyasından okumak için

# Lokal Embedding'e (Rota 2) geri döndük
from langchain_community.embeddings import HuggingFaceEmbeddings #
from langchain_community.vectorstores import FAISS #
from langchain_text_splitters import RecursiveCharacterTextSplitter #
from langchain_core.documents import Document #
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# --- 1. PROJE KURULUMU VE API ANAHTARI YÜKLEME ---

# .env dosyasını tam yolunu bularak güvenle yükle
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(script_dir, ".env")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
    else:
        print(f"HATA: .env dosyası şu yolda bulunamadı: {dotenv_path}")
except Exception as e:
    st.error(f".env dosyası yüklenirken beklenmedik hata: {e}")
    st.stop()

# Google API anahtarını al (SADECE "generation" için, embedding için değil)
try:
    api_key = os.environ["GOOGLE_API_KEY"]
    if not api_key:
        raise ValueError("GOOGLE_API_KEY .env dosyasında bulunamadı veya boş.")
    genai.configure(api_key=api_key)
    print("API Anahtarı başarıyla yüklendi.") 
except (KeyError, ValueError) as e:
    st.error(f"HATA: GOOGLE_API_KEY ortam değişkeni ayarlanmamış. Lütfen .env dosyanızı kontrol edin. Hata: {e}")
    st.stop() 

# Streamlit arayüzünün başlığını ayarla
st.set_page_config(page_title="Psikoloji Bilgi Asistanı", page_icon="🧠")
st.title("🧠 Psikoloji Bilgi Asistanı")
st.caption("Bu chatbot, RAG mimarisi kullanılarak geliştirilmiştir ve yalnızca yüklenen veri setindeki bilgilere dayanarak yanıt verir.")


# --- 2. VERİ SETİNİ YÜKLEME (ADIM 2) ---
try:
    with st.spinner("Psikoloji 'kitaplığı' (veri seti) yükleniyor..."):
        dataset_dict = load_dataset("Amod/mental_health_counseling_conversations")
        split_name = list(dataset_dict.keys())[0]
        dataset = dataset_dict[split_name]

    st.success(f"Veri seti başarıyla yüklendi! Toplam {len(dataset)} adet konuşma bulundu.")

    # Verinin nasıl göründüğünü kontrol etmek için bir örnek göster
    st.subheader("Veri Setinden Bir Örnek:")
    st.text_area("Örnek Soru (Context)", dataset[0]['Context'], height=100)
    st.text_area("Örnek Cevap (Response)", dataset[0]['Response'], height=200)

except Exception as e:
    st.error(f"Veri seti yüklenirken bir hata oluştu: {e}")
    # Bu, DLL hatası ise burada görünecek
    st.stop()

# --- 3. VERİYİ HAFIZAYA (VEKTÖR VERİTABANI) YÜKLEME (LOKAL MODEL) ---

@st.cache_resource
def create_vector_db(_dataset): 
    with st.spinner("RAG 'hafızası' (Lokal Model) oluşturuluyor... (Bu işlem 5-10 dakika sürebilir)"):

        # 1. Veriyi Hazırla
        documents = []
        for item in _dataset: 
            doc = Document(page_content=item['Response'])
            documents.append(doc)

        # 2. Metinleri Parçala
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        # 3. LOKAL Embedding Modeli'ni Yükle (API YOK)
        # Bu model, API'ye gitmek yerine kendi bilgisayarında çalışır.
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

        # 4. FAISS Vektör Veritabanı'nı Oluştur
        st.info(f"Toplam {len(chunks)} metin parçası (chunk) bulundu. Embedding işlemi başlıyor...")
        vector_db = FAISS.from_documents(chunks, embeddings)
        return vector_db

# Hafıza fonksiyonunu çağır
try:
    vector_db = create_vector_db(dataset)
    st.success("RAG 'hafızası' başarıyla oluşturuldu ve yüklendi!")
except Exception as e:
    st.error(f"RAG hafızası oluşturulurken bir hata oluştu: {e}")
    st.stop()




# --- 4. CHATBOT ARAYÜZÜ VE RAG PIPELINE ---

st.header("🧠 Chatbot ile Konuşun")

# Streamlit'in session_state özelliğini kullanarak sohbet geçmişini tut
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sohbet geçmişini ekranda göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcıdan girdi al
if prompt := st.chat_input("Psikoloji ile ilgili sorunuzu buraya yazın..."):
    # Kullanıcının mesajını sohbet geçmişine ekle ve ekranda göster
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG Pipeline'ını Çalıştır
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # 1. Retrieval (FAISS Hafızasından Bilgi Getirme)
            # vector_db'yi bir retriever objesine dönüştür (en benzer 3 cevabı getirsin)
            retriever = vector_db.as_retriever(search_kwargs={"k": 5})
            
            # 2. Prompt Tasarımı (Gemini'ye Nasıl Soracağımız)
            # Hafızadan gelen bilgileri ({context}) ve kullanıcının sorusunu ({question}) alacak bir şablon
            template = """Aşağıdaki bağlamı kullanarak soruyu yanıtlamaya çalışın. 
                Eğer bağlamda yeterli bilgi yoksa, genel bilginizi kullanarak kısa bir cevap verin 
                ancak cevabın genel bilgiye dayandığını belirtin. Cevap uydurmaktan kaçının.

                Bağlam: {context}

                Soru: {question}

                Yardımcı Cevap:"""
            prompt_template = ChatPromptTemplate.from_template(template)

            # 3. Generation (Gemini Modeli ile Cevap Üretme)
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.7)

            # 4. LangChain Expression Language (LCEL) ile RAG Pipeline'ını Oluşturma
            # Bu zincir:
            #   - Kullanıcının sorusunu alır ({question})
            #   - Retriever ile ilgili belgeleri bulur ve {context}'e atar
            #   - Prompt şablonunu doldurur
            #   - LLM (Gemini) ile cevap üretir
            #   - Cevabı metin olarak formatlar (StrOutputParser)
            rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()} 
                | prompt_template 
                | llm 
                | StrOutputParser() 
            )

            # Pipeline'ı çalıştır ve cevabı al (Streamlit'in streaming'i ile uyumlu değil, o yüzden direkt invoke)
            full_response = rag_chain.invoke(prompt)
            message_placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"Cevap üretilirken bir hata oluştu: {e}")
            full_response = "Üzgünüm, bir hata oluştu."
            message_placeholder.markdown(full_response)

    # Botun cevabını sohbet geçmişine ekle
    st.session_state.messages.append({"role": "assistant", "content": full_response})