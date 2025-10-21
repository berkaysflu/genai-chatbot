# Psikoloji Bilgi Asistanı Chatbot

Bu proje, Akbank GenAI Bootcamp kapsamında geliştirilmiş RAG (Retrieval Augmented Generation) temelli bir sohbet robotudur.

## 1. Projenin Amacı

Bu chatbot'un amacı, kullanıcılardan gelen psikoloji ile ilgili soruları, güvenilir bir bilgi kaynağına dayanarak yanıtlamaktır. RAG mimarisi sayesinde, botun cevapları önceden yüklenmiş bir veri setindeki bilgilerle zenginleştirilir. Eğer veri setinde yeterli bilgi bulunamazsa, bot genel bilgisini kullanarak bir cevap üretebilir ve bu durumu kullanıcıya belirtir.

## 2. Veri Seti

Bu projede, Hugging Face platformunda bulunan [`Amod/mental_health_counseling_conversations`](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations) veri seti kullanılmıştır. Bu veri seti, çeşitli online danışmanlık platformlarından toplanmış, anonimleştirilmiş soru (Context) ve uzman psikolog cevaplarını (Response) içermektedir. RAG sistemimizin "hafızası", bu veri setindeki uzman cevapları üzerine kurulmuştur.

## 3. Kullanılan Yöntemler (Çözüm Mimarisi)

Projenin çözüm mimarisi RAG üzerine kurulmuştur ve aşağıdaki teknolojileri kullanır:

* **RAG Pipeline Framework:** LangChain - RAG akışını (Retrieval, Prompting, Generation) yönetmek için kullanılmıştır.
* **Embedding Model (Lokal):** `sentence-transformers/all-MiniLM-L6-v2` (HuggingFaceEmbeddings aracılığıyla) - Veri setindeki metinleri ve kullanıcı sorularını sayısal vektörlere dönüştürmek için kullanılmıştır. Bu model lokalde çalıştığı için API kotası sorunu yaşanmamıştır.
* **Vektör Veritabanı:** FAISS (CPU) - Metin vektörlerini verimli bir şekilde saklamak ve benzerlik araması yapmak için kullanılmıştır. `@st.cache_resource` ile önbelleğe alınarak performans artırılmıştır.
* **Generation Model (Beyin):** Google Gemini API (`gemini-2.5-pro` modeli) - LangChain aracılığıyla çağrılarak, bulunan bağlam ve kullanıcı sorusuna göre nihai cevabı üretmek için kullanılmıştır.
* **Web Arayüzü:** Streamlit - Kullanıcı ile etkileşim kurmak için basit ve hızlı bir web arayüzü oluşturmak amacıyla kullanılmıştır.
* **Diğer Kütüphaneler:** `python-dotenv` (API anahtarlarını güvenli yönetmek için), `datasets` (Hugging Face veri setini yüklemek için), `torch` (Lokal embedding modeli için gereklidir).

## 4. Çalıştırma Kılavuzu

Bu projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1.  **Depoyu Klonlayın:**
    ```bash
    git clone [https://github.com/berkaysflu/genai-chatbot.git](https://github.com/berkaysflu/genai-chatbot.git)
    cd genai-chatbot
    ```

2.  **Sanal Ortamı Kurun ve Aktifleştirin:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # Windows için
    # source venv/bin/activate  # macOS/Linux için
    ```

3.  **Gerekli Kütüphaneleri Kurun:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Bu komut, `requirements.txt` dosyasındaki `--index-url` yönergesi sayesinde PyTorch'un doğru CPU versiyonunu kuracaktır)*

4.  **API Anahtarını Ayarlayın:**
    * Proje ana dizininde `.env` adında bir dosya oluşturun.
    * İçine `GOOGLE_API_KEY="AIza...SİZİN_ANAHTARINIZ"` satırını ekleyerek kendi Google Gemini API anahtarınızı girin. (Not: Bu proje lokal embedding kullandığı için Cohere API anahtarı gerekmez).

5.  **Uygulamayı Çalıştırın:**
    ```bash
    streamlit run app.py
    ```
    *(İlk çalıştırmada RAG hafızasının oluşturulması 5-10 dakika sürebilir)*

## 5. Elde Edilen Sonuçlar ve Web Arayüzü

Chatbot, yüklenen veri setindeki bilgilere dayanarak psikoloji ile ilgili sorulara cevap verebilmektedir. Eğer sorunun cevabı veri setinde bulunamazsa, genel bilgisini kullanarak bir cevap üretebilir ve bu durumu belirtir. Arayüz, kullanıcıların soru sormasına ve botun cevaplarını görmesine olanak tanır.

**Not:** Cevap üretimi, kullanılan Google Gemini API'sinin yoğunluğuna ve RAG hafızasından getirilen bilgilerin miktarına bağlı olarak birkaç saniye sürebilir.

---
**(OPSİYONEL) EKRAN GÖRÜNTÜLERİ ALANI:**


[Chatbot Çalışma Örnekleri Albümü](https://imgur.com/a/Onk21Kz)

---

### Web Arayüzü Linki

(Uygulama yayına alındıktan sonra (deploy) link buraya eklenecektir.)