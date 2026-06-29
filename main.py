"""
=============================================================================
BOT ASISTEN RESEP MASAKAN (INTERACTIVE RAG INTERFACE)
=============================================================================
Deskripsi:
Script ini merupakan antarmuka (interface) aplikasi Bot Asisten Resep Masakan.
Menggunakan arsitektur RAG (Retrieval-Augmented Generation) berbasis 
Vector Database ChromaDB Cloud untuk memberikan rekomendasi resep masakan 
yang paling relevan berdasarkan input bahan atau nama makanan dari pengguna.

Alur Kerja:
1. Menghubungkan ke ChromaDB Cloud dan membuka koleksi 'recipes'.
2. Menerima input pertanyaan/kueri dari pengguna di terminal secara interaktif.
3. Melakukan pencarian kesamaan vektor (semantic similarity search) di ChromaDB.
4. Menampilkan hasil resep terbaik beserta detail bahan dan langkah memasaknya.
=============================================================================
"""

import os
import chromadb
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 1. INISIALISASI & KONFIGURASI LINGKUNGAN
# ---------------------------------------------------------------------------
# Memuat variabel lingkungan dari file .env
load_dotenv()

# Nama koleksi vector tempat resep tersimpan di ChromaDB Cloud
COLLECTION_NAME = "recipes"


# ---------------------------------------------------------------------------
# 2. FUNGSI KONEKSI DATABASE
# ---------------------------------------------------------------------------
def get_chroma_client():
    """
    Membuat dan mengembalikan koneksi klien ke ChromaDB Cloud.
    
    Mengambil variabel lingkungan berikut:
    - CHROMADB_API_KEY : Kunci akses otentikasi API ChromaDB Cloud.
    - CHROMADB_TENANT  : Identifier tenant/organisasi pengguna.
    - CHROMADB_DATABASE: Nama database ruang kerja di cloud.
    """
    api_key = os.getenv("CHROMADB_API_KEY")
    tenant = os.getenv("CHROMADB_TENANT", "7a558a95-2c85-42be-af99-a90a9019afb9")
    database = os.getenv("CHROMADB_DATABASE", "forRAG")

    if not api_key:
        raise ValueError("❌ Error: CHROMADB_API_KEY belum diatur di file .env")

    return chromadb.CloudClient(
        api_key=api_key,
        tenant=tenant,
        database=database
    )


# ---------------------------------------------------------------------------
# 3. FUNGSI PENCARIAN REKOMENDASI RESEP (VECTOR QUERY)
# ---------------------------------------------------------------------------
def search_recipes(collection, query, n_results=3):
    """
    Melakukan pencarian vektor (semantic vector query) pada koleksi ChromaDB.
    
    Parameter:
    - collection : Instance koleksi ChromaDB yang sedang aktif.
    - query (str): Teks masukan dari pengguna (nama masakan / daftar bahan).
    - n_results (int): Jumlah hasil resep terbaik yang ingin dikembalikan.
    
    Return:
    - Dict berisi dokumen, metadata, dan nilai jarak (distances) dari hasil pencarian.
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results


# ---------------------------------------------------------------------------
# 4. FUNGSI FORMATTING HASIL REKAMAN RESEP
# ---------------------------------------------------------------------------
def format_recipe_output(index, doc, metadata):
    """
    Merapikan dan mencetak detail informasi resep ke terminal.
    
    Parameter:
    - index (int)   : Nomor urut hasil rekomendasi.
    - doc (str)     : Teks dokumen asal yang di-index di ChromaDB.
    - metadata (dict): Dictionary berisi informasi tambahan resep (judul, kategori, dll).
    """
    title = metadata.get("title", "Tanpa Judul")
    category = metadata.get("category", "Umum")
    loves = metadata.get("loves", 0)
    steps = metadata.get("steps", "Tidak ada langkah-langkah.")
    url = metadata.get("url", "")

    # Mengekstrak daftar bahan makanan dari teks dokumen pendukung
    ingredients = ""
    if "Bahan-bahan:" in doc:
        ingredients = doc.split("Bahan-bahan:")[1].strip()

    # Cetak hasil dengan format yang rapi dan menarik
    print(f"\n--- Resep #{index} ---")
    print(f"📌 Judul     : {title}")
    print(f"🏷️  Kategori  : {category}")
    print(f"❤️  Suka      : {loves}")
    if url:
        print(f"🔗 URL       : {url}")
    print(f"🛒 Bahan     : {ingredients}")
    print(f"📝 Langkah   :\n{steps}")


# ---------------------------------------------------------------------------
# 5. FUNGSI UTAMA & LOOP INTERAKTIF TERMINAL
# ---------------------------------------------------------------------------
def main():
    print("==========================================")
    print("🍳  BOT ASISTEN RESEP MASAKAN (RAG)     🍳")
    print("==========================================")
    print("🔄 Menghubungkan ke ChromaDB Cloud...")
    
    # Inisialisasi koneksi ke database saat aplikasi dimulai
    try:
        client = get_chroma_client()
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        total_docs = collection.count()
        print(f"✅ Terhubung! Total resep terindeks saat ini: {total_docs}\n")
    except Exception as e:
        print(f"❌ Gagal terhubung ke ChromaDB Cloud: {e}")
        return

    print("💡 Petunjuk: Ketik nama masakan atau bahan yang Anda miliki (contoh: 'ayam woku' atau 'telur santan').")
    print("💡 Ketik 'keluar' atau 'exit' untuk mengakhiri program.\n")

    # Loop utama untuk mendengarkan kueri pengguna secara terus-menerus
    while True:
        try:
            # Menerima input dari keyboard pengguna
            user_input = input("🔍 Cari resep > ").strip()
            
            # Jika input kosong, ulangi loop
            if not user_input:
                continue
                
            # Cek perintah keluar dari aplikasi
            if user_input.lower() in ["exit", "keluar", "q"]:
                print("👋 Terima kasih telah menggunakan Bot Asisten Resep Masakan! Sampai jumpa.")
                break

            print(f"🔎 Mencari resep terbaik untuk: '{user_input}'...")
            
            # Panggil fungsi pencarian RAG (mengembalikan 3 resep teratas)
            results = search_recipes(collection, user_input, n_results=3)

            # Ekstrak daftar dokumen dan metadatas dari hasil respons ChromaDB
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]

            # Tangani kondisi jika tidak ada resep yang ditemukan
            if not documents:
                print("❌ Tidak ditemukan resep yang sesuai.")
                continue

            # Tampilkan setiap hasil resep yang ditemukan
            for idx, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
                format_recipe_output(idx, doc, meta)

            print("\n" + "="*45 + "\n")

        except KeyboardInterrupt:
            # Penanganan penghentian paksa dengan Ctrl+C
            print("\n👋 Program dihentikan.")
            break
        except Exception as e:
            print(f"❌ Terjadi kesalahan saat mencari resep: {e}")


# Entry point pembuka ketika file ini dijalankan langsung oleh interpreter Python
if __name__ == "__main__":
    main()
