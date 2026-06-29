"""
=============================================================================
SCRIPT INGESTION RESEP MASAKAN (ingest.py)
=============================================================================
Fungsi Utama Script Ini:
1. Membaca dataset resep masakan Indonesia dari file CSV (data-sample/Indonesian_Food_Recipes.csv).
2. Mengonversi data resep menjadi objek Dokumen terstruktur (Judul + Bahan Makanan).
3. Membersihkan teks resep, memvalidasi kualitas data, dan memperkaya metadata (panjang karakter & jumlah kata).
4. Menghubungkan aplikasi ke ChromaDB Cloud dan mengunggah (ingest) dokumen vektor secara ber-batch.
=============================================================================
"""

import os
import sys
import csv
import re
import argparse
import chromadb
from typing import List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

CSV_PATH = os.path.join("data-sample", "Indonesian_Food_Recipes.csv")
COLLECTION_NAME = "recipes"


@dataclass
class Document:
    id: str
    text: str
    metadata: Dict[str, Any]
    source: str

    def __repr__(self):
        return f"Document(id={self.id}, title='{self.metadata.get('title', '')}', length={len(self.text)})"


class RecipeDocumentLoader:
    @staticmethod
    def load_from_csv(filepath: str, limit: int = None) -> List[Document]:
        if not os.path.exists(filepath):
            print(f"❌ Error: File dataset tidak ditemukan di '{filepath}'.")
            return []

        documents = []
        print(f"📖 Membaca dataset resep dari {filepath}...")

        with open(filepath, mode="r", encoding="utf-8-sig", errors="replace") as file:
            reader = csv.DictReader(file)
            for index, row in enumerate(reader):
                if limit and len(documents) >= limit:
                    break

                title = row.get("Title", "").strip()
                ingredients = row.get("Ingredients", "").strip()
                steps = row.get("Steps", "").strip()
                category = row.get("Category", "").strip()
                url = row.get("URL", "").strip()
                loves = row.get("Loves", "0").strip()

                if not title:
                    continue

                doc_text = f"Judul Resep: {title}\nBahan-bahan: {ingredients}"
                
                doc = Document(
                    id=f"recipe_{index+1}",
                    text=doc_text,
                    metadata={
                        "title": title[:500],
                        "category": category[:200] if category else "Umum",
                        "url": url[:500] if url else "",
                        "loves": int(loves) if loves.isdigit() else 0,
                        "steps": steps[:2000] if steps else ""
                    },
                    source=filepath
                )
                documents.append(doc)

        print(f"✅ Berhasil memuat {len(documents)} dokumen resep dari CSV.")
        return documents


class RecipeDocumentPreprocessor:
    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = text.replace('\r\n', '\n')
        return text.strip()

    @staticmethod
    def validate_document(doc: Document) -> bool:
        if not doc.text or len(doc.text.strip()) < 10:
            return False
        if '\ufffd' in doc.text:
            return False
        return True

    def process(self, documents: List[Document]) -> List[Document]:
        processed_docs = []
        for doc in documents:
            if not self.validate_document(doc):
                continue

            cleaned_text = self.clean_text(doc.text)
            enriched_metadata = {
                **doc.metadata,
                "char_count": len(cleaned_text),
                "word_count": len(cleaned_text.split())
            }

            processed_doc = Document(
                id=doc.id,
                text=cleaned_text,
                metadata=enriched_metadata,
                source=doc.source
            )
            processed_docs.append(processed_doc)

        print(f"✅ Selesai pra-pemrosesan {len(processed_docs)} dokumen resep.")
        return processed_docs


class ChromaVectorIndexer:
    def __init__(self, collection_name: str = COLLECTION_NAME):
        self.collection_name = collection_name
        self.client = self._get_chroma_client()

    def _get_chroma_client(self):
        api_key = os.getenv("CHROMADB_API_KEY")
        tenant = os.getenv("CHROMADB_TENANT", "7a558a95-2c85-42be-af99-a90a9019afb9")
        database = os.getenv("CHROMADB_DATABASE", "forRAG")

        if not api_key or api_key == "YOUR_API_KEY":
            raise ValueError("❌ CHROMADB_API_KEY belum diatur di file .env")

        print(f"🔄 Menghubungkan ke ChromaDB Cloud (Tenant: {tenant}, Database: {database})...")
        return chromadb.CloudClient(
            api_key=api_key,
            tenant=tenant,
            database=database
        )

    def index_documents(self, documents: List[Document], batch_size: int = 100):
        collection = self.client.get_or_create_collection(name=self.collection_name)
        total_added = 0
        
        batch_docs = []
        batch_meta = []
        batch_ids = []

        for doc in documents:
            batch_docs.append(doc.text)
            batch_meta.append(doc.metadata)
            batch_ids.append(doc.id)

            if len(batch_docs) >= batch_size:
                collection.add(documents=batch_docs, metadatas=batch_meta, ids=batch_ids)
                total_added += len(batch_docs)
                print(f"🚀 Indexed {total_added}/{len(documents)} dokumen ke ChromaDB...")
                batch_docs, batch_meta, batch_ids = [], [], []

        if batch_docs:
            collection.add(documents=batch_docs, metadatas=batch_meta, ids=batch_ids)
            total_added += len(batch_docs)
            print(f"🚀 Indexed {total_added}/{len(documents)} dokumen ke ChromaDB...")

        print(f"\n✨ Ingestion selesai! Total {total_added} dokumen tersimpan di koleksi '{self.collection_name}'.")


def print_document_stats(documents: List[Document]):
    print("\n" + "="*60)
    print("📊 STATISTIK KOLEKSI DOKUMEN RESEP")
    print("="*60)
    total_chars = sum(len(doc.text) for doc in documents)
    print(f"Total Dokumen    : {len(documents)}")
    print(f"Total Karakter   : {total_chars:,}")
    print(f"Rata-rata Panjang: {total_chars / len(documents):,.0f} karakter per dokumen")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Ingestion Dokumentasi Resep RAG")
    parser.add_argument("--limit", type=int, default=None, help="Maksimal dokumen yang diproses (opsional)")
    parser.add_argument("--batch-size", type=int, default=100, help="Ukuran batch per upload (default: 100)")
    args = parser.parse_args()

    print("="*60)
    print("🍳 RAG DOCUMENT INGESTION PIPELINE")
    print("="*60)

    raw_docs = RecipeDocumentLoader.load_from_csv(CSV_PATH, limit=args.limit)
    if not raw_docs:
        sys.exit(1)

    preprocessor = RecipeDocumentPreprocessor()
    processed_docs = preprocessor.process(raw_docs)

    print_document_stats(processed_docs)

    indexer = ChromaVectorIndexer()
    indexer.index_documents(processed_docs, batch_size=args.batch_size)
