import os
import sqlite3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# Ambil API Key dari Environment Variable (aman untuk GitHub)
API_KEY = os.environ.get("GEMINI_API_KEY", "")
DB_FILE = "apotek.db"

app = FastAPI(title="Pharmacy AI Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(timeout=60000)
)

# Otomatis inisialisasi tabel & data awal saat server cloud menyala
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS master_obat (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kode_obat TEXT UNIQUE,
            nama_obat TEXT,
            sediaan_mg REAL,
            stok INTEGER,
            lokasi_rak TEXT,
            harga_satuan INTEGER
        )
    """)
    data_obat = [
        ("OBT001", "Paracetamol 500mg", 500, 100, "A1", 500),
        ("OBT002", "CTM 4mg", 4, 50, "B2", 200),
        ("OBT003", "Amoxicillin 500mg", 500, 0, "C1", 1200),
        ("OBT004", "Dexamethasone 0.5mg", 0.5, 80, "B1", 300),
        ("OBT005", "Ambroxol 30mg", 30, 45, "A3", 600),
    ]
    cur.executemany("""
        INSERT OR IGNORE INTO master_obat (kode_obat, nama_obat, sediaan_mg, stok, lokasi_rak, harga_satuan)
        VALUES (?, ?, ?, ?, ?, ?)
    """, data_obat)
    conn.commit()
    conn.close()

init_db()

class ResepRequest(BaseModel):
    teks_resep: str = Field(..., example="R/ Paracetamol 250 mg\n   CTM 2 mg\n   m.f. pulv. dtd No. X\nR/ Amoxicillin 500 mg No. XV")

class ItemObatAnalisis(BaseModel):
    nama_obat: str
    jenis: str
    jumlah_diambil: float
    satuan: str
    lokasi_rak: str
    status_stok: str
    catatan: str

class HasilAnalisisResep(BaseModel):
    ringkasan_klinis: str
    daftar_obat: list[ItemObatAnalisis]
    langkah_kerja_apoteker: list[str]
    perlu_konfirmasi_dokter: bool

def cek_stok_obat(nama_obat: str) -> list[dict]:
    """Mencari ketersediaan stok, dosis sediaan, harga, dan lokasi rak obat."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "SELECT kode_obat, nama_obat, sediaan_mg, stok, lokasi_rak, harga_satuan FROM master_obat WHERE nama_obat LIKE ?",
        (f"%{nama_obat}%",)
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows] if rows else [{"info": f"Obat '{nama_obat}' tidak ditemukan"}]

def hitung_tablet_racikan(dosis_diminta_mg: float, sediaan_tablet_mg: float, jumlah_bungkus: int) -> float:
    """Menghitung jumlah tablet utuh yang harus diambil untuk membuat puyer."""
    return (dosis_diminta_mg * jumlah_bungkus) / sediaan_tablet_mg

@app.get("/")
def health_check():
    return {"status": "online", "service": "Pharmacy AI Agent"}

@app.post("/api/resep/analisis", response_model=HasilAnalisisResep)
def analisis_resep(payload: ResepRequest):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY belum diatur di server.")

    daftar_model = [
        "gemini-3.5-flash",
        "gemini-3.7-flash",
    ]
    error_terakhir = None

    for nama_model in daftar_model:
        try:
            chat = client.chats.create(
                model=nama_model,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "Anda adalah Asisten AI Farmasi. Wajib panggil tools untuk cek stok database "
                        "dan hitung dosis racikan sebelum menyusun hasil akhir terstruktur."
                    ),
                    tools=[cek_stok_obat, hitung_tablet_racikan],
                    response_mime_type="application/json",
                    response_schema=HasilAnalisisResep,
                    temperature=0.1,
                )
            )
            response = chat.send_message(
                f"Proses resep berikut, cek stok serta raknya, dan hitung kebutuhan tablet:\n{payload.teks_resep}"
            )
            return response.parsed
        except Exception as e:
            error_msg = str(e)
            if "503" in error_msg or "504" in error_msg or "UNAVAILABLE" in error_msg:
                error_terakhir = e
                continue
            raise HTTPException(status_code=500, detail=error_msg)

    raise HTTPException(status_code=503, detail=f"Semua model sibuk: {str(error_terakhir)}")