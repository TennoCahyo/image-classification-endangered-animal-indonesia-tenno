import os
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.responses import RedirectResponse

from inference import run_inference
from inference import ConvNet 

MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_checkpoint.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(
    title="Endangered Animal Classification",
    description="API untuk klasifikasi 5 hewan endemik menggunakan ConvNet",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


_model: ConvNet | None = None

def get_model() -> ConvNet:
    global _model
    if _model is None:
        print(f"Loading model dari {MODEL_PATH} ke {DEVICE}...")
        model = ConvNet(num_classes=5)
        state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
        _model = model
        print("Model berhasil dimuat!")
    return _model

# Load model secara otomatis saat server FastAPI baru dinyalakan
@app.on_event("startup")
async def startup_event():
    get_model()

@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(DEVICE), "model_loaded": _model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="File kosong")

    model = get_model()


    try:
        result = run_inference(model, image_bytes, DEVICE)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inferensi gagal: {exc}") from exc

    return JSONResponse(result)