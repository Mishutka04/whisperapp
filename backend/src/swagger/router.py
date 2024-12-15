

import shutil
from fastapi import APIRouter, File, UploadFile

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline



device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)




router = APIRouter(
    prefix='/operations',
    tags=['Operation']
)
from pydantic import BaseModel

class Operation(BaseModel):
    id: int
    quantity: str
    figi: str
    instrument_type: str
    date: str  # Adjust type as needed
    type: str

    class Config:
        orm_mode = True


@router.post("/file_to_text")
def get_file(file: UploadFile = File(...)):
    # Сохранение загруженного файла
    file_location = file.filename
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # Обработка файла (например, передаем имя файла в pipe)
    result = pipe(file_location)

    return result["text"]
