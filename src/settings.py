"""Настройки Call Transcriber из env."""

import os

from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CLEF_FLASH_MODEL = os.getenv("CLEF_FLASH_MODEL", "clef-flash")
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "5000"))
IDLE_UNLOAD_SEC = int(os.getenv("IDLE_UNLOAD_SEC", "300"))

ROLE_PROMPT = (
    "Ты — классификатор спикеров в звонке по страхованию. "
    "Определи ТОЛЬКО менеджера. Признаки менеджера: "
    "представляется, предлагает оформить полис/расчёт, называет компании (ВСК, "
    "Согласие, Альфа и т.п.), задаёт вопросы о потребностях, называет тарифы/цены, "
    "принимает решение, просит данные клиента. "
    "Верни метку спикера, который является менеджером."
)

MODEL_NAME = "v3_e2e_rnnt"
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
