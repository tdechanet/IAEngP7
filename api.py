import numpy as np
from scipy.special import softmax
from transformers import BertTokenizer
import onnxruntime as rt
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager


ONNX_PATH = "models/bert_reduced_model.onnx"
TOKENIZER_PATH = "tokenizer_artifacts"
MAX_LENGTH = 64
LABEL_MAPPING = {0: "NEGATIF", 1: "POSITIF"}

session: rt.InferenceSession
tokenizer: BertTokenizer

class TextIn(BaseModel):
	text: str

class PredictionOut(BaseModel):
	sentiment: str
	confidence: float


@asynccontextmanager
async def lifespan(app: FastAPI):
	
	global session, tokenizer

	print("INFO: Chargement du tokenizer...")
	try:
		tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
		print("INFO: Tokenizer chargé")
	except Exception as e:
		print(f"ERREUR: Impossible de charger le tokenizer à partir de {TOKENIZER_PATH}. {e}")

	print("INFO: Chargement de la session ONNX Runtime...")
	try:
		session = rt.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
		print("INFO: Modèle ONNX chargé")
	except Exception as e:
		print(f"ERREUR: Impossible de charger le modèle ONNX à partir de {ONNX_PATH}. {e}")
	yield

app = FastAPI(title="Projet 7", lifespan=lifespan)
@app.post("/predict", response_model=PredictionOut)
def predict_sentiment(data: TextIn):

	if session is None or tokenizer is None:
		return {"sentiment": "ERREUR", "confiance": 0.0}

	text = data.text

	inputs = tokenizer.encode_plus(
		text,
		add_special_tokens=True,
		max_length=MAX_LENGTH,
		padding="max_length",
		truncation=True,
		return_tensors="np"
	)

	onnx_input = {
		"input_ids": inputs["input_ids"].astype(np.int64), #type:ignore
		"attention_mask": inputs["attention_mask"].astype(np.int64) #type:ignore
	}

	try:
		logits = session.run(["output"], onnx_input)[0][0] #type:ignore
	except RuntimeError as e:
		print(f"Erreur d'exécution ONNX: {e}")
		return {"sentiment": "ERREUR_RUNTIME", "confiance": 0.0}

	probabilities = softmax(logits)
	predicted_class_id = int(np.argmax(probabilities))
	confidence = probabilities[predicted_class_id]
	sentiment = LABEL_MAPPING[predicted_class_id]
	
	return PredictionOut(
		sentiment=sentiment,
		confidence=float(confidence)
	)