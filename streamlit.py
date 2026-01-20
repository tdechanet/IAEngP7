import streamlit as st
import requests

API_URL = "https://project-7-duagencrcrhsg9ca.francecentral-01.azurewebsites.net" 

st.set_page_config(page_title="Air Paradis", page_icon="✈️")

st.title("Air Paradis")

if 'prediction' not in st.session_state:
	st.session_state.prediction = None
if 'score' not in st.session_state:
	st.session_state.score = None
if 'last_tweet' not in st.session_state:
	st.session_state.last_tweet = None
if 'msg_success' not in st.session_state:
    st.session_state.msg_success = None

tweet_input = st.text_area("Saisissez un tweet :")

if st.button("Analyser"):
	st.session_state.msg_success = None
	if tweet_input.strip():
		with st.spinner("Analyse en cours..."):
			try:
				response = requests.post(f"{API_URL}/predict", json={"text": tweet_input})
				if response.status_code == 200:
					data = response.json()

					st.session_state.prediction = data["sentiment"]
					st.session_state.score = data["confiance"]
					st.session_state.last_tweet = tweet_input
				else:
					st.error(f"Erreur API : {response.status_code}")
			except Exception as e:
				st.error(f"Erreur de connexion : {e}")
	else:
		st.warning("Veuillez entrer du texte.")


if st.session_state.prediction:
	st.divider()
	st.subheader(f"Résultat : {st.session_state.prediction}")
	st.progress(st.session_state.score)
	st.write(f"Score de confiance : {st.session_state.score:.2%}")
	
	st.write("La prédiction est-elle correcte ?")
	col1, col2 = st.columns(2)
	
	if col1.button("Oui"):
		st.session_state.msg_success = "Merci pour votre retour !"
		st.session_state.prediction = None
		st.rerun()

	if col2.button("Non, signaler une erreur"):
		try:
			fb_resp = requests.post(
				f"{API_URL}/feedback", 
				json={
					"text": st.session_state.last_tweet, 
					"prediction": st.session_state.prediction
				}
			)
			if fb_resp.status_code == 200:
				st.session_state.msg_success = "Feedback envoyé à Azure Application Insights."
			else:
				st.session_state.msg_success = f"Erreur feedback : {fb_resp.status_code}"
		except Exception as e:
			st.session_state.msg_success = f"Erreur de connexion feedback : {e}"
		
		st.session_state.prediction = None 
		st.rerun()

if st.session_state.msg_success:
    st.divider()
    st.success(st.session_state.msg_success)