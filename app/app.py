# Pkgs
import streamlit as st 
import altair as alt
import plotly.express as px 
import pandas as pd 
import numpy as np
import requests
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Text Emotion Analyser",layout='wide',initial_sidebar_state="expanded")

def load_lottieURL(url):
    r = requests.get(url)
    if r.status_code != 200: # not succesful
        return None
    else: return r.json()
        
lottie_animation=load_lottieURL("https://assets10.lottiefiles.com/packages/lf20_RXoWL0oIUr.json")

#load model from pkl file
import joblib 
emo_model = joblib.load(open("model\emotion_analyzer.pkl","rb"))

def predict_emotions(docx):
	results = emo_model.predict([docx])
	return results[0]

def get_prediction_proba(docx):
	results = emo_model.predict_proba([docx])
	return results

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}


# Main Application
def main():
	st.title("Text Emotion Analyser")
	nav = st.sidebar.radio("**Navigation**",["Home","About"],index=0)
	
	
	if nav == "Home":
		
		st_lottie(lottie_animation,height=500,key="mood")
		
		with st.form(key='emotion_clf_form'):
				raw_text = st.text_area("Type Here")
				submit_text = st.form_submit_button(label='Submit')

		if submit_text:
			col1,col2  = st.columns(2)

			# Apply Fxn Here
			prediction = predict_emotions(raw_text)
			probability = get_prediction_proba(raw_text)
			
			with col1:
				st.success("Original Text")
				st.write(raw_text)

				st.success("Prediction")
				emoji_icon = emotions_emoji_dict[prediction]
				st.write("{}:{}".format(prediction,emoji_icon))
				st.write("Confidence:{}".format(np.max(probability)))

			with col2:
				st.success("Prediction Probability")
				# st.write(probability)
				proba_df = pd.DataFrame(probability,columns=emo_model.classes_)
				# st.write(proba_df.T)
				proba_df_clean = proba_df.T.reset_index()
				proba_df_clean.columns = ["emotions","probability"]

				fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
				st.altair_chart(fig,use_container_width=True)
	else:
		st.subheader("About")
		st.text("This project is created by Shivansh Tiwari")
		st.subheader("Linkedin")
		st.text("https://www.linkedin.com/in/shivansh-tiwari-82790a231")
 	
if __name__ == '__main__':
	main()