# Import Libraries
import os 
import joblib 
import librosa 
import librosa.display 
import numpy as np
import streamlit as st 
import matplotlib.pyplot as plt 
from streamlit_option_menu import option_menu 

# -------------------- PAGE CONFIG -------------------------------------
st.set_page_config(page_title='Grammer Scoring Engine', page_icon="🗣️", layout='wide')

# -------------------- LOAD MODEL ---------------------------------

MODEL_PATH = 'grammar_scoring_model.pkl'
model = joblib.load(MODEL_PATH)

# ======================= FEATURE EXTRACTOR =================================
def extract_features(file_path, sr=22050, n_mfcc=13):
    return None, None, None, None, None, None

# ---------------------- MENU BAR ----------------------------------
with st.container():

    st.write('This is inside container')

    selected=option_menu(
        menu_title=None,
        options=['About', 'Upload & Score','Visual Insights', 'Batch Scoring'],
        icons=['info-circle', 'mic', 'bar-chart-line', 'folder'],
        orientation='horizontal',
        default_index=0,
        styles={
            'container': {'padding': '0!important'}, 
            'icon' : {'color':'#9A341', 'font-size':'16px'},
            'nav-link': {'font-size': '16px', 'text-align':'center', 'margin':'5px', '--hover-color':'#FFA07A'},
            'nav-link-selected': {'background-color': '#2c7be5'}
        }
    )

# ----------------------------- ABOUT --------------------------------

if selected == 'About':
    st.title('📘 Introduction to the Grammer Scoring Engine of Voice')
    st.markdown('''
        This Grammer Scoring uses:
        - 🎵 **Audio Features** (MFCCs, Spectral Contrast, ZCR)
        - 🧠 **XgBoost Regressor**
        - 🎯 Outputs a score from **0 to 5** based on grammer quality, fluency and language level.
        
        ### 🛠 Applications:
        - Fluency assessment
        - Language learning 
        - Interview evaluation
            
        ---
        💬 Built with [Librosa](https://librosa.org), [Scikit-learn](https://scikit-learn.org), and [Streamlit](https://streamlit.io)
''')
    
# ============================ UPLOAD AND SCORE =========================

elif selected == 'Upload & Score':
    st.title('🎤 Upload Your Audio')
    uploaded_file = st.file_uploader('📂 Upload a `.wav` file', type=["wav"])

    if uploaded_file is not None:
        temp_file_path = 'temp_audio.wav'
        with open(temp_file_path, 'wb') as f:
            f.write(uploaded_file.read())

        st.audio(uploaded_file, format='audio/wav') 
        st.info('🎧 Audio loaded. Ready to analyze!')
        features, duration, zcr, y, sr, mfccs = extract_features(temp_file_path)

        if features is not None:
            prediction = model.predict(features.reshape(1, -1))[0]
            score = round(prediction, 2) 
            st.metric('⭐ Predicted Grammer Score', f"{score}/5.0") 

            # Save to session state for Visual Insights 
            st.session_state.y = y 
            st.session_state.sr= sr 
            st.session_state.mfccs = mfccs 

            with st.expander('📘 Grammar Feedback'):
                if score<2:
                    st.error("Beginner (A1): Focus on basic sentence structure.")
                elif score<4:
                    st.warning('Intermediate: Work on fluency and complex grammer.')
                else:
                    st.success('Advanced: Great grammetical usage!')

            with st.expander('🎛 Audio Features'):
                st.write(f'• **Duration:** {round(duration, 2)} sec')
                st.write(f'• **Zero Crossing Rate:** {round(zcr, 4)}')
        
        os.remove(temp_file_path)

# ============================ VISUAL INSIGHTS =============================

elif selected == 'Visual Insights':
    st.title('📊 Visual Insights')

    if "y" in st.session_state and st.session_state.y is not None:
        y = st.session_state.y 
        sr= st.session_state.sr 
        mfccs=st.session_state.mfccs 

        st.subheader('🔉 Waveform') 
        fig, ax = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title('Audio Waveform') 
        st.pyplot(fig) 

        st.subheader('🎼 MFCC Heatmap')
        fig2, ax2=plt.subplots()
        librosa.display.specshow(mfccs, x_axis='time', sr=sr, ax=ax2) 
        ax2.set_title('MFCCs') 
        st.pyplot(fig2)

    else:
        st.warning("⚠️ Please upload an audio file firs in 'Upload & Score'.")

# ============================= BATCH SCORING ==================================

elif selected=='Batch Scoring':
    st.title('📁 Batch Audio Scoring (Coming Soon)')
    st.info("Upload multiple `.wav` files and get bulk grammar scores.")


        


