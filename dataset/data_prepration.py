import os
import whisper
import textstat 
import pandas as pd
import language_tool_python
from pydub import AudioSegment

import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")


# =============================== PATHS =========================================

mp3_dir = '/home/sunny/Desktop/Grammer-Scoring-From-Voic-Sample-Engine/en/train_clips_1'
AUDIO_DIR = "/home/sunny/Desktop/Grammer-Scoring-From-Voic-Sample-Engine/en/train_clips_2"
# AUDIO_DIR = '/home/sunny/Desktop/Grammer-Scoring-From-Voic-Sample-Engine/en/train_clips_2'
OUTPUT_CSV = 'test.csv'

os.makedirs(AUDIO_DIR, exist_ok=True)

# ============================ MP3 TO WAV CONVERSION =========================

for filename in os.listdir(mp3_dir):
    if filename.endswith('.mp3'):
        mp3_path = os.path.join(mp3_dir, filename)
        wav_filename=os.path.splitext(filename)[0]+'.wav'
        wav_path = os.path.join(AUDIO_DIR, wav_filename)

        # Convert and Save
        try:
            audio = AudioSegment.from_mp3(mp3_path)
            audio.export(wav_path, format='wav')
            print(f"✔ Converted: {filename}")
        except Exception as e:
            print(f'❌ Failed: {filename} — {e}')

# =========================== RENAME =========================================

files = [f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')]
files.sort() 

for idx, filename in enumerate(files, start=918):
    new_name = f'audio_{idx}.wav'
    src=os.path.join(AUDIO_DIR, filename) 
    dst=os.path.join(AUDIO_DIR, new_name) 
    os.rename(src, dst) 
print('Renaming complete starting from 918')

# =========================== Load Models ====================================
model = whisper.load_model('base')
tool = language_tool_python.LanguageTool('en-US')

# ============================ SCORING FUNCTIONS =============================

def grammer_score(text):
    matches = tool.check(text)
    errors = len(matches)
    words = len(text.split())
    if words == 0:
        return 0.0
    ratio = errors/words

    if ratio ==0: return 5.0
    elif ratio <= 0.03:
        return 4.5
    elif ratio <= 0.07:
        return 4.0
    elif ratio <= 0.12:
        return 3.5
    elif ratio <= 0.18:
        return 3.0
    elif ratio <= 0.25:
        return 2.0
    elif ratio <= 0.40:
        return 1.0
    else: return 0.5

# ============================= FLUENCY SCORE ============================

def fluency_score(text):
    words = text.split()
    word_count = len(words)
    sentence_count = max(1, text.count('.')+text.count('?')+text.count('!'))
    avg_length = word_count / sentence_count
    if avg_length >= 20: 
        return 5.0
    elif avg_length >= 15:
        return 4.0
    elif avg_length >= 10:
        return 3.0 
    elif avg_length >= 5:
        return 2.0
    else: return 1

# =========================== Language Level ================================

def language_level(text):
    fk_score = textstat.flesch_kincaid_grade(text)
    if fk_score <= 3:
        return 'A1'
    elif fk_score <= 5:
        return 'A2'
    elif fk_score <= 7:
        return 'B1'
    elif fk_score <= 9:
        return 'B2'
    elif fk_score <= 12:
        return 'C1'
    else: return 'C2'

# ============================ LANGUAGE LEVEL ===============================

def interview_score(grammer, fluency):
    avg = (grammer + fluency)/2
    if avg >= 4.5: return 5.0
    elif avg >= 4: return 4.5
    elif avg >=3.5:return 4.0
    elif avg >= 3: return 3.5
    elif avg >=2.5:return 3.0
    elif avg >= 2: return 2.5
    elif avg >=1.5:return 2.0
    elif avg >=1 : return 1.5
    else: return 1.0

# ============================= PROCESSING loop =============================

records = []

for file in os.listdir(AUDIO_DIR):
    if file.endswith('.wav'):
        filepath = os.path.join(AUDIO_DIR, file)
        print(f'🔊 Processing {file}')
        try:
            result = model.transcribe(filepath)
            text = result['text'].strip()
        except Exception as e:
            print(f'❌ Failed on {file}: {e}')
            continue

        g_score = grammer_score(text) 
        f_score = fluency_score(text)
        level = language_level(text) 
        i_score= interview_score(g_score, f_score) 

        records.append({
            'filename': file,
            'transcription': text, 
            'grammer_score': g_score,
            'fluency_score': f_score,
            'language_level': level, 
            'interview_score': i_score
        }) 



# ============================ SAVE TO .CSV =============================

data = pd.DataFrame(records) 
cefr_map = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'C1': 5, 'C2': 6}
data['cefr_score'] = data['language_level'].map(cefr_map) 

data['rating'] = data[['grammer_score', 'fluency_score', 'interview_score', 'cefr_score']].mean(axis=1)

def map_to_rating(score):
    rating_levels = [1, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    return min(rating_levels, key=lambda x: abs(x - score))

data['final_rating'] = data['rating'].apply(map_to_rating)

data = data[['filename', 'final_rating']]
data.to_csv(OUTPUT_CSV, index=False)

print(f'\n✅ Saved to {OUTPUT_CSV}')
