import streamlit as st
import json
import os
import random
from datetime import datetime
import speech_recognition as sr
from sentence_transformers import SentenceTransformer, util
import torch
import pyttsx3

# Custom Streamlit theme for professional look
st.set_page_config(page_title="InterviewBot - DSA Practice", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    .sidebar .sidebar-content { background-color: #ffffff; border-right: 1px solid #e0e0e0; }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stTextArea textarea {
        border: 1px solid #ced4da;
        border-radius: 8px;
        font-size: 16px;
    }
    .chat-message {
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        font-size: 16px;
    }
    .assistant-message {
        background-color: #e9ecef;
        color: #000000;  /* Black text for questions */
    }
    .user-message {
        background-color: #007bff;
        color: white;
    }
    .stProgress .st-bo { background-color: #007bff; }
    h1, h2, h3 { color: #2c3e50; font-family: 'Arial', sans-serif; }
    .stSidebar { padding: 20px; }
    </style>
""", unsafe_allow_html=True)

# Main app logic
def main():
    # Initialize session state variables
    if 'datasets' not in st.session_state:
        st.session_state.datasets = {}
    if 'performance' not in st.session_state:
        st.session_state.performance = {}
    if 'current_difficulty' not in st.session_state:
        st.session_state.current_difficulty = {}
    if 'question_pools' not in st.session_state:
        st.session_state.question_pools = {}
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'scores' not in st.session_state:
        st.session_state.scores = {}  # Track similarity scores for progression
    if 'model' not in st.session_state:
        st.session_state.model = SentenceTransformer('all-MiniLM-L6-v2')
    if 'tts_engine' not in st.session_state:
        st.session_state.tts_engine = pyttsx3.init()
        st.session_state.tts_engine.setProperty('rate', 150)
    if 'current_topic' not in st.session_state:
        st.session_state.current_topic = None
    if 'input_text' not in st.session_state:
        st.session_state.input_text = ""
    if 'questions_answered' not in st.session_state:
        st.session_state.questions_answered = {}  # Track questions answered per difficulty

    # Sidebar UI
    with st.sidebar:
        st.header(" InterviewBot")
        topics = [f[:-5] for f in os.listdir('data') if f.endswith('.json')]
        if not topics:
            st.error("No JSON files found in /data/ folder.")
            return
        selected_topic = st.selectbox(" Topic", topics, help="Choose a DSA topic")
        order = st.radio(" Question Order", ["Sequential", "Random"], help="Select question presentation order")
        speech = st.checkbox(" Enable Speech Input", help="Use microphone for answers")
        
        st.subheader(" Progress")
        if selected_topic in st.session_state.performance:
            for d in ['easy', 'medium', 'hard']:
                p = st.session_state.performance[selected_topic][d]
                avg_score = sum(st.session_state.scores.get(selected_topic, {}).get(d, [])) / len(st.session_state.scores.get(selected_topic, {}).get(d, [0])) if st.session_state.scores.get(selected_topic, {}).get(d, []) else 0
                st.write(f"**{d.capitalize()}**: {p['correct']}/{p['attempted']} correct (Avg. Score: {avg_score:.2f})")
        st.write(f"**Current Difficulty**: {st.session_state.current_difficulty.get(selected_topic, 'easy').capitalize()}")

        if st.button(" Export Session", key="export"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = f"logs/session_{timestamp}.json"
            os.makedirs('logs', exist_ok=True)
            export_data = {
                'topic': selected_topic,
                'history': st.session_state.history,
                'performance': st.session_state.performance.get(selected_topic, {}),
                'scores': st.session_state.scores.get(selected_topic, {})
            }
            with open(log_path, 'w') as f:
                json.dump(export_data, f, indent=4)
            st.success(f"Exported to {log_path}")

    # Main UI
    st.title("InterviewBot - Data Structures & Algorithms Practice")
    st.markdown("Practice DSA interview questions with real-time feedback and speech interaction.")

    # Display chat history
    with st.container():
        st.subheader(" Chat History")
        for entry in st.session_state.history:
            with st.container():
                st.markdown(f"<div class='chat-message assistant-message'> {entry['question']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='chat-message user-message'> {entry['user_answer']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div>{entry['evaluation']} (Similarity Score: {entry['score']})</div>", unsafe_allow_html=True)

    # Handle topic change or initialization
    if st.session_state.current_topic != selected_topic:
        st.session_state.current_topic = selected_topic
        if selected_topic not in st.session_state.datasets:
            try:
                with open(f'data/{selected_topic}.json', 'r') as f:
                    data = json.load(f)
                st.session_state.datasets[selected_topic] = data
            except Exception as e:
                st.error(f"Error loading {selected_topic}.json: {e}")
                return
        if selected_topic not in st.session_state.performance:
            st.session_state.performance[selected_topic] = {
                'easy': {'attempted': 0, 'correct': 0, 'partial': 0, 'incorrect': 0},
                'medium': {'attempted': 0, 'correct': 0, 'partial': 0, 'incorrect': 0},
                'hard': {'attempted': 0, 'correct': 0, 'partial': 0, 'incorrect': 0}
            }
        if selected_topic not in st.session_state.scores:
            st.session_state.scores[selected_topic] = {'easy': [], 'medium': [], 'hard': []}
        if selected_topic not in st.session_state.current_difficulty:
            st.session_state.current_difficulty[selected_topic] = 'easy'
        if selected_topic not in st.session_state.questions_answered:
            st.session_state.questions_answered[selected_topic] = {'easy': 0, 'medium': 0, 'hard': 0}
        if selected_topic not in st.session_state.question_pools:
            pools = {}
            data = st.session_state.datasets[selected_topic]
            for d in ['easy', 'medium', 'hard']:
                qlist = data[d][:]
                qlist = random.sample(qlist, min(3, len(qlist)))
                if order == 'Sequential':
                    qlist.sort(key=lambda x: x['qid'])
                pools[d] = qlist
            st.session_state.question_pools[selected_topic] = pools
        st.session_state.history = []
        st.session_state.current_question = None
        st.session_state.input_text = ""

    # Get and display current question
    if 'current_question' not in st.session_state or not st.session_state.current_question:
        current_diff = st.session_state.current_difficulty[selected_topic]
        pool = st.session_state.question_pools[selected_topic][current_diff]
        answered = st.session_state.questions_answered[selected_topic][current_diff]
        if answered >= 3:
            scores = st.session_state.scores[selected_topic][current_diff]
            avg_score = sum(scores) / len(scores) if scores else 0
            if current_diff == 'easy' and avg_score > 0.75:
                st.session_state.current_difficulty[selected_topic] = 'medium'
                st.session_state.questions_answered[selected_topic][current_diff] = 0
                st.success("Moving to medium questions!")
            elif current_diff == 'medium' and avg_score > 0.50:
                st.session_state.current_difficulty[selected_topic] = 'hard'
                st.session_state.questions_answered[selected_topic][current_diff] = 0
                st.success("Moving to hard questions!")
            else:
                st.warning(f"No more questions in {current_diff} difficulty. Average score: {avg_score:.2f}")
                return
            pool = st.session_state.question_pools[selected_topic][st.session_state.current_difficulty[selected_topic]]
        if pool:
            q = pool.pop(0)
            st.session_state.current_question = q
            st.session_state.tts_engine.say(q['question'])
            st.session_state.tts_engine.runAndWait()
        else:
            st.warning(f"No more questions available in {current_diff} difficulty.")
            return

    # Display current question
    with st.container():
        st.subheader(" Current Question")
        if st.session_state.current_question:
            st.markdown(f"<div class='chat-message assistant-message'>{st.session_state.current_question['question']}</div>", unsafe_allow_html=True)
            answered = st.session_state.questions_answered[selected_topic][st.session_state.current_difficulty[selected_topic]]
            st.progress(min(answered / 3, 1.0))
            st.caption(f"Questions answered in {st.session_state.current_difficulty[selected_topic]}: {answered}/3")
            if st.button(" Replay Question", key="replay"):
                st.session_state.tts_engine.say(st.session_state.current_question['question'])
                st.session_state.tts_engine.runAndWait()

    # Speech input
    if speech:
        if st.button(" Record Answer", key="record"):
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                st.info("Listening... (Speak clearly)")
                try:
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    text = recognizer.recognize_google(audio)
                    st.session_state.input_text = text
                    st.rerun()
                except sr.WaitTimeoutError:
                    st.error("No speech detected within 5 seconds.")
                except sr.UnknownValueError:
                    st.error("Sorry, could not understand the audio.")
                except sr.RequestError as e:
                    st.error(f"Speech recognition error: {e}")

    # Text input for answer
    with st.container():
        st.subheader(" Your Answer")
        user_answer = st.text_area("Type your answer here:", value=st.session_state.input_text, key="user_answer_input_unique", placeholder="Enter your answer or use speech input...")
        st.session_state.input_text = user_answer

    # Submit button
    if st.button(" Submit Answer", key="submit"):
        if user_answer.strip():
            q = st.session_state.current_question
            model = st.session_state.model
            emb_user = model.encode(user_answer)
            emb_golden = model.encode(q['golden_answer'])
            score = util.cos_sim(emb_user, emb_golden)[0][0].item()
            rounded_score = round(score, 2)
            if score >= 0.75:
                evaluation = " Correct"
                key = 'correct'
            elif score >= 0.5:
                evaluation = " Partially Correct"
                key = 'partial'
            else:
                evaluation = " Incorrect"
                key = 'incorrect'
            current_diff = st.session_state.current_difficulty[selected_topic]
            st.session_state.performance[selected_topic][current_diff]['attempted'] += 1
            st.session_state.performance[selected_topic][current_diff][key] += 1
            st.session_state.scores[selected_topic][current_diff].append(score)
            st.session_state.questions_answered[selected_topic][current_diff] += 1
            st.session_state.history.append({
                "qid": q['qid'],
                "question": q['question'],
                "user_answer": user_answer,
                "evaluation": evaluation,
                "score": rounded_score
            })
            st.session_state.current_question = None
            st.session_state.input_text = ""
            st.rerun()
        else:
            st.error("Please provide an answer before submitting.")

if __name__ == "__main__":
    main()
