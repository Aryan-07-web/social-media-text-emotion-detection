import streamlit as st
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import io
from track_utils import create_page_visited_table, add_page_visited_details, view_all_page_visited_details, add_prediction_details, view_all_prediction_details, create_emotionclf_table, IST

# Page Configuration
st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .emotion-box {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .example-button {
        margin: 0.3rem;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    return joblib.load(open("./models/emotion_classifier_pipe_lr.pkl", "rb"))

pipe_lr = load_model()

# Functions
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def predict_batch(texts):
    """Predict emotions for multiple texts"""
    predictions = pipe_lr.predict(texts)
    probabilities = pipe_lr.predict_proba(texts)
    return predictions, probabilities

emotions_emoji_dict = {
    "anger": "😠", 
    "disgust": "🤮", 
    "fear": "😨", 
    "happy": "🤗", 
    "joy": "😂", 
    "neutral": "😐", 
    "sad": "😔", 
    "sadness": "😔", 
    "shame": "😳", 
    "surprise": "😮"
}

emotion_colors = {
    "anger": "#FF4500",
    "disgust": "#8B4513",
    "fear": "#9370DB",
    "happy": "#FFD700",
    "joy": "#32CD32",
    "neutral": "#808080",
    "sad": "#4169E1",
    "sadness": "#4169E1",
    "shame": "#FF69B4",
    "surprise": "#FFA500"
}

def get_emotion_color(emotion):
    return emotion_colors.get(emotion, "#808080")

# Main Application
def main():
    st.markdown('<h1 class="main-header">😊 Emotion Detection App</h1>', unsafe_allow_html=True)
    
    menu = ["🏠 Home", "📊 Batch Analysis", "📈 Monitor", "ℹ️ About"]
    choice = st.sidebar.selectbox("Navigation", menu)
    
    create_page_visited_table()
    create_emotionclf_table()
    
    if choice == "🏠 Home":
        add_page_visited_details("Home", datetime.now(IST))
        
        st.markdown("### 📝 Single Text Analysis")
        st.markdown("Enter text below to detect emotions")
        
        # Example texts
        example_texts = [
            "I'm so happy today! Everything is going great!",
            "This is so frustrating! I can't believe this happened.",
            "I'm really scared about what might happen tomorrow.",
            "What a surprise! I never expected this!",
            "I feel so sad and lonely right now.",
            "This is disgusting! I can't stand it!"
        ]
        
        st.markdown("**Try these examples:**")
        cols = st.columns(3)
        for idx, example in enumerate(example_texts):
            with cols[idx % 3]:
                if st.button(f"Example {idx + 1}", key=f"ex_{idx}"):
                    st.session_state.example_text = example

        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area(
                "Enter your text here:",
                height=150,
                value=st.session_state.get('example_text', ''),
                placeholder="Type or paste your text here to analyze emotions..."
            )
            submit_text = st.form_submit_button(label='🔍 Analyze Emotion', use_container_width=True)
            
            # Clear example text after using
            if 'example_text' in st.session_state:
                del st.session_state.example_text

        if submit_text and raw_text.strip():
            with st.spinner("Analyzing emotions..."):
                prediction = predict_emotions(raw_text)
                probability = get_prediction_proba(raw_text)
                max_prob = np.max(probability)

                add_prediction_details(raw_text, prediction, max_prob, datetime.now(IST))

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 📄 Original Text")
                    st.info(raw_text)
                    
                    st.markdown("### 🎯 Detected Emotion")
                    emoji_icon = emotions_emoji_dict.get(prediction, "😐")
                    emotion_color = get_emotion_color(prediction)
                    
                    st.markdown(f"""
                    <div class="emotion-box" style="background: linear-gradient(135deg, {emotion_color}20 0%, {emotion_color}40 100%); 
                        border: 2px solid {emotion_color};">
                        <div style="font-size: 4rem; margin-bottom: 0.5rem;">{emoji_icon}</div>
                        <div style="color: {emotion_color}; font-size: 2rem; text-transform: capitalize;">{prediction}</div>
                        <div style="color: #666; font-size: 1.2rem; margin-top: 0.5rem;">
                            Confidence: {max_prob:.1%}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown("### 📊 Emotion Probabilities")
                    proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                    proba_df_clean = proba_df.T.reset_index()
                    proba_df_clean.columns = ["emotions", "probability"]
                    proba_df_clean = proba_df_clean.sort_values('probability', ascending=False)
                    
                    # Create interactive Plotly chart
                    fig = px.bar(
                        proba_df_clean, 
                        x='emotions', 
                        y='probability',
                        color='emotions',
                        color_discrete_map={emotion: get_emotion_color(emotion) for emotion in proba_df_clean['emotions']},
                        labels={'emotions': 'Emotion', 'probability': 'Probability'},
                        title='Emotion Distribution'
                    )
                    fig.update_layout(
                        showlegend=False,
                        height=400,
                        xaxis_title="Emotion",
                        yaxis_title="Probability",
                        yaxis=dict(tickformat='.0%')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show top 3 emotions
                    st.markdown("**Top 3 Emotions:**")
                    top_3 = proba_df_clean.head(3)
                    for idx, row in top_3.iterrows():
                        emoji = emotions_emoji_dict.get(row['emotions'], "😐")
                        st.write(f"{emoji} **{row['emotions'].capitalize()}**: {row['probability']:.1%}")
        
        elif submit_text and not raw_text.strip():
            st.warning("⚠️ Please enter some text to analyze!")

    elif choice == "📊 Batch Analysis":
        add_page_visited_details("Batch Analysis", datetime.now(IST))
        st.markdown("### 📊 Batch Text Analysis")
        st.markdown("Analyze multiple texts at once. Enter each text on a new line.")
        
        batch_text = st.text_area(
            "Enter multiple texts (one per line):",
            height=200,
            placeholder="Text 1\nText 2\nText 3\n..."
        )
        
        if st.button("🔍 Analyze All", use_container_width=True):
            if batch_text.strip():
                texts = [line.strip() for line in batch_text.split('\n') if line.strip()]
                
                if texts:
                    with st.spinner(f"Analyzing {len(texts)} texts..."):
                        predictions, probabilities = predict_batch(texts)
                        max_probs = np.max(probabilities, axis=1)
                        
                        # Create results dataframe
                        results_df = pd.DataFrame({
                            'Text': texts,
                            'Emotion': predictions,
                            'Confidence': max_probs,
                            'Emoji': [emotions_emoji_dict.get(pred, "😐") for pred in predictions]
                        })
                        
                        # Save predictions to database
                        for text, pred, prob in zip(texts, predictions, max_probs):
                            add_prediction_details(text, pred, prob, datetime.now(IST))
                        
                        st.success(f"✅ Successfully analyzed {len(texts)} texts!")
                        
                        # Display results
                        st.markdown("### 📋 Results")
                        st.dataframe(
                            results_df[['Emoji', 'Emotion', 'Text', 'Confidence']],
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Texts", len(texts))
                        with col2:
                            st.metric("Unique Emotions", results_df['Emotion'].nunique())
                        with col3:
                            st.metric("Avg Confidence", f"{results_df['Confidence'].mean():.1%}")
                        
                        # Emotion distribution chart
                        st.markdown("### 📊 Emotion Distribution")
                        emotion_counts = results_df['Emotion'].value_counts()
                        fig = px.pie(
                            values=emotion_counts.values,
                            names=emotion_counts.index,
                            color_discrete_map={emotion: get_emotion_color(emotion) for emotion in emotion_counts.index},
                            title="Distribution of Detected Emotions"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Export to CSV
                        csv_buffer = io.StringIO()
                        results_df.to_csv(csv_buffer, index=False)
                        csv_data = csv_buffer.getvalue()
                        
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv_data,
                            file_name=f"emotion_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                else:
                    st.warning("⚠️ No valid text found. Please enter at least one text.")
            else:
                st.warning("⚠️ Please enter some texts to analyze!")
    
    elif choice == "📈 Monitor":
        add_page_visited_details("Monitor", datetime.now(IST))
        st.markdown("### 📈 App Analytics & Monitoring")

        with st.expander("📊 Page Visit Metrics", expanded=True):
            page_visited_details = pd.DataFrame(view_all_page_visited_details(), columns=['Page Name', 'Time of Visit'])
            
            if not page_visited_details.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Page Visit Data**")
                    st.dataframe(page_visited_details, use_container_width=True, hide_index=True)
                
                with col2:
                    pg_count = page_visited_details['Page Name'].value_counts().rename_axis('Page Name').reset_index(name='Counts')
                    
                    st.markdown("**Visit Counts**")
                    fig_bar = px.bar(
                        pg_count, 
                        x='Page Name', 
                        y='Counts',
                        color='Page Name',
                        title="Page Visits by Count"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                st.markdown("**Visit Distribution**")
                fig_pie = px.pie(
                    pg_count, 
                    values='Counts', 
                    names='Page Name',
                    title="Page Visit Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No page visit data available yet.")

        with st.expander('😊 Emotion Classification Metrics', expanded=True):
            df_emotions = pd.DataFrame(
                view_all_prediction_details(), 
                columns=['Rawtext', 'Prediction', 'Probability', 'Time_of_Visit']
            )
            
            if not df_emotions.empty:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**All Predictions**")
                    # Truncate long texts for display
                    df_display = df_emotions.copy()
                    df_display['Text Preview'] = df_display['Rawtext'].apply(
                        lambda x: x[:50] + "..." if len(x) > 50 else x
                    )
                    st.dataframe(
                        df_display[['Text Preview', 'Prediction', 'Probability', 'Time_of_Visit']],
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Export functionality
                    csv_buffer = io.StringIO()
                    df_emotions.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="📥 Export All Predictions as CSV",
                        data=csv_data,
                        file_name=f"all_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    st.markdown("**Statistics**")
                    st.metric("Total Predictions", len(df_emotions))
                    st.metric("Unique Emotions", df_emotions['Prediction'].nunique())
                    st.metric("Avg Confidence", f"{df_emotions['Probability'].mean():.1%}")
                
                st.markdown("**Emotion Distribution**")
                prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
                
                fig_bar = px.bar(
                    prediction_count,
                    x='Prediction',
                    y='Counts',
                    color='Prediction',
                    color_discrete_map={emotion: get_emotion_color(emotion) for emotion in prediction_count['Prediction']},
                    title="Emotion Prediction Counts"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
                fig_pie = px.pie(
                    prediction_count,
                    values='Counts',
                    names='Prediction',
                    color_discrete_map={emotion: get_emotion_color(emotion) for emotion in prediction_count['Prediction']},
                    title="Emotion Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Time series analysis
                if len(df_emotions) > 1:
                    st.markdown("**Predictions Over Time**")
                    df_emotions['Time_of_Visit'] = pd.to_datetime(df_emotions['Time_of_Visit'])
                    df_emotions['Date'] = df_emotions['Time_of_Visit'].dt.date
                    daily_counts = df_emotions.groupby('Date').size().reset_index(name='Count')
                    
                    fig_line = px.line(
                        daily_counts,
                        x='Date',
                        y='Count',
                        title="Daily Prediction Counts",
                        markers=True
                    )
                    st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.info("No prediction data available yet. Start analyzing texts to see metrics here!")

    else:
        add_page_visited_details("About", datetime.now(IST))
        
        st.markdown("### ℹ️ About This App")
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; color: white; margin-bottom: 2rem;'>
            <h2 style='color: white;'>Welcome to the Emotion Detection App! 😊</h2>
            <p style='font-size: 1.1rem;'>This application utilizes the power of natural language processing and machine learning 
            to analyze and identify emotions in textual data.</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Our Mission")
            st.info("""
            Our mission is to provide a user-friendly and efficient tool that helps individuals 
            and organizations understand the emotional content hidden within text. We believe that 
            emotions play a crucial role in communication, and by uncovering these emotions, we can 
            gain valuable insights into the underlying sentiments and attitudes expressed in written text.
            """)
            
            st.markdown("### ⚙️ How It Works")
            st.info("""
            When you input text into the app, our system processes it and applies advanced natural 
            language processing algorithms to extract meaningful features from the text. These features 
            are then fed into the trained model, which predicts the emotions associated with the input text.
            """)

        with col2:
            st.markdown("### ✨ Key Features")
            
            features = [
                ("🔍 Real-time Detection", "Instantly analyze emotions in any text"),
                ("📊 Batch Analysis", "Analyze multiple texts at once"),
                ("📈 Analytics Dashboard", "Track and monitor all predictions"),
                ("📥 Export Data", "Download results as CSV files"),
                ("🎨 Beautiful UI", "Modern and intuitive interface"),
                ("📉 Confidence Scores", "See prediction reliability")
            ]
            
            for title, desc in features:
                with st.expander(title):
                    st.write(desc)

        st.markdown("### 🌟 Applications")
        
        apps = [
            "📱 Social media sentiment analysis",
            "💬 Customer feedback analysis",
            "📊 Market research and consumer insights",
            "🏢 Brand monitoring and reputation management",
            "📝 Content analysis and recommendation systems",
            "🎓 Educational research and analysis"
        ]
        
        for app in apps:
            st.markdown(f"- {app}")
        
        st.markdown("---")
        st.markdown("### 📊 Supported Emotions")
        
        emotion_cols = st.columns(5)
        emotions_list = list(emotions_emoji_dict.items())[:10]
        
        for idx, (emotion, emoji) in enumerate(emotions_list):
            with emotion_cols[idx % 5]:
                color = get_emotion_color(emotion)
                st.markdown(f"""
                <div style='text-align: center; padding: 1rem; border-radius: 10px; 
                    background: {color}20; border: 2px solid {color}; margin: 0.5rem 0;'>
                    <div style='font-size: 2rem;'>{emoji}</div>
                    <div style='color: {color}; font-weight: bold;'>{emotion.capitalize()}</div>
                </div>
                """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
