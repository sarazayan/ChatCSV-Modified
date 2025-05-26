import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import re
from io import StringIO

# Try different OpenAI import approaches to handle different versions
try:
    from openai import OpenAI
    HAS_NEW_OPENAI = True
except ImportError:
    import openai
    HAS_NEW_OPENAI = False

# Page configuration
st.set_page_config(page_title="Student Data Chat", layout="wide")

# Initialize OpenAI API - ROBUST VERSION that handles different client versions
def get_openai_client(api_key):
    if not api_key:
        return None
    
    # Try multiple initialization methods to handle different OpenAI versions
    try:
        if HAS_NEW_OPENAI:
            # New OpenAI client (v1.0.0+)
            client = OpenAI(api_key=api_key)
            # Test the client with a simple completion to verify it works
            client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return client
        else:
            # Legacy OpenAI client
            openai.api_key = api_key
            # Test with a simple completion
            openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return openai
    except Exception as e:
        # If the standard approach fails, try without proxies parameter
        try:
            if HAS_NEW_OPENAI:
                # Try creating client by directly modifying __init__ parameters
                from inspect import signature
                sig = signature(OpenAI.__init__)
                if 'api_key' in sig.parameters:
                    # Only pass parameters that are accepted
                    kwargs = {'api_key': api_key}
                    client = OpenAI(**kwargs)
                    # Test the client
                    client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "Hello"}],
                        max_tokens=5
                    )
                    return client
            else:
                # For legacy client, just set the api key
                openai.api_key = api_key
                return openai
        except Exception as nested_e:
            st.error(f"Could not initialize OpenAI client: {str(nested_e)}")
            return None

# Get OpenAI API key
def get_openai_api():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")
    return api_key

# Load data function with upload capability
@st.cache_data
def load_csv_data(uploaded_file=None):
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            return None
    
    try:
        # Check various possible locations
        if os.path.exists("student_habits_performance.csv"):
            return pd.read_csv("student_habits_performance.csv")
        elif os.path.exists("../student_habits_performance.csv"):
            return pd.read_csv("../student_habits_performance.csv")
        elif os.path.exists("data/student_habits_performance.csv"):
            return pd.read_csv("data/student_habits_performance.csv")
        else:
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to get data context
def get_data_context(df, query):
    # Basic dataframe info
    context = f"Dataset: {len(df)} students, {len(df.columns)} features\n\n"
    
    # Get basic stats for key metrics
    if 'exam_score' in df.columns:
        context += f"Average exam score: {df['exam_score'].mean():.2f}\n"
    if 'study_hours_per_day' in df.columns:
        context += f"Average study hours: {df['study_hours_per_day'].mean():.2f}\n"
    if 'sleep_hours' in df.columns:
        context += f"Average sleep hours: {df['sleep_hours'].mean():.2f}\n"
    if 'attendance_percentage' in df.columns:
        context += f"Average attendance: {df['attendance_percentage'].mean():.2f}%\n"
    
    # Check for specific columns mentioned in query
    query_lower = query.lower()
    for col in df.columns:
        if col.replace('_', ' ') in query_lower:
            # Add detailed stats for mentioned columns
            if pd.api.types.is_numeric_dtype(df[col]):
                context += f"\n{col} stats:\n"
                context += f"- Mean: {df[col].mean():.2f}\n"
                context += f"- Min: {df[col].min():.2f}\n"
                context += f"- Max: {df[col].max():.2f}\n"
            else:
                context += f"\n{col} value counts:\n"
                for val, count in df[col].value_counts().items():
                    context += f"- {val}: {count} students\n"
    
    # Check for relationship queries
    if "correlation" in query_lower or "relationship" in query_lower:
        if 'exam_score' in df.columns:
            context += "\nCorrelations with exam_score:\n"
            for col in df.select_dtypes(include=['number']).columns:
                if col != 'exam_score':
                    corr = df['exam_score'].corr(df[col])
                    context += f"- {col}: {corr:.3f}\n"
    
    return context

# Function to chat with the LLM - WORKS WITH BOTH OPENAI VERSIONS
def chat_with_data(client, df, query):
    if not client:
        return "Please provide a valid OpenAI API key to enable chat."
    
    try:
        # Get relevant data context
        data_context = get_data_context(df, query)
        
        # Prepare the system message
        system_message = f"""You are a data analyst assistant for student data.
        Analyze this student dataset and answer questions about it.
        
        DATA CONTEXT:
        {data_context}
        
        When answering:
        1. Be specific with numbers and statistics
        2. Provide insights based on the data
        3. If asked for code, use Python with pandas
        4. For visualizations, include matplotlib or seaborn code
        """
        
        # Send request to OpenAI - handle both new and legacy client
        if HAS_NEW_OPENAI:
            # New OpenAI client (v1.0.0+)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": query}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content
        else:
            # Legacy OpenAI client
            response = client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": query}
                ],
                temperature=0.2
            )
            return response.choices[0].message['content']
    
    except Exception as e:
        return f"Error: {str(e)}"

# Extract and execute code from responses
def execute_code(df, response_text):
    # Find code blocks between triple backticks
    code_start = response_text.find("```python")
    if code_start == -1:
        code_start = response_text.find("```")
    
    if code_start == -1:
        return []  # No code blocks found
    
    # Find the end of the code block
    code_start = response_text.find("\n", code_start) + 1
    code_end = response_text.find("```", code_start)
    
    if code_end == -1:
        return []  # Incomplete code block
    
    # Extract the code
    code = response_text[code_start:code_end].strip()
    
    try:
        # Set non-interactive backend for matplotlib
        plt.switch_backend('agg')
        
        # Set up namespace with dataframe and libraries
        namespace = {
            'df': df, 
            'pd': pd, 
            'np': np, 
            'plt': plt, 
            'sns': sns
        }
        
        # Execute the code
        exec(code, namespace)
        
        # Check if a plot was created
        if plt.get_fignums():
            fig = plt.gcf()
            return [("figure", fig)]
        
        return []
    
    except Exception as e:
        return [("error", f"Error executing code: {str(e)}")]

# Main app
def main():
    st.title("Student Data Chat")
    st.markdown("Ask questions about student performance data using natural language")
    
    # Get OpenAI API key
    api_key = get_openai_api()
    
    # Initialize OpenAI client
    openai_client = get_openai_client(api_key) if api_key else None
    
    # File uploader for CSV
    st.sidebar.title("Data")
    uploaded_file = st.sidebar.file_uploader("Upload your student data CSV", type="csv")
    
    # Load data - either from upload or default
    df = load_csv_data(uploaded_file)
    
    if df is None:
        st.warning("No data loaded. Please upload a CSV file with student data.")
        
        # Show expected format
        st.info("""
        The CSV should have columns like:
        - student_id, age, gender, study_hours_per_day
        - social_media_hours, netflix_hours, part_time_job
        - attendance_percentage, sleep_hours, diet_quality
        - exercise_frequency, parental_education_level
        - internet_quality, mental_health_rating
        - extracurricular_participation, exam_score
        """)
        
        # Create sample data to download
        if st.button("Download Sample CSV"):
            sample_data = pd.DataFrame({
                'student_id': ['S001', 'S002', 'S003'],
                'age': [18, 19, 20],
                'gender': ['Male', 'Female', 'Male'],
                'study_hours_per_day': [2.5, 3.0, 1.5],
                'social_media_hours': [3.0, 2.0, 4.0],
                'netflix_hours': [2.0, 1.0, 3.0],
                'part_time_job': ['Yes', 'No', 'Yes'],
                'attendance_percentage': [85.0, 92.0, 78.0],
                'sleep_hours': [7.0, 8.0, 6.0],
                'diet_quality': ['Good', 'Excellent', 'Poor'],
                'exercise_frequency': [3, 5, 1],
                'parental_education_level': ['College', 'Graduate', 'High School'],
                'internet_quality': ['Good', 'Excellent', 'Fair'],
                'mental_health_rating': [8, 9, 6],
                'extracurricular_participation': ['Yes', 'Yes', 'No'],
                'exam_score': [78.5, 92.0, 65.0]
            })
            
            csv = sample_data.to_csv(index=False)
            st.download_button(
                label="Download sample data",
                data=csv,
                file_name="sample_student_data.csv",
                mime="text/csv"
            )
    else:
        # Display data preview
        with st.expander("Preview Data"):
            st.dataframe(df.head())
            
            # Show basic stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Students", len(df))
                if 'exam_score' in df.columns:
                    st.metric("Avg Exam Score", f"{df['exam_score'].mean():.1f}")
            with col2:
                if 'study_hours_per_day' in df.columns:
                    st.metric("Avg Study Hours", f"{df['study_hours_per_day'].mean():.1f}")
                if 'sleep_hours' in df.columns:
                    st.metric("Avg Sleep Hours", f"{df['sleep_hours'].mean():.1f}")
        
        # Chat interface
        st.header("Ask about the data")
        
        # Example questions
        st.markdown("""
        **Example questions:**
        - What's the average exam score?
        - Show the correlation between study hours and exam scores
        - How does gender affect performance?
        - Which factors impact mental health ratings?
        """)
        
        # Initialize session state for messages
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display any charts
                for result_type, result in message.get("results", []):
                    if result_type == "figure":
                        st.pyplot(result)
                    elif result_type == "error":
                        st.error(result)
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the student data..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Display assistant response
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                response_placeholder.text("Analyzing data...")
                
                if not openai_client:
                    response_placeholder.error("OpenAI client initialization failed. Please check your API key.")
                    # Add error response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "Error: OpenAI client initialization failed. Please check your API key.",
                        "results": []
                    })
                else:
                    try:
                        # Get response
                        response_text = chat_with_data(openai_client, df, prompt)
                        response_placeholder.markdown(response_text)
                        
                        # Extract and execute code if present
                        results = execute_code(df, response_text)
                        
                        # Display results
                        for result_type, result in results:
                            if result_type == "figure":
                                st.pyplot(result)
                            elif result_type == "error":
                                st.error(result)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response_text,
                            "results": results
                        })
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        response_placeholder.error(error_msg)
                        # Add error response to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_msg,
                            "results": []
                        })

if __name__ == "__main__":
    main()
