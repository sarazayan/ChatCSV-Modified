import streamlit as st
import pandas as pd
import os

# Try OpenAI import
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Page configuration
st.set_page_config(page_title="CSV Chat", layout="wide")

# Get OpenAI API key
def get_openai_api_key():
    """Get OpenAI API key from environment or user input"""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")
    return api_key

# Function to handle file upload
@st.cache_data
def load_csv(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# Function to chat with OpenAI about the data
def chat_with_openai(api_key, df, query):
    if not HAS_OPENAI:
        return "OpenAI package not installed. Please check your installation."
    
    try:
        # Set up openai
        openai.api_key = api_key
        
        # Create a context with dataframe information
        context = f"DataFrame info: {len(df)} rows, {df.columns.tolist()} columns\n\n"
        
        # Add basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            context += f"{col} - mean: {df[col].mean()}, min: {df[col].min()}, max: {df[col].max()}\n"
        
        # Create the prompt
        prompt = f"""
        Based on the following dataframe information, answer the question.
        
        {context}
        
        Question: {query}
        """
        
        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a data analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        return response.choices[0].message['content']
    
    except Exception as e:
        return f"Error: {str(e)}"

# Main application
def main():
    st.title("CSV Chat App")
    st.markdown("Upload a CSV file and ask questions about it")
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Get OpenAI API key
    api_key = get_openai_api_key()
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load data
        df = load_csv(uploaded_file)
        
        if df is not None:
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Initialize session state
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Chat interface
            st.subheader("Ask about your data")
            
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Input for new message
            if prompt := st.chat_input("Ask a question about your data..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate and display response
                with st.chat_message("assistant"):
                    if not api_key:
                        st.warning("Please provide an OpenAI API key.")
                    else:
                        with st.spinner("Thinking..."):
                            response = chat_with_openai(api_key, df, prompt)
                            st.markdown(response)
                            
                            # Add assistant response to chat history
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response
                            })
    else:
        st.info("Please upload a CSV file to begin.")

if __name__ == "__main__":
    main()
