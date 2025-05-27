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
st.set_page_config(page_title="Data Chat", layout="wide")

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
        # Check if there's any default CSV in the directory
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        if csv_files:
            return pd.read_csv(csv_files[0])
        elif os.path.exists("data") and any(f.endswith('.csv') for f in os.listdir('data')):
            csv_files = [f for f in os.listdir('data') if f.endswith('.csv')]
            return pd.read_csv(os.path.join("data", csv_files[0]))
        else:
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to get data context - GENERIC VERSION
def get_data_context(df, query):
    # Basic dataframe info
    context = f"Dataset: {len(df)} rows, {len(df.columns)} columns\n\n"
    
    # Add columns info
    context += "Columns in dataset:\n"
    for col in df.columns:
        context += f"- {col} ({df[col].dtype})\n"
    
    # Get basic stats for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        context += "\nSummary statistics for numeric columns:\n"
        # Get at most 5 numeric columns to avoid overwhelming the context
        for col in numeric_cols[:5]:
            context += f"{col} - mean: {df[col].mean():.2f}, min: {df[col].min():.2f}, max: {df[col].max():.2f}\n"
        
        if len(numeric_cols) > 5:
            context += f"...and {len(numeric_cols) - 5} more numeric columns\n"
    
    # Get basic stats for categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        context += "\nMost common values for categorical columns:\n"
        # Get at most 3 categorical columns to avoid overwhelming the context
        for col in categorical_cols[:3]:
            # Get top 3 most common values
            value_counts = df[col].value_counts().head(3)
            context += f"{col} - "
            for val, count in value_counts.items():
                percent = count / len(df) * 100
                context += f"{val}: {count} ({percent:.1f}%), "
            context = context.rstrip(", ") + "\n"
        
        if len(categorical_cols) > 3:
            context += f"...and {len(categorical_cols) - 3} more categorical columns\n"
    
    # Check for specific columns mentioned in query
    query_lower = query.lower()
    mentioned_cols = []
    
    for col in df.columns:
        col_name_in_query = col.lower().replace('_', ' ') in query_lower or col.lower() in query_lower
        if col_name_in_query:
            mentioned_cols.append(col)
    
    if mentioned_cols:
        context += "\nDetailed stats for columns mentioned in query:\n"
        for col in mentioned_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                context += f"{col} stats:\n"
                context += f"- Mean: {df[col].mean():.2f}\n"
                context += f"- Median: {df[col].median():.2f}\n"
                context += f"- Std: {df[col].std():.2f}\n"
                context += f"- Min: {df[col].min():.2f}\n"
                context += f"- Max: {df[col].max():.2f}\n"
            else:
                context += f"{col} value counts (top 5):\n"
                for val, count in df[col].value_counts().head(5).items():
                    context += f"- {val}: {count} rows ({count/len(df)*100:.1f}%)\n"
    
    # Check for relationship/correlation queries
    if "correlation" in query_lower or "relationship" in query_lower:
        numeric_df = df.select_dtypes(include=['number'])
        if len(numeric_df.columns) >= 2:
            # If specific columns are mentioned, prioritize those for correlation
            if len(mentioned_cols) >= 2:
                numeric_mentioned = [col for col in mentioned_cols if col in numeric_df.columns]
                if len(numeric_mentioned) >= 2:
                    context += "\nCorrelations between mentioned numeric columns:\n"
                    for i, col1 in enumerate(numeric_mentioned[:-1]):
                        for col2 in numeric_mentioned[i+1:]:
                            if col1 in numeric_df.columns and col2 in numeric_df.columns:
                                corr = df[col1].corr(df[col2])
                                context += f"- {col1} and {col2}: {corr:.3f}\n"
            else:
                # If no specific columns or only one column mentioned, show top correlations
                context += "\nTop 5 highest correlations between numeric columns:\n"
                corr_matrix = numeric_df.corr().abs().unstack()
                # Remove self-correlations
                corr_matrix = corr_matrix[corr_matrix < 1.0]
                # Get top 5 correlations
                top_corr = corr_matrix.sort_values(ascending=False).head(5)
                for (col1, col2), corr_value in top_corr.items():
                    context += f"- {col1} and {col2}: {df[col1].corr(df[col2]):.3f}\n"
    
    return context

# Function to chat with the LLM - WORKS WITH BOTH OPENAI VERSIONS
def chat_with_data(client, df, query):
    if not client:
        return "Please provide a valid OpenAI API key to enable chat."
    
    try:
        # Get relevant data context
        data_context = get_data_context(df, query)
        
        # Prepare the system message - GENERIC VERSION
        system_message = f"""You are a data analyst assistant.
        Analyze this dataset and answer questions about it.
        
        DATA CONTEXT:
        {data_context}
        
        When answering:
        1. Be specific with numbers and statistics
        2. Provide insights based on the data
        3. If asked for code, use Python with pandas
        4. For visualizations, include matplotlib or seaborn code
        5. Identify potential patterns, trends, or outliers in the data when relevant
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

# Generate example questions based on dataset
def generate_example_questions(df):
    questions = []
    
    # Basic stats questions
    questions.append("What's the basic summary of this dataset?")
    
    # Get a random numeric column for average question
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        col = np.random.choice(numeric_cols)
        questions.append(f"What's the average {col.replace('_', ' ')}?")
    
    # Correlation question
    if len(numeric_cols) >= 2:
        col1 = np.random.choice(numeric_cols)
        col2 = np.random.choice([c for c in numeric_cols if c != col1])
        questions.append(f"Show the correlation between {col1.replace('_', ' ')} and {col2.replace('_', ' ')}")
    
    # Categorical column question
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols and numeric_cols:
        cat_col = np.random.choice(cat_cols)
        num_col = np.random.choice(numeric_cols)
        questions.append(f"How does {cat_col.replace('_', ' ')} affect {num_col.replace('_', ' ')}?")
    
    # Add a generic trends question
    questions.append("What interesting patterns or trends do you see in this data?")
    
    return questions

# Generate sample data for demonstration
def generate_sample_data():
    # Create a generic sample dataset with various column types
    np.random.seed(42)
    n_samples = 100
    
    # Generate a variety of column types for a generic dataset
    data = {
        'id': [f'ID{i:03d}' for i in range(1, n_samples + 1)],
        'numeric_value_1': np.random.normal(100, 20, n_samples),
        'numeric_value_2': np.random.normal(50, 10, n_samples),
        'category_1': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'category_2': np.random.choice(['Type1', 'Type2', 'Type3'], n_samples),
        'date': pd.date_range(start='2023-01-01', periods=n_samples).strftime('%Y-%m-%d'),
        'boolean_flag': np.random.choice([True, False], n_samples),
        'percentage': np.random.uniform(0, 100, n_samples),
        'count': np.random.randint(0, 50, n_samples)
    }
    
    # Create correlations between some columns for more interesting analysis
    correlation_factor = 0.7
    data['correlated_to_numeric_1'] = data['numeric_value_1'] * correlation_factor + np.random.normal(0, 10, n_samples)
    
    # Add category-specific effects
    for i, category in enumerate(data['category_1']):
        if category == 'A':
            data['numeric_value_2'][i] += 15
        elif category == 'D':
            data['numeric_value_2'][i] -= 10
    
    return pd.DataFrame(data)

# Main app
def main():
    st.title("Data Chat")
    st.markdown("Ask questions about your data using natural language")
    
    # Get OpenAI API key
    api_key = get_openai_api()
    
    # Initialize OpenAI client
    openai_client = get_openai_client(api_key) if api_key else None
    
    # File uploader for CSV
    st.sidebar.title("Data")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV data", type="csv")
    
    # Load data - either from upload or default
    df = load_csv_data(uploaded_file)
    
    if df is None:
        st.warning("No data loaded. Please upload a CSV file.")
        
        # Show expected format
        st.info("""
        Upload any CSV file with your data to get started!
        
        The app will automatically analyze your data and allow you to ask 
        questions about it using natural language.
        """)
        
        # Create sample data to download
        if st.button("Download Sample Data"):
            sample_data = generate_sample_data()
            
            csv = sample_data.to_csv(index=False)
            st.download_button(
                label="Download sample data",
                data=csv,
                file_name="sample_data.csv",
                mime="text/csv"
            )
    else:
        # Display data preview
        with st.expander("Preview Data"):
            st.dataframe(df.head())
            
            # Show basic stats dynamically based on data
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Rows", len(df))
                st.metric("Total Columns", len(df.columns))
            with col2:
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    # Show stats for the first numeric column
                    first_num_col = numeric_cols[0]
                    col_name = first_num_col.replace('_', ' ').title()
                    st.metric(f"Avg {col_name}", f"{df[first_num_col].mean():.2f}")
                    
                    # Show stats for the second numeric column if available
                    if len(numeric_cols) > 1:
                        second_num_col = numeric_cols[1]
                        col_name = second_num_col.replace('_', ' ').title()
                        st.metric(f"Avg {col_name}", f"{df[second_num_col].mean():.2f}")
        
        # Chat interface
        st.header("Ask about the data")
        
        # Generate example questions based on the dataset
        example_questions = generate_example_questions(df)
        
        # Display example questions
        st.markdown("**Example questions:**")
        for question in example_questions:
            st.markdown(f"- {question}")
        
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
        if prompt := st.chat_input("Ask a question about your data..."):
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
