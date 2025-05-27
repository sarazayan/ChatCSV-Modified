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
st.set_page_config(page_title="CSV Data Chat", layout="wide")

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
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except:
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
        for filepath in ["data.csv", "dataset.csv", "../data.csv", "data/data.csv"]:
            if os.path.exists(filepath):
                return pd.read_csv(filepath)
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Helper function to classify query intent
def classify_query_intent(query):
    """Classify the intent of the query to determine relevant context"""
    query_lower = query.lower()
    
    # Define intent categories and their keywords
    intents = {
        'summary': ['summary', 'overview', 'describe', 'what is', 'tell me about'],
        'statistics': ['average', 'mean', 'median', 'maximum', 'minimum', 'std', 'variance', 'count'],
        'comparison': ['compare', 'difference', 'versus', 'vs', 'against', 'than'],
        'correlation': ['correlation', 'relationship', 'associated', 'connection', 'related'],
        'distribution': ['distribution', 'spread', 'range', 'histogram'],
        'outliers': ['outlier', 'anomaly', 'unusual', 'extreme'],
        'grouping': ['group by', 'categorize', 'segment', 'bucket'],
        'filtering': ['filter', 'where', 'condition', 'criteria', 'matching'],
        'ranking': ['top', 'bottom', 'highest', 'lowest', 'best', 'worst', 'rank']
    }
    
    # Detect intents
    detected_intents = []
    for intent, keywords in intents.items():
        if any(keyword in query_lower for keyword in keywords):
            detected_intents.append(intent)
    
    return detected_intents or ['summary']  # Default to summary if no intent detected

# Helper function to detect important features
def detect_important_features(df, target_col=None):
    """Detect important features in the dataset using simple heuristics"""
    # If target column is specified, compute correlations
    if target_col and target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
        numeric_cols = df.select_dtypes(include=['number']).columns
        correlations = {}
        for col in numeric_cols:
            if col != target_col:
                try:
                    correlations[col] = abs(df[col].corr(df[target_col]))
                except:
                    correlations[col] = 0  # If correlation fails
        
        # Sort by absolute correlation
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        return [col for col, corr in sorted_features[:5]]  # Return top 5
    
    # If no target, use variance as a simple importance metric
    numeric_cols = df.select_dtypes(include=['number']).columns
    variances = {}
    for col in numeric_cols:
        # Normalize by mean to make it comparable across features
        try:
            if df[col].mean() != 0:
                variances[col] = df[col].std() / abs(df[col].mean())
            else:
                variances[col] = df[col].std()
        except:
            variances[col] = 0  # If calculation fails
    
    # Sort by variance
    sorted_features = sorted(variances.items(), key=lambda x: x[1], reverse=True)
    return [col for col, var in sorted_features[:5]]  # Return top 5

# NEW FLEXIBLE PREPROCESSING FUNCTION
def get_data_context(df, query, max_context_length=4000):
    """Generate flexible context based on dataset and query"""
    # 1. Analyze dataset structure
    data_info = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
        "datetime_columns": df.select_dtypes(include=['datetime64']).columns.tolist(),
        "missing_values": df.isna().sum().sum(),
        "duplicate_rows": df.duplicated().sum()
    }
    
    # 2. Analyze query intent
    intents = classify_query_intent(query)
    
    # 3. Identify mentioned columns
    mentioned_cols = []
    for col in df.columns:
        col_variations = [col.lower(), col.lower().replace('_', ' ')]
        if any(variation in query.lower() for variation in col_variations):
            mentioned_cols.append(col)
    
    # 4. If no specific columns mentioned, find important ones
    target_cols = mentioned_cols
    if not target_cols:
        # Try to identify potential target columns
        potential_targets = [col for col in df.columns if any(
            term in col.lower() for term in ['score', 'target', 'result', 'output', 'performance', 'rating'])]
        
        if potential_targets:
            # If target columns exist, find columns correlated with them
            target = potential_targets[0]  # Use the first one
            target_cols = detect_important_features(df, target)
        else:
            # Otherwise, just use columns with highest variance
            target_cols = detect_important_features(df)
    
    # 5. Build context based on intents and columns
    context = f"Dataset: {data_info['row_count']} rows, {data_info['column_count']} columns\n"
    context += f"Column types: {len(data_info['numeric_columns'])} numeric, {len(data_info['categorical_columns'])} categorical\n"
    context += f"Data quality: {data_info['missing_values']} missing values, {data_info['duplicate_rows']} duplicate rows\n\n"
    
    # Add intent-specific information
    for intent in intents:
        if intent == 'summary':
            # Add dataset summary
            context += "Dataset Summary:\n"
            sample_cols = df.columns[:5].tolist()  # First 5 columns as sample
            context += f"Sample columns: {', '.join(sample_cols)}\n"
            
        elif intent == 'statistics' or intent == 'summary':
            # Add statistics for target columns
            for col in target_cols:
                if col in data_info['numeric_columns']:
                    context += f"\n{col} statistics:\n"
                    context += f"- Mean: {df[col].mean():.2f}\n"
                    context += f"- Median: {df[col].median():.2f}\n"
                    context += f"- Std Dev: {df[col].std():.2f}\n"
                    context += f"- Min: {df[col].min():.2f}\n"
                    context += f"- Max: {df[col].max():.2f}\n"
                elif col in data_info['categorical_columns']:
                    value_counts = df[col].value_counts()
                    context += f"\n{col} categories (top 5):\n"
                    for val, count in value_counts.head(5).items():
                        context += f"- {val}: {count} ({count/len(df)*100:.1f}%)\n"
                    if len(value_counts) > 5:
                        context += f"- Plus {len(value_counts)-5} more categories\n"
        
        elif intent == 'correlation':
            # Add correlation information
            num_mentioned = [col for col in mentioned_cols if col in data_info['numeric_columns']]
            if len(num_mentioned) >= 2:
                # Correlations between mentioned numeric columns
                context += "\nCorrelations between mentioned columns:\n"
                for i, col1 in enumerate(num_mentioned):
                    for col2 in num_mentioned[i+1:]:
                        corr = df[col1].corr(df[col2])
                        context += f"- {col1} and {col2}: {corr:.3f}\n"
            else:
                # General top correlations
                numeric_cols = data_info['numeric_columns']
                if len(numeric_cols) >= 2:
                    corr_matrix = df[numeric_cols].corr()
                    corrs = []
                    for i, col1 in enumerate(numeric_cols):
                        for j, col2 in enumerate(numeric_cols):
                            if i < j:  # Avoid duplicates and self-correlations
                                corrs.append((col1, col2, abs(corr_matrix.loc[col1, col2])))
                    
                    # Sort by correlation strength and take top 5
                    corrs.sort(key=lambda x: x[2], reverse=True)
                    context += "\nTop correlations:\n"
                    for col1, col2, corr_abs in corrs[:5]:
                        corr = corr_matrix.loc[col1, col2]  # Get actual value with sign
                        context += f"- {col1} and {col2}: {corr:.3f}\n"
        
        elif intent == 'distribution':
            # Add distribution information for numeric columns
            for col in target_cols:
                if col in data_info['numeric_columns']:
                    context += f"\n{col} distribution:\n"
                    # Calculate percentiles
                    percentiles = [10, 25, 50, 75, 90]
                    for p in percentiles:
                        context += f"- {p}th percentile: {df[col].quantile(p/100):.2f}\n"
        
        elif intent == 'grouping' and mentioned_cols:
            # Add groupby information if categorical and numeric columns are mentioned
            cat_cols = [col for col in mentioned_cols if col in data_info['categorical_columns']]
            num_cols = [col for col in mentioned_cols if col in data_info['numeric_columns']]
            
            if cat_cols and num_cols:
                cat_col = cat_cols[0]  # Use first categorical column
                num_col = num_cols[0]  # Use first numeric column
                
                try:
                    grouped = df.groupby(cat_col)[num_col].agg(['mean', 'count'])
                    context += f"\n{num_col} grouped by {cat_col}:\n"
                    for idx, row in grouped.head(5).iterrows():
                        context += f"- {idx}: mean {row['mean']:.2f} (count: {row['count']})\n"
                    if len(grouped) > 5:
                        context += f"- Plus {len(grouped)-5} more groups\n"
                except:
                    pass  # Skip if groupby fails
    
    # Ensure context doesn't exceed max length
    if len(context) > max_context_length:
        context = context[:max_context_length-100] + "...[truncated for brevity]"
    
    return context

# Function to chat with the LLM - WORKS WITH BOTH OPENAI VERSIONS
def chat_with_data(client, df, query):
    if not client:
        return "Please provide a valid OpenAI API key to enable chat."
    
    try:
        # Get relevant data context
        data_context = get_data_context(df, query)
        
        # Prepare the system message
        system_message = f"""You are a data analyst assistant.
        Analyze the dataset and answer questions about it.
        
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
    st.title("CSV Data Chat")
    st.markdown("Ask questions about your CSV data using natural language")
    
    # Get OpenAI API key
    api_key = get_openai_api()
    
    # Initialize OpenAI client
    openai_client = get_openai_client(api_key) if api_key else None
    
    # File uploader for CSV
    st.sidebar.title("Data")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
    
    # Load data - either from upload or default
    df = load_csv_data(uploaded_file)
    
    if df is None:
        st.warning("No data loaded. Please upload a CSV file.")
        
        # Show expected format
        st.info("""
        Upload any CSV file with data you want to analyze.
        The app will work with any CSV format, regardless of the data domain.
        """)
        
        # Create sample data to download
        if st.button("Download Sample CSV"):
            # Create a generic sample dataset
            import numpy as np
            
            # Generate sample data
            n_rows = 100
            sample_data = pd.DataFrame({
                'id': range(1, n_rows + 1),
                'numeric_col_1': np.random.normal(100, 15, n_rows),
                'numeric_col_2': np.random.normal(50, 10, n_rows),
                'numeric_col_3': np.random.normal(25, 5, n_rows),
                'category_1': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
                'category_2': np.random.choice(['Group 1', 'Group 2', 'Group 3'], n_rows),
                'date_col': pd.date_range(start='2023-01-01', periods=n_rows).strftime('%Y-%m-%d'),
                'binary_col': np.random.choice(['Yes', 'No'], n_rows)
            })
            
            # Add some correlated data
            sample_data['numeric_col_4'] = sample_data['numeric_col_1'] * 0.5 + sample_data['numeric_col_2'] * 0.3 + np.random.normal(0, 10, n_rows)
            
            # Add some missing values
            for col in sample_data.columns[1:]:
                mask = np.random.random(n_rows) < 0.05  # 5% missing values
                sample_data.loc[mask, col] = np.nan
            
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
            
            # Show basic stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", len(df))
                st.metric("Columns", len(df.columns))
            with col2:
                st.metric("Missing Values", df.isna().sum().sum())
                st.metric("Duplicate Rows", df.duplicated().sum())
        
        # Chat interface
        st.header("Ask about the data")
        
        # Example questions
        st.markdown("""
        **Example questions:**
        - Give me a summary of this dataset
        - What are the statistics for [column name]?
        - Show the correlation between [column1] and [column2]
        - What's the distribution of values in [column]?
        - Which columns have the strongest relationship?
        - Group the data by [categorical column] and show average [numeric column]
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
        if prompt := st.chat_input("Ask a question about the data..."):
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
