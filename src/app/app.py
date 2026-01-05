avg_delay_mins = None
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import pickle
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Blinkit Business Decision Dashboard", layout="wide")

# Sidebar navigation
page = st.sidebar.radio("Select Page", ["Business Dashboard", "Delay Risk Calculator", "AI Feedback Chat"])

if page == "Business Dashboard":
    # Load data from CSV
    df = pd.read_csv("./Blinkit/master_analytical_view.csv", parse_dates=["date"])

    # Date filter with quick selectors
    import datetime
    min_date = df["date"].min()
    max_date = df["date"].max()

    col_date1, col_date2 = st.columns([2,1])
    with col_date1:
        date_range = st.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
    with col_date2:
        quick_range = st.selectbox(
            "Quick Range",
            ("Custom Range", "Last 7 Days", "Last Month", "Last 3 Months")
        )

    if quick_range == "Last 7 Days":
        end = pd.to_datetime(max_date)
        start = end - pd.Timedelta(days=6)
        mask = (df["date"] >= start) & (df["date"] <= end)
        df = df.loc[mask]
    elif quick_range == "Last Month":
        end = pd.to_datetime(max_date)
        start = (end - pd.DateOffset(months=1)) + pd.Timedelta(days=1)
        mask = (df["date"] >= start) & (df["date"] <= end)
        df = df.loc[mask]
    elif quick_range == "Last 3 Months":
        end = pd.to_datetime(max_date)
        start = (end - pd.DateOffset(months=3)) + pd.Timedelta(days=1)
        mask = (df["date"] >= start) & (df["date"] <= end)
        df = df.loc[mask]
    else:
        if len(date_range) == 2:
            mask = (df["date"] >= pd.to_datetime(date_range[0])) & (df["date"] <= pd.to_datetime(date_range[1]))
            df = df.loc[mask]

    # KPI Cards
    avg_delay_mins = None
    if "average_delay_mins" in df.columns:
        avg_delay_mins = df["average_delay_mins"].mean()
    elif "delay_mins" in df.columns:
        avg_delay_mins = df["delay_mins"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Revenue", f"₹{df['total_revenue'].sum():,.0f}")
    col2.metric("Total Ad Spend", f"₹{df['total_spend'].sum():,.0f}")
    col3.metric("Avg ROAS", f"{df['roas'].mean():.2f}")
    col4.metric("Avg Delay Rate", f"{df['late_rate'].mean():.2%}")
    if avg_delay_mins is not None:
        col5.metric("Avg Delay (mins)", f"{avg_delay_mins:.2f}")
    else:
        col5.metric("Avg Delay (mins)", "N/A")

    # Dual-Axis Chart: Revenue (line) and Ad Spend (bar)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["date"], y=df["total_spend"], name="Ad Spend", marker_color="indianred", yaxis="y2"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["total_revenue"], name="Revenue", mode="lines+markers", line=dict(color="green")))
    fig.update_layout(
        title="Revenue vs Ad Spend Over Time",
        xaxis_title="Date",
        yaxis=dict(
            title=dict(text="Revenue", font=dict(color="green")),
            tickfont=dict(color="green")
        ),
        yaxis2=dict(
            title=dict(text="Ad Spend", font=dict(color="indianred")),
            tickfont=dict(color="indianred"),
            overlaying="y",
            side="right"
        ),
        legend=dict(x=0.01, y=0.99),
        bargap=0.2,
        height=500,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Highlight non-performing campaigns (ROAS < 2.0)
    low_roas = df[df["roas"] < 2.0]
    if not low_roas.empty:
        st.warning(f"{len(low_roas)} day(s) with ROAS < 2.0. Review campaign effectiveness below:")
        st.dataframe(low_roas[["date", "total_revenue", "total_spend", "roas"]])

    # Show data table
    with st.expander("Show Analytical Data Table"):
        st.dataframe(df)

elif page == "Delay Risk Calculator":

    st.title("Delivery Delay Risk Calculator")
    # Load model and encoder
    with open("./Blinkit/delay_risk_model.pkl", "rb") as f:
        model, le_loaded = pickle.load(f)
    region_list = list(le_loaded.classes_)
    region = st.selectbox("Select Area/Region", region_list)
    hour = st.slider("Select Hour of Day (24h)", 0, 23, 18)
    day_of_week = st.selectbox("Select Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    day_of_week_num = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_of_week)
    if st.button("Calculate Delay Risk"):
        region_enc = le_loaded.transform([region])[0]
        X_input = np.array([[hour, day_of_week_num, region_enc]])
        risk_prob = model.predict_proba(X_input)[0, 1]
        if risk_prob > 0.7:
            st.error(f"⚠️ High Risk of Delay ({risk_prob*100:.1f}%)")
        elif risk_prob > 0.4:
            st.warning(f"Moderate Risk of Delay ({risk_prob*100:.1f}%)")
        else:
            st.success(f"Low Risk of Delay ({risk_prob*100:.1f}%)")

elif page == "AI Feedback Chat":
    
    st.title("🤖 Customer Feedback RAG Chat")
    st.markdown("*AI-powered feedback analysis using local Ollama LLM*")
    
    # --- Configuration ---
    OLLAMA_API_URL = "http://localhost:11434/api/generate"
    OLLAMA_MODEL = "llama3.2"
    
    # --- Load Data ---
    @st.cache_data
    def load_feedback():
        df = pd.read_csv("./Blinkit/feedback_analytical_view.csv")
        return df
    
    df_feedback = load_feedback()
    
    # --- Prepare Documents for Embedding ---
    feedback_texts = df_feedback["feedback_text"].fillna("").astype(str).tolist()
    meta = df_feedback[[
        "feedback_id", "rating", "feedback_category", "sentiment", "feedback_date",
        "order_id", "order_date", "order_total", "delivery_status", "payment_method",
        "customer_id", "customer_name", "area",
        "product_names", "product_categories", "total_quantity"
    ]].to_dict("records")
    
    # Use sentence-transformers for embeddings
    @st.cache_resource
    def get_embedder():
        import os
        model_path = os.path.expanduser("~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf")
        return SentenceTransformer(model_path, local_files_only=True)
    
    embedder = get_embedder()
    
    @st.cache_data
    def generate_embeddings(_embedder, texts):
        return _embedder.encode(texts, show_progress_bar=False)
    
    feedback_embeddings = generate_embeddings(embedder, feedback_texts)
    
    # --- RAG Search Function ---
    def search_feedback(query, k=8):
        query_emb = embedder.encode([query])[0]
        sims = cosine_similarity([query_emb], feedback_embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:k]
        return [(feedback_texts[i], meta[i], sims[i]) for i in top_idx]
    
    # --- Ollama LLM Function ---
    def query_ollama(prompt, model=OLLAMA_MODEL):
        """Query local Ollama instance"""
        try:
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 512
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get("response", "No response generated")
            else:
                return f"Error: {response.status_code} - {response.text}"
        except requests.exceptions.ConnectionError:
            return "❌ **Ollama not running!**\n\nPlease start Ollama with: `brew services start ollama`"
        except Exception as e:
            return f"Error: {str(e)}"
    
    # --- Chat Interface ---
    st.markdown("#### Ask about customer feedback")
    
    user_query = st.text_input("Your Question", "Why are customers unhappy with delivery?")
    
    if st.button("Get Root Cause Analysis") and user_query:
        with st.spinner("Retrieving relevant feedback and generating answer..."):
            # Search relevant feedback
            top_feedback = search_feedback(user_query, k=8)
            
            # Build context with more details
            context_lines = []
            for txt, m, _ in top_feedback:
                context_line = f"- Feedback: '{txt}' | Rating: {m.get('rating', 'N/A')}/5 | Sentiment: {m.get('sentiment', 'N/A')} | Area: {m.get('area', 'N/A')} | Products: {m.get('product_names', 'N/A')} | Category: {m.get('product_categories', 'N/A')} | Order Date: {m.get('order_date', 'N/A')}"
                context_lines.append(context_line)
            context = "\n".join(context_lines)
            
            # Create prompt
            prompt = f"""You are an expert business analyst specializing in root cause analysis for e-commerce operations.

QUESTION: {user_query}

RELEVANT CUSTOMER FEEDBACK DATA:
{context}

INSTRUCTIONS:
1. Read all the feedback comments carefully
2. Identify SPECIFIC patterns across the data (product names, areas, categories, dates, ratings, sentiments)
3. Provide a root cause summary that answers "WHY" this is happening, not just "WHAT" happened
4. Include specific details: Which products? Which areas? What specific complaints?
5. Focus on actionable insights that can drive business decisions

OUTPUT FORMAT:
Provide a clear, specific root cause analysis in 3-5 sentences. Example: "Customers are reporting that Alphonso Mangoes in the South Zone are arriving damaged due to poor packaging."

ROOT CAUSE ANALYSIS:"""
            
            # Query Ollama
            summary = query_ollama(prompt, model=OLLAMA_MODEL)
            
            # Display results
            st.subheader("🎯 Root Cause Summary:")
            st.write(summary)
            
            st.markdown("---")
            st.markdown("**📊 Top Relevant Feedback Comments:**")
            
            for txt, meta_info, sim in top_feedback:
                with st.expander(f"Feedback (Similarity: {sim:.2%}) - Rating: {meta_info.get('rating', 'N/A')}"):
                    st.write(f"**Comment:** {txt}")
                    st.json(meta_info)
    
    # Sidebar Info for Feedback Chat
    if page == "AI Feedback Chat":
        with st.sidebar:
            st.markdown("---")
            st.subheader("📈 Dataset Info")
            st.write(f"Total Feedback: {len(df_feedback)}")
            st.write(f"Embeddings: {feedback_embeddings.shape}")

