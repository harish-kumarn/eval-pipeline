import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="RAG Metrics Dashboard", layout="wide", page_icon="📊")

# Title and description
st.title("🔍 RAG Performance Analysis Dashboard")
st.markdown("Upload your RAG metrics CSV to analyze retrieval and generation performance")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Filter by tag
    if 'tag' in df.columns:
        tags = ['All'] + list(df['tag'].unique())
        selected_tag = st.sidebar.selectbox("Query Tag", tags)
        if selected_tag != 'All':
            df_filtered = df[df['tag'] == selected_tag]
        else:
            df_filtered = df.copy()
    else:
        df_filtered = df.copy()
    
    # Filter by category
    if 'category' in df.columns:
        categories = ['All'] + list(df['category'].unique())
        selected_category = st.sidebar.selectbox("Category", categories)
        if selected_category != 'All':
            df_filtered = df_filtered[df_filtered['category'] == selected_category]
    
    # Summary Metrics
    st.header("📈 Summary Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_ret_score = df_filtered['retrieval_score'].mean() * 100
        st.metric("Avg Retrieval Score", f"{avg_ret_score:.1f}%")
    
    with col2:
        avg_gen_score = df_filtered['generation_score'].mean() * 100
        st.metric("Avg Generation Score", f"{avg_gen_score:.1f}%")
    
    with col3:
        avg_mrr = df_filtered['ret_mrr'].mean() * 100
        st.metric("Avg MRR", f"{avg_mrr:.1f}%")
    
    with col4:
        avg_ndcg = df_filtered['ret_ndcg'].mean() * 100
        st.metric("Avg NDCG", f"{avg_ndcg:.1f}%")
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        avg_faith = df_filtered['gen_faithfulness'].mean() * 100
        st.metric("Avg Faithfulness", f"{avg_faith:.1f}%")
    
    with col6:
        avg_rel = df_filtered['gen_answer_relevancy'].mean() * 100
        st.metric("Avg Answer Relevancy", f"{avg_rel:.1f}%")
    
    with col7:
        avg_correct = df_filtered['gen_answer_correctness'].mean() * 100
        st.metric("Avg Correctness", f"{avg_correct:.1f}%")
    
    with col8:
        hit_rate = (df_filtered['retrieval_status'] == 'HIT').sum() / len(df_filtered) * 100
        st.metric("Hit Rate", f"{hit_rate:.1f}%")
    
    # Distribution Analysis
    st.header("📊 Performance Distribution")
    
    tab1, tab2, tab3 = st.tabs(["Retrieval Metrics", "Generation Metrics", "Comparative Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Retrieval metrics comparison
            ret_metrics = df_filtered[['ret_precision', 'ret_recall', 'ret_ndcg', 'ret_mrr']].mean()
            fig_ret = go.Figure(data=[
                go.Bar(
                    x=['Precision', 'Recall', 'NDCG', 'MRR'],
                    y=ret_metrics.values * 100,
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
                )
            ])
            fig_ret.update_layout(
                title="Average Retrieval Metrics",
                yaxis_title="Score (%)",
                height=400
            )
            st.plotly_chart(fig_ret, width='stretch')
        
        with col2:
            # Retrieval score by tag
            if 'tag' in df_filtered.columns:
                ret_by_tag = df_filtered.groupby('tag')['retrieval_score'].mean().reset_index()
                fig_tag = px.bar(
                    ret_by_tag, 
                    x='tag', 
                    y='retrieval_score',
                    title="Retrieval Score by Query Tag",
                    labels={'retrieval_score': 'Retrieval Score', 'tag': 'Query Tag'}
                )
                fig_tag.update_traces(marker_color='#45B7D1')
                fig_tag.update_layout(height=400)
                st.plotly_chart(fig_tag, width='stretch')
        
        # Rank distribution
        rank_counts = df_filtered['ret_rank'].value_counts().sort_index().reset_index()
        rank_counts.columns = ['Rank Position', 'Count']
        fig_rank = px.bar(
            rank_counts, 
            x='Rank Position',
            y='Count',
            title="Distribution of Retrieved Document Ranks",
            labels={'Rank Position': 'Rank Position', 'Count': 'Frequency'}
        )
        fig_rank.update_traces(marker_color='#4ECDC4')
        fig_rank.update_xaxes(type='category')
        st.plotly_chart(fig_rank, width='stretch')
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Generation metrics comparison
            gen_metrics = df_filtered[['gen_faithfulness', 'gen_answer_relevancy', 
                                       'gen_answer_correctness']].mean()
            fig_gen = go.Figure(data=[
                go.Bar(
                    x=['Faithfulness', 'Relevancy', 'Correctness'],
                    y=gen_metrics.values * 100,
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1']
                )
            ])
            fig_gen.update_layout(
                title="Average Generation Metrics",
                yaxis_title="Score (%)",
                height=400
            )
            st.plotly_chart(fig_gen, width='stretch')
        
        with col2:
            # Generation score by outcome
            if 'outcome' in df_filtered.columns:
                gen_by_outcome = df_filtered.groupby('outcome')['generation_score'].mean().reset_index()
                fig_outcome = px.bar(
                    gen_by_outcome, 
                    x='outcome', 
                    y='generation_score',
                    title="Generation Score by Outcome",
                    labels={'generation_score': 'Generation Score', 'outcome': 'Outcome'}
                )
                fig_outcome.update_traces(marker_color='#FF6B6B')
                fig_outcome.update_layout(height=400)
                st.plotly_chart(fig_outcome, width='stretch')
        
        # Faithfulness vs Correctness scatter
        fig_scatter = px.scatter(
            df_filtered,
            x='gen_faithfulness',
            y='gen_answer_correctness',
            color='tag' if 'tag' in df_filtered.columns else None,
            size='gen_answer_relevancy',
            hover_data=['query_id'],
            title="Faithfulness vs Correctness (sized by Relevancy)",
            labels={'gen_faithfulness': 'Faithfulness', 'gen_answer_correctness': 'Correctness'}
        )
        st.plotly_chart(fig_scatter, width='stretch')
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Retrieval vs Generation scores
            fig_comp = px.scatter(
                df_filtered,
                x='retrieval_score',
                y='generation_score',
                color='category' if 'category' in df_filtered.columns else None,
                hover_data=['query_id'],
                title="Retrieval Score vs Generation Score",
                labels={'retrieval_score': 'Retrieval Score', 'generation_score': 'Generation Score'}
            )
            fig_comp.add_shape(
                type="line",
                x0=0, y0=0, x1=1, y1=1,
                line=dict(color="gray", dash="dash")
            )
            st.plotly_chart(fig_comp, width='stretch')
        
        with col2:
            # Category distribution
            if 'category' in df_filtered.columns:
                cat_dist = df_filtered['category'].value_counts().reset_index()
                cat_dist.columns = ['category', 'count']
                fig_cat = px.pie(
                    cat_dist,
                    values='count',
                    names='category',
                    title="Query Category Distribution"
                )
                st.plotly_chart(fig_cat, width='stretch')
        
        # Radar chart for overall performance
        avg_metrics = {
            'Retrieval': df_filtered['retrieval_score'].mean(),
            'Generation': df_filtered['generation_score'].mean(),
            'Faithfulness': df_filtered['gen_faithfulness'].mean(),
            'Relevancy': df_filtered['gen_answer_relevancy'].mean(),
            'Correctness': df_filtered['gen_answer_correctness'].mean()
        }
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=list(avg_metrics.values()),
            theta=list(avg_metrics.keys()),
            fill='toself',
            name='Overall Performance'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Overall RAG Performance Radar"
        )
        st.plotly_chart(fig_radar, width='stretch')
    
    # Detailed Query Analysis
    st.header("🔎 Detailed Query Analysis")
    
    # Add search functionality
    search_query = st.text_input("Search queries", "")
    if search_query:
        df_display = df_filtered[df_filtered['query'].str.contains(search_query, case=False, na=False)]
    else:
        df_display = df_filtered
    
    # Display dataframe with formatting
    st.dataframe(
        df_display.style.background_gradient(subset=['retrieval_score', 'generation_score'], cmap='RdYlGn'),
        width='stretch',
        height=400
    )
    
    
else:
    st.info("👆 Please upload a CSV file to begin analysis")
    
    # Show sample format
    st.subheader("Expected CSV Format")
    st.markdown("""
    Your CSV should contain the following columns:
    - `query_id`: Unique identifier for the query
    - `query`: The actual query text
    - `tag`: Query tag (e.g., direct, indirect, image_based)
    - `category`: Performance category
    - `retrieval_score`: Overall retrieval score (0-1)
    - `generation_score`: Overall generation score (0-1)
    - `ret_*`: Retrieval metrics (precision, recall, ndcg, mrr, rank)
    - `gen_*`: Generation metrics (faithfulness, relevancy, correctness, etc.)
    """)