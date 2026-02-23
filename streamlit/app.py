import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pymongo import MongoClient
import time
from datetime import datetime, timedelta

st.set_page_config(page_title="RAG Metrics Dashboard", layout="wide", page_icon="📊")

# MongoDB connection
MONGO_URI = "mongodb://admin:admin123@localhost:27017/"

def get_mongo_collection(db_name, collection_name):
    client = MongoClient(MONGO_URI)
    return client[db_name][collection_name]

def performance_analysis_page():
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
                gen_metrics = df_filtered[['gen_faithfulness', 'gen_answer_relevancy', 'gen_answer_correctness']].mean()
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

def telemetry_analysis_page():
    st.title("📊 Telemetry Analysis - Advanced Insights")
    st.markdown("Real-time monitoring of RAG token usage with comprehensive cost and performance analytics")
    
    if 'scraping_active' not in st.session_state:
        st.session_state.scraping_active = False
    
    # Connection and Collection Selection
    with st.sidebar.expander("🔌 MongoDB Settings", expanded=True):
        mongo_uri = st.text_input("Mongo URI", MONGO_URI)
        db_name = st.text_input("Database", "telemetry_db")
        
        try:
            client = MongoClient(mongo_uri)
            default_collections = ["query_usage", "admin_usage"]
            try:
                available_db_names = client.list_database_names()
                if db_name in available_db_names:
                    collections = client[db_name].list_collection_names()
                    if collections:
                        col_options = collections
                    else:
                        col_options = default_collections
                else:
                    col_options = default_collections
            except:
                col_options = default_collections
            
            col_name = st.selectbox("Collection", col_options, 
                                   index=col_options.index("query_usage") if "query_usage" in col_options else 0)
        except Exception as e:
            st.error(f"Connection error: {e}")
            col_name = "query_usage"
    
    # Time range filter
    with st.sidebar.expander("⏰ Time Range Filter", expanded=False):
        time_range = st.selectbox("Select Time Range", 
                                 ["Last 100 records", "Last hour", "Last 24 hours", "Last 7 days", "All time"])
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if not st.session_state.scraping_active:
            if st.button("🚀 Start Scraping", width='stretch'):
                st.session_state.scraping_active = True
                st.rerun()
        else:
            if st.button("🛑 Stop Scraping", width='stretch'):
                st.session_state.scraping_active = False
                st.rerun()
    
    placeholder = st.empty()
    
    if st.session_state.scraping_active:
        while st.session_state.scraping_active:
            with placeholder.container():
                try:
                    client = MongoClient(mongo_uri)
                    db = client[db_name]
                    collection = db[col_name]
                    
                    # Fetch data based on time range
                    if time_range == "Last 100 records":
                        cursor = collection.find().sort("timestamp", -1).limit(100)
                    else:
                        time_filter = {}
                        if time_range == "Last hour":
                            time_filter = {"timestamp": {"$gte": datetime.now() - timedelta(hours=1)}}
                        elif time_range == "Last 24 hours":
                            time_filter = {"timestamp": {"$gte": datetime.now() - timedelta(days=1)}}
                        elif time_range == "Last 7 days":
                            time_filter = {"timestamp": {"$gte": datetime.now() - timedelta(days=7)}}
                        cursor = collection.find(time_filter).sort("timestamp", -1)
                    
                    data = list(cursor)
                    
                    if not data:
                        st.info(f"Waiting for data in {db_name}.{col_name}...")
                     else:
                        # ── Schema normalisation ───────────────────────────
                        # Real schema:
                        #   usage.gemini.{agent_1_call_1, agent_1_call_2?, agent_2}
                        #   usage.cohere.{embed, rerank}
                        #   metadata.username
                        raw_df = pd.DataFrame(data)

                        def _parse_row(doc):
                            gemini = doc.get('usage', {}).get('gemini', {})
                            cohere = doc.get('usage', {}).get('cohere', {})

                            # Aggregate Gemini tokens across all agents
                            g_input, g_output = 0, 0
                            for agent_val in gemini.values():
                                if isinstance(agent_val, dict):
                                    g_input  += agent_val.get('input_tokens',  0) or agent_val.get('prompt_token_count',     0)
                                    g_output += agent_val.get('output_tokens', 0) or agent_val.get('candidates_token_count', 0)

                            c_embed  = cohere.get('embed',  {}).get('input_tokens', 0)
                            c_rerank = cohere.get('rerank', {}).get('search_units', 0)

                            return {
                                'gemini_input':  g_input,
                                'gemini_output': g_output,
                                'cohere_embed':  c_embed,
                                'cohere_rerank': c_rerank,
                                'cohere_total':  c_embed + c_rerank,
                                'gemini_usage':  gemini,  # kept for agent-level breakdown tab
                            }

                        df = pd.DataFrame([_parse_row(d) for d in data])

                        # Pull top-level fields from raw_df
                        for field in ['user_id', 'query', 'thread_id', 'timestamp', 'latency_seconds']:
                            if field in raw_df.columns:
                                df[field] = raw_df[field].values
                            else:
                                df[field] = 0 if field == 'latency_seconds' else 'Unknown'

                        # username lives in metadata
                        if 'metadata' in raw_df.columns:
                            df['username'] = raw_df['metadata'].apply(
                                lambda m: m.get('username', 'Unknown') if isinstance(m, dict) else 'Unknown'
                            ).values
                        elif 'username' in raw_df.columns:
                            df['username'] = raw_df['username'].values
                        else:
                            df['username'] = 'Unknown'

                        # Convert timestamp to datetime
                        if 'timestamp' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])


                        
                        # ========== EXECUTIVE SUMMARY ==========
                        st.header("📊 Executive Summary")
                        
                        # Calculate key metrics
                        total_queries = len(df)
                        unique_users = df['username'].nunique()
                        unique_threads = df['thread_id'].nunique()
                        avg_latency = df['latency_seconds'].mean()
                        
                        
                        total_gemini_input = df['gemini_input'].sum()
                        total_gemini_output = df['gemini_output'].sum()
                        total_cohere_embed = df['cohere_embed'].sum()
                        total_cohere_rerank = df['cohere_rerank'].sum()
                        
                        
                        m1, m2, m3, m4, m5, m6 = st.columns(6)
                        with m1:
                            st.metric("Total Queries", f"{total_queries:,}")
                        with m2:
                            st.metric("Unique Users", f"{unique_users}")
                        with m3:
                            st.metric("Active Threads", f"{unique_threads}")
                        with m4:
                            st.metric("Avg Latency", f"{avg_latency:.3f}s")
                        with m6:
                            st.metric("Last Update", datetime.now().strftime("%H:%M:%S"))
                        
                        
                            # ========== PERFORMANCE INSIGHTS ==========
                        st.markdown("---")
                        st.header("⚡ Performance Insights")
                        
                        perf_tab1, perf_tab2, perf_tab3 = st.tabs([
                            "🚀 Latency Analysis", 
                            "📈 Token Efficiency", 
                            "🎯 Query Patterns"
                        ])
                        
                        with perf_tab1:
                            col_perf1, col_perf2 = st.columns(2)
                            
                            with col_perf1:
                                # Latency distribution
                                fig_latency_dist = px.histogram(
                                    df,
                                    x='latency_seconds',
                                    nbins=50,
                                    title='Latency Distribution',
                                    labels={'latency_seconds': 'Latency (seconds)', 'count': 'Frequency'}
                                )
                                fig_latency_dist.add_vline(
                                    x=df['latency_seconds'].mean(),
                                    line_dash="dash",
                                    line_color="red",
                                    annotation_text=f"Mean: {df['latency_seconds'].mean():.3f}s"
                                )
                                st.plotly_chart(fig_latency_dist, width='stretch')
                            
                            with col_perf2:
                                # Latency by user
                                user_latency = df.groupby('username')['latency_seconds'].agg(['mean', 'min', 'max']).reset_index()
                                fig_user_latency = go.Figure()
                                fig_user_latency.add_trace(go.Bar(
                                    name='Mean',
                                    x=user_latency['username'],
                                    y=user_latency['mean'],
                                    marker_color='#4285F4'
                                ))
                                fig_user_latency.update_layout(
                                    title='Average Latency by User',
                                    yaxis_title='Latency (seconds)',
                                    xaxis_title='User'
                                )
                                st.plotly_chart(fig_user_latency, width='stretch')
                            
                            # Latency percentiles
                            percentiles = df['latency_seconds'].quantile([0.5, 0.75, 0.9, 0.95, 0.99])
                            st.subheader("📊 Latency Percentiles")
                            perc_cols = st.columns(5)
                            for i, (perc, val) in enumerate(percentiles.items()):
                                with perc_cols[i]:
                                    st.metric(f"P{int(perc*100)}", f"{val:.3f}s")
                        
                        with perf_tab2:
                            col_eff1, col_eff2 = st.columns(2)
                            
                            with col_eff1:
                                # Tokens per query
                                df['total_tokens'] = df['gemini_input'] + df['gemini_output'] + df['cohere_total']
                                avg_tokens_per_query = df['total_tokens'].mean()
                                
                                st.metric("Avg Tokens/Query", f"{avg_tokens_per_query:.0f}")
                                
                                fig_tokens_dist = px.box(
                                    df,
                                    y='total_tokens',
                                    title='Token Usage Distribution per Query',
                                    labels={'total_tokens': 'Total Tokens'}
                                )
                                st.plotly_chart(fig_tokens_dist, width='stretch')
                            
                            with col_eff2:
                                # Input/Output ratio
                                df['io_ratio'] = df['gemini_output'] / (df['gemini_input'] + 1)  # +1 to avoid division by zero
                                
                                fig_io_ratio = px.scatter(
                                    df,
                                    x='gemini_input',
                                    y='gemini_output',
                                    color='username',
                                    title='Input vs Output Tokens',
                                    labels={'gemini_input': 'Input Tokens', 'gemini_output': 'Output Tokens'},
                                    hover_data=['query', 'latency_seconds']
                                )
                                fig_io_ratio.add_shape(
                                    type="line",
                                    x0=0, y0=0,
                                    x1=df['gemini_input'].max(),
                                    y1=df['gemini_input'].max(),
                                    line=dict(color="gray", dash="dash")
                                )
                                st.plotly_chart(fig_io_ratio, width='stretch')
                            
                            # Efficiency metrics
                            st.subheader("⚙️ Efficiency Metrics")
                            eff_col1, eff_col2, eff_col3, eff_col4 = st.columns(4)
                            
                            with eff_col1:
                                tokens_per_second = df['total_tokens'].sum() / df['latency_seconds'].sum()
                                st.metric("Tokens/Second", f"{tokens_per_second:.0f}")
                            
                            with eff_col2:
                                avg_io_ratio = df['io_ratio'].mean()
                                st.metric("Avg Output/Input Ratio", f"{avg_io_ratio:.2f}")
                            
                            with eff_col3:
                                cost_per_query = estimated_cost / total_queries if total_queries > 0 else 0
                                st.metric("Cost/Query", f"${cost_per_query:.5f}")
                            
                            with eff_col4:
                                tokens_per_dollar = (df['total_tokens'].sum() / estimated_cost) if estimated_cost > 0 else 0
                                st.metric("Tokens/$", f"{tokens_per_dollar:.0f}")
                        
                        with perf_tab3:
                            # Query length analysis
                            if 'query' in df.columns:
                                df['query_length'] = df['query'].str.len()
                                
                                col_pat1, col_pat2 = st.columns(2)
                                
                                with col_pat1:
                                    fig_query_len = px.histogram(
                                        df,
                                        x='query_length',
                                        nbins=30,
                                        title='Query Length Distribution',
                                        labels={'query_length': 'Query Length (characters)'}
                                    )
                                    st.plotly_chart(fig_query_len, width='stretch')
                                
                                with col_pat2:
                                    # Correlation: query length vs latency
                                    fig_corr = px.scatter(
                                        df,
                                        x='query_length',
                                        y='latency_seconds',
                                        color='username',
                                        title='Query Length vs Latency',
                                        labels={'query_length': 'Query Length', 'latency_seconds': 'Latency (s)'},
                                        opacity=0.6
                                    )
                                    
                                    # Add a simple correlation line if we have enough data
                                    if len(df) > 1:
                                        from numpy.polynomial import Polynomial
                                        # Filter out any NaN values
                                        valid_data = df[['query_length', 'latency_seconds']].dropna()
                                        if len(valid_data) > 1:
                                            x_data = valid_data['query_length'].values
                                            y_data = valid_data['latency_seconds'].values
                                            
                                            # Fit a simple linear regression
                                            p = Polynomial.fit(x_data, y_data, 1)
                                            x_trend = np.linspace(x_data.min(), x_data.max(), 100)
                                            y_trend = p(x_trend)
                                            
                                            # Add trend line
                                            fig_corr.add_trace(go.Scatter(
                                                x=x_trend,
                                                y=y_trend,
                                                mode='lines',
                                                name='Trend',
                                                line=dict(color='red', dash='dash', width=2)
                                            ))
                                            
                                            # Calculate correlation coefficient
                                            corr = np.corrcoef(x_data, y_data)[0, 1]
                                            fig_corr.add_annotation(
                                                text=f'Correlation: {corr:.3f}',
                                                xref="paper", yref="paper",
                                                x=0.05, y=0.95,
                                                showarrow=False,
                                                bgcolor="white",
                                                bordercolor="black",
                                                borderwidth=1
                                            )
                                    
                                    st.plotly_chart(fig_corr, width='stretch')
                            
                            # Most active hours
                            if 'timestamp' in df.columns:
                                df['hour_of_day'] = df['timestamp'].dt.hour
                                hourly_activity = df.groupby('hour_of_day').size().reset_index(name='query_count')
                                
                                fig_hourly = px.bar(
                                    hourly_activity,
                                    x='hour_of_day',
                                    y='query_count',
                                    title='Query Volume by Hour of Day',
                                    labels={'hour_of_day': 'Hour', 'query_count': 'Number of Queries'}
                                )
                                st.plotly_chart(fig_hourly, width='stretch')
                        
                        # ========== PROVIDER BREAKDOWN ==========
                        st.markdown("---")
                        st.header("🔬 Detailed Provider Breakdown")
                        
                        provider_tab1, provider_tab2 = st.tabs(["🤖 Gemini Breakdown", "🔷 Cohere Breakdown"])
                        
                        with provider_tab1:
                            st.subheader("Gemini Agent-Level Usage")
                            
                            if 'gemini_usage' in df.columns:
                                gemini_data = []
                                for idx, row in df.iterrows():
                                    usage = row.get('gemini_usage', {})
                                    if isinstance(usage, dict):
                                        record = {
                                            'timestamp': row.get('timestamp'),
                                            'username': row.get('username', 'Unknown'),
                                            'query': row.get('query', '')[:50] + '...' if len(str(row.get('query', ''))) > 50 else row.get('query', ''),
                                            'agent1_call1_input': usage.get('agent_1_call_1', {}).get('input_tokens', 0),
                                            'agent1_call1_output': usage.get('agent_1_call_1', {}).get('output_tokens', 0),
                                            'agent1_call2_input': usage.get('agent_1_call_2', {}).get('input_tokens', 0),
                                            'agent1_call2_output': usage.get('agent_1_call_2', {}).get('output_tokens', 0),
                                            'agent2_input': usage.get('agent_2', {}).get('prompt_token_count', 0),
                                            'agent2_output': usage.get('agent_2', {}).get('candidates_token_count', 0),
                                        }
                                        gemini_data.append(record)
                                
                                if gemini_data:
                                    gemini_df = pd.DataFrame(gemini_data)
                                    
                                    # Summary metrics
                                    g1, g2, g3 = st.columns(3)
                                    with g1:
                                        total_a1c1 = gemini_df['agent1_call1_input'].sum() + gemini_df['agent1_call1_output'].sum()
                                        st.metric("Agent 1 Call 1 Total", f"{total_a1c1:,.0f}")
                                    with g2:
                                        total_a1c2 = gemini_df['agent1_call2_input'].sum() + gemini_df['agent1_call2_output'].sum()
                                        st.metric("Agent 1 Call 2 Total", f"{total_a1c2:,.0f}")
                                    with g3:
                                        total_a2 = gemini_df['agent2_input'].sum() + gemini_df['agent2_output'].sum()
                                        st.metric("Agent 2 Total", f"{total_a2:,.0f}")
                                    
                                    col_g1, col_g2 = st.columns(2)
                                    
                                    with col_g1:
                                        # Stacked bar chart
                                        agent_totals = {
                                            'Agent 1 Call 1': {
                                                'Input': gemini_df['agent1_call1_input'].sum(),
                                                'Output': gemini_df['agent1_call1_output'].sum()
                                            },
                                            'Agent 1 Call 2': {
                                                'Input': gemini_df['agent1_call2_input'].sum(),
                                                'Output': gemini_df['agent1_call2_output'].sum()
                                            },
                                            'Agent 2': {
                                                'Input': gemini_df['agent2_input'].sum(),
                                                'Output': gemini_df['agent2_output'].sum()
                                            },
                                        }
                                        
                                        fig_gemini_stacked = go.Figure()
                                        agents = list(agent_totals.keys())
                                        inputs = [agent_totals[a]['Input'] for a in agents]
                                        outputs = [agent_totals[a]['Output'] for a in agents]
                                        
                                        fig_gemini_stacked.add_trace(go.Bar(
                                            name='Input Tokens',
                                            x=agents,
                                            y=inputs,
                                            marker_color='#4285F4'
                                        ))
                                        fig_gemini_stacked.add_trace(go.Bar(
                                            name='Output Tokens',
                                            x=agents,
                                            y=outputs,
                                            marker_color='#34A853'
                                        ))
                                        fig_gemini_stacked.update_layout(
                                            barmode='stack',
                                            title='Gemini Token Usage by Agent',
                                            yaxis_title='Total Tokens',
                                            height=400
                                        )
                                        st.plotly_chart(fig_gemini_stacked, width='stretch')
                                    
                                    with col_g2:
                                        # Pie chart
                                        pie_data = {
                                            'Agent 1 Call 1': total_a1c1,
                                            'Agent 1 Call 2': total_a1c2,
                                            'Agent 2': total_a2
                                        }
                                        fig_gemini_pie = px.pie(
                                            values=list(pie_data.values()),
                                            names=list(pie_data.keys()),
                                            title='Token Distribution by Agent',
                                            color_discrete_sequence=['#4285F4', '#EA4335', '#FBBC04']
                                        )
                                        fig_gemini_pie.update_layout(height=400)
                                        st.plotly_chart(fig_gemini_pie, width='stretch')
                                    
                                    # Input vs Output ratio per agent
                                    st.subheader("📊 Input vs Output Token Ratio")
                                    ratio_data = []
                                    for agent_name, values in agent_totals.items():
                                        if values['Input'] > 0:
                                            ratio = values['Output'] / values['Input']
                                        else:
                                            ratio = 0
                                        ratio_data.append({
                                            'Agent': agent_name,
                                            'Input': values['Input'],
                                            'Output': values['Output'],
                                            'Ratio': ratio
                                        })
                                    
                                    ratio_df = pd.DataFrame(ratio_data)
                                    fig_ratio = px.bar(
                                        ratio_df,
                                        x='Agent',
                                        y='Ratio',
                                        title='Output/Input Token Ratio by Agent',
                                        color='Ratio',
                                        color_continuous_scale='Blues'
                                    )
                                    st.plotly_chart(fig_ratio, width='stretch')
                                    
                                    # Detailed table
                                    st.subheader("📋 Gemini Usage Details")
                                    display_cols = ['username', 'query', 'agent1_call1_input', 'agent1_call1_output',
                                                  'agent1_call2_input', 'agent1_call2_output', 'agent2_input', 'agent2_output']
                                    st.dataframe(gemini_df[display_cols].head(15), width='stretch')
                                else:
                                    st.info("No Gemini usage data available")
                            else:
                                st.info("No Gemini usage data in collection")
                        
                        with provider_tab2:
                            st.subheader("Cohere Embed & Rerank Breakdown")
                            
                            if 'cohere_usage' in df.columns:
                                cohere_data = []
                                for idx, row in df.iterrows():
                                    usage = row.get('cohere_usage', {})
                                    if isinstance(usage, dict):
                                        record = {
                                            'timestamp': row.get('timestamp'),
                                            'username': row.get('username', 'Unknown'),
                                            'query': row.get('query', '')[:50] + '...' if len(str(row.get('query', ''))) > 50 else row.get('query', ''),
                                            'embed_input_tokens': usage.get('embed', {}).get('input_tokens', 0),
                                            'embed_search_units': usage.get('embed', {}).get('search_units', 0),
                                            'rerank_input_tokens': usage.get('rerank', {}).get('input_tokens', 0),
                                            'rerank_search_units': usage.get('rerank', {}).get('search_units', 0),
                                        }
                                        cohere_data.append(record)
                                
                                if cohere_data:
                                    cohere_df = pd.DataFrame(cohere_data)
                                    
                                    # Summary metrics
                                    c1, c2, c3, c4 = st.columns(4)
                                    with c1:
                                        total_embed_tokens = cohere_df['embed_input_tokens'].sum()
                                        st.metric("Embed Tokens", f"{total_embed_tokens:,.0f}")
                                    with c2:
                                        total_embed_units = cohere_df['embed_search_units'].sum()
                                        st.metric("Embed Search Units", f"{total_embed_units:,.0f}")
                                    with c3:
                                        total_rerank_tokens = cohere_df['rerank_input_tokens'].sum()
                                        st.metric("Rerank Tokens", f"{total_rerank_tokens:,.0f}")
                                    with c4:
                                        total_rerank_units = cohere_df['rerank_search_units'].sum()
                                        st.metric("Rerank Search Units", f"{total_rerank_units:,.0f}")
                                    
                                    col_c1, col_c2 = st.columns(2)
                                    
                                    with col_c1:
                                        # Bar chart
                                        fig_cohere_bar = go.Figure()
                                        operations = ['Embed', 'Rerank']
                                        tokens = [total_embed_tokens, total_rerank_tokens]
                                        units = [total_embed_units, total_rerank_units]
                                        
                                        fig_cohere_bar.add_trace(go.Bar(
                                            name='Input Tokens',
                                            x=operations,
                                            y=tokens,
                                            marker_color='#FF7F50'
                                        ))
                                        fig_cohere_bar.add_trace(go.Bar(
                                            name='Search Units',
                                            x=operations,
                                            y=units,
                                            marker_color='#20B2AA'
                                        ))
                                        fig_cohere_bar.update_layout(
                                            barmode='group',
                                            title='Cohere Usage: Embed vs Rerank',
                                            yaxis_title='Count',
                                            height=400
                                        )
                                        st.plotly_chart(fig_cohere_bar, width='stretch')
                                    
                                    with col_c2:
                                        # Pie chart
                                        if total_embed_tokens + total_rerank_tokens > 0:
                                            fig_cohere_pie = px.pie(
                                                values=[total_embed_tokens, total_rerank_tokens],
                                                names=['Embed Tokens', 'Rerank Tokens'],
                                                title='Token Distribution: Embed vs Rerank',
                                                color_discrete_sequence=['#FF7F50', '#20B2AA']
                                            )
                                            fig_cohere_pie.update_layout(height=400)
                                            st.plotly_chart(fig_cohere_pie, width='stretch')
                                        else:
                                            st.info("No token data to display")
                                    
                                    # User-wise breakdown
                                    st.subheader("👤 User-wise Cohere Usage")
                                    user_cohere = cohere_df.groupby('username').agg({
                                        'embed_input_tokens': 'sum',
                                        'embed_search_units': 'sum',
                                        'rerank_input_tokens': 'sum',
                                        'rerank_search_units': 'sum'
                                    }).reset_index()
                                    
                                    fig_user_cohere = px.bar(
                                        user_cohere,
                                        x='username',
                                        y=['embed_input_tokens', 'rerank_input_tokens'],
                                        title='Cohere Token Usage by User',
                                        barmode='group',
                                        labels={'value': 'Tokens', 'variable': 'Operation'},
                                        color_discrete_map={
                                            'embed_input_tokens': '#FF7F50',
                                            'rerank_input_tokens': '#20B2AA'
                                        }
                                    )
                                    st.plotly_chart(fig_user_cohere, width='stretch')
                                    
                                    # Detailed table
                                    st.subheader("📋 Cohere Usage Details")
                                    display_cols = ['username', 'query', 'embed_input_tokens', 'embed_search_units',
                                                  'rerank_input_tokens', 'rerank_search_units']
                                    st.dataframe(cohere_df[display_cols].head(15), width='stretch')
                                else:
                                    st.info("No Cohere usage data available")
                            else:
                                st.info("No Cohere usage data in collection")
                        # ========== RAW DATA ==========
                        st.markdown("---")
                        st.header("📝 Recent Logs")
                        
                        cols_to_plot = [c for c in ['gemini_input', 'gemini_output', 'cohere_total', 'cohere_embed', 'cohere_rerank'] if c in df.columns]
                        cols_to_show = [c for c in ['timestamp', 'username', 'thread_id', 'query', 'latency_seconds'] + cols_to_plot if c in df.columns]
                        
                        st.dataframe(df[cols_to_show].head(20), width='stretch')
                
                except Exception as e:
                    st.error(f"Error fetching data: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                
                time.sleep(10)
                st.rerun()
    else:
        st.info("Click 'Start Scraping' to begin monitoring MongoDB telemetry data.")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Performance Analysis", "Telemetry Analysis"])

if page == "Performance Analysis":
    performance_analysis_page()
else:
    telemetry_analysis_page()