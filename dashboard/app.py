import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="TOEIC Recommendation Dashboard", page_icon="📚", layout="wide")
API_URL = "http://localhost:8000"

# Initialize session state
if 'selected_user' not in st.session_state:
    st.session_state.selected_user = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = {
        'hybrid': [],
        'content': [],
        'collaborative': [],
        'advanced_hybrid': []
    }
if 'top_k' not in st.session_state:
    st.session_state.top_k = 10

st.markdown("""
<style>
.recommendation-card {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 10px;
    background: #f9f9f9;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.score-badge {
    background: #4CAF50;
    padding: 4px 8px;
    color: white;
    border-radius: 4px;
    font-size: 12px;
}
.subject-badge {
    background: #FF9800;
    color: white;
    border-radius: 4px;
    font-size: 10px;
    padding: 3px 7px;
    margin-right: 5px;
}
.part-badge {
    background: #2196F3;
    color: white;
    border-radius: 4px;
    font-size: 10px;
    padding: 3px 7px;
    margin-right: 5px;
}
</style>
""", unsafe_allow_html=True)

def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException as e:
        st.error(f"API health check failed: {str(e)}")
        return False

def fetch_users():
    try:
        response = requests.get(f"{API_URL}/users", timeout=5)
        if response.status_code == 200:
            return response.json().get("users", [])
        else:
            st.warning(f"Failed to fetch users: {response.status_code} {response.text}")
            return []
    except requests.RequestException as e:
        st.error(f"Error fetching users: {str(e)}")
        return []

def fetch_user_history(user_id):
    try:
        response = requests.get(f"{API_URL}/user/{user_id}/history", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            st.warning(f"Failed to fetch history for {user_id}: {response.status_code} {response.text}")
            return {"history": [], "total_interactions": 0}
    except requests.RequestException as e:
        st.error(f"Error fetching history for {user_id}: {str(e)}")
        return {"history": [], "total_interactions": 0}

def fetch_recommendations(user_id, rec_type, n):
    try:
        response = requests.post(f"{API_URL}/recommendations", json={
            "user_id": user_id,
            "n_recommendations": n,
            "recommendation_type": rec_type
        }, timeout=30)  # Keep your increased timeout
        if response.status_code == 200:
            return response.json().get("recommendations", [])
        else:
            st.warning(f"Failed to fetch {rec_type} recommendations: {response.status_code} {response.text}")
            return []
    except requests.RequestException as e:
        st.error(f"Error fetching {rec_type} recommendations: {str(e)}")
        return []

def display_user_metrics(history):
    df = pd.DataFrame(history.get("history", []))
    if df.empty:
        st.info("No user interaction data available.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Interactions", len(df))
    col2.metric("Accuracy", f"{df['is_correct'].mean() * 100:.1f}%")
    col3.metric("Avg Response Time", f"{df['elapsed_time'].mean() / 1000:.1f}s")
    col4.metric("TOEIC Parts Practiced", df['part'].nunique())

    st.subheader("📈 Accuracy by TOEIC Part")
    acc = df.groupby('part')['is_correct'].mean() * 100
    fig1 = px.bar(x=acc.index, y=acc.values, labels={'x': 'Part', 'y': 'Accuracy (%)'},
                  color=acc.index, color_discrete_sequence=px.colors.qualitative.Plotly)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("⏱️ Response Time Distribution")
    fig2 = px.histogram(df, x=df['elapsed_time'] / 1000, nbins=20,
                        labels={'x': 'Seconds', 'y': 'Count'},
                        color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig2, use_container_width=True)

def render_recommendation_card(rec, idx):
    with st.container():
        st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**{idx + 1}. {rec.get('title', 'Untitled')}**")
            st.markdown(f"<span class='part-badge'>{rec.get('part', 'N/A')}</span>", unsafe_allow_html=True)
            if rec.get("subjects"):
                tags_html = "".join(f"<span class='subject-badge'>{s}</span>" for s in rec["subjects"])
                st.markdown(tags_html, unsafe_allow_html=True)
            duration = rec.get('duration_minutes')
            duration_display = f"{duration:.1f} mins" if isinstance(duration, (int, float)) else "N/A"
            st.caption(f"⏱️ Estimated Time: {duration_display}")
        with col2:
            score = rec.get("score")
            score_display = f"{score:.3f}" if isinstance(score, (int, float)) else "N/A"
            st.markdown(f"<span class='score-badge'>Score: {score_display}</span>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def display_recommendation_list(title, recs):
    st.subheader(title)
    if not recs:
        st.info(f"No {title.lower()} available. Click 'Generate' to fetch recommendations.")
        return
    for idx, rec in enumerate(recs):
        render_recommendation_card(rec, idx)


def display_overlap_distribution(hybrid, content, collab, advanced):
    def extract_id(rec):
        # Normalize all IDs to strings, prefer `item_id`, fallback to others
        return str(
            rec.get("item_id") or
            rec.get("bundle_id") or
            rec.get("lecture_id") or ""
        ).strip().lower()

    def get_unique_ids(recs):
        return set(extract_id(r) for r in recs if extract_id(r))

    # Get distinct IDs
    h_ids = get_unique_ids(hybrid)
    c_ids = get_unique_ids(content)
    b_ids = get_unique_ids(collab)
    a_ids = get_unique_ids(advanced)

    # Set-based overlap computation
    overlap_df = pd.DataFrame({
        "Comparison": [
            "Hybrid ∩ Content",
            "Hybrid ∩ Collaborative",
            "Hybrid ∩ Advanced",
            "Content ∩ Collaborative",
            "Content ∩ Advanced",
            "Collaborative ∩ Advanced",
            "All Four"
        ],
        "Overlap Count": [
            len(h_ids & c_ids),
            len(h_ids & b_ids),
            len(h_ids & a_ids),
            len(c_ids & b_ids),
            len(c_ids & a_ids),
            len(b_ids & a_ids),
            len(h_ids & c_ids & b_ids & a_ids),
        ]
    }).sort_values("Overlap Count", ascending=False)

    st.subheader("🔗 Recommendation Overlap")
    st.dataframe(overlap_df, use_container_width=True)

    # Part Distribution Chart
    part_records = []
    for source, group in [
        ("Hybrid", hybrid),
        ("Content", content),
        ("Collaborative", collab),
        ("Advanced", advanced)
    ]:
        for rec in group:
            part = rec.get("part")
            if part:
                part_records.append({"Type": source, "Part": part})
    if part_records:
        df_parts = pd.DataFrame(part_records)
        fig = px.histogram(df_parts, x="Part", color="Type", barmode="group",
                           title="TOEIC Part Distribution by Strategy")
        st.subheader("📊 Part Distribution")
        st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("📚 TOEIC Recommendation Dashboard")

    if not check_api_health():
        st.error("⚠️ Could not connect to the API. Please ensure the API server is running (`uvicorn api.main:app --reload`).")
        st.stop()

    with st.sidebar:
        st.header("⚙️ Settings")
        users = fetch_users()
        if not users:
            st.error("No users available. Ensure data is loaded in the API.")
            st.stop()
        selected_user = st.selectbox("Select User", users)
        top_k = st.slider("Number of Recommendations", min_value=5, max_value=20, value=10)

        # Clear recommendations if user or slider value changes
        if (selected_user != st.session_state.selected_user) or (top_k != st.session_state.top_k):
            st.session_state.selected_user = selected_user
            st.session_state.top_k = top_k
            st.session_state.recommendations = {
                'hybrid': [],
                'content': [],
                'collaborative': [],
                'advanced_hybrid': []
            }
        else:
            st.session_state.top_k = top_k


    if not selected_user:
        return

    st.header(f"👤 Insights for {selected_user}")
    user_history = fetch_user_history(selected_user)
    display_user_metrics(user_history)

    st.markdown("---")
    st.header(f"🎓 Personalized Recommendations for {selected_user}")

    tab1, tab2, tab3, tab4 = st.tabs(["🔀 Hybrid", "📋 Content-Based", "👥 Collaborative", "🌟 Advanced Hybrid"])

    with tab1:
        if st.button("Generate Hybrid Recommendations"):
            with st.spinner("Fetching hybrid recommendations..."):
                st.session_state.recommendations['hybrid'] = fetch_recommendations(selected_user, "hybrid", st.session_state.top_k)
        display_recommendation_list("Hybrid Recommendations", st.session_state.recommendations['hybrid'])

    with tab2:
        if st.button("Generate Content-Based Recommendations"):
            with st.spinner("Fetching content-based recommendations..."):
                st.session_state.recommendations['content'] = fetch_recommendations(selected_user, "content", st.session_state.top_k)
        display_recommendation_list("Content-Based Recommendations", st.session_state.recommendations['content'])

    with tab3:
        if st.button("Generate Collaborative Recommendations"):
            with st.spinner("Fetching collaborative recommendations..."):
                st.session_state.recommendations['collaborative'] = fetch_recommendations(selected_user, "collaborative", st.session_state.top_k)
        display_recommendation_list("Collaborative Recommendations", st.session_state.recommendations['collaborative'])

    with tab4:
        if st.button("Generate Advanced Hybrid Recommendations"):
            with st.spinner("Fetching advanced hybrid recommendations..."):
                st.session_state.recommendations['advanced_hybrid'] = fetch_recommendations(selected_user, "advanced_hybrid", st.session_state.top_k)
        display_recommendation_list("Advanced Hybrid Recommendations", st.session_state.recommendations['advanced_hybrid'])

    st.markdown("---")
    display_overlap_distribution(
        st.session_state.recommendations['hybrid'],
        st.session_state.recommendations['content'],
        st.session_state.recommendations['collaborative'],
        st.session_state.recommendations['advanced_hybrid']
    )

if __name__ == "__main__":
    main()