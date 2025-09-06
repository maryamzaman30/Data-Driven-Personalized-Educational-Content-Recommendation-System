# =========================================================
# File: dashboard/app.py
# Description:
#   Streamlit-based TOEIC Recommendation Dashboard.
#   Connects to FastAPI backend to fetch:
#       - User list
#       - User history & metrics
#       - Recommendations (Hybrid, Content, Collaborative, Advanced Hybrid)
#   Provides visualizations & overlap analysis.
# =========================================================

# Import Streamlit for building the web interface
import streamlit as st
# Import requests to communicate with the FastAPI backend
import requests
# Import pandas for data manipulation and display
import pandas as pd
# Import Plotly for interactive visualizations
import plotly.express as px

# =========================================================
# 1. Page Configuration
# =========================================================

# Set up the Streamlit app's layout and metadata
st.set_page_config(
    page_title="TOEIC Recommendation", # Title shown in browser tab
    page_icon="📚",       # Icon shown in browser tab
    layout="wide"         # Use full-width layout for better visuals
)

# Define the base URL for the FastAPI backend
API_URL = "http://localhost:8000"

# =========================================================
# 2. Session State Initialization
# =========================================================

# Initialize selected user if not already set
if 'selected_user' not in st.session_state:
    st.session_state.selected_user = None  # Tracks which user is currently selected

# Initialize empty recommendation lists for each method
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = {
        'hybrid': [],            # NCF-based hybrid recommendations
        'content': [],           # SBERT-based content recommendations
        'collaborative': [],     # SVD-based collaborative recommendations
        'advanced_hybrid': []    # Meta-learned hybrid recommendations
    }

# Set default number of recommendations to show
if 'top_k' not in st.session_state:
    st.session_state.top_k = 10

# =========================================================
# 3. Custom CSS Styling
# =========================================================

# Inject custom CSS into the Streamlit app to style recommendation cards & badges
st.markdown("""
<style>
/* Style for each recommendation card */
.recommendation-card {
    border: 1px solid #ddd;         /* Light border around the card */
    border-radius: 8px;             /* Rounded corners */
    padding: 3px;                   /* Inner spacing */
    margin-bottom: 10px;            /* Space between cards */
    background: #f9f9f9;            /* Light gray background */
}

/* Style for score badge  */
.score-badge {
    background: #4CAF50;            /* Green background */
    padding: 4px 8px;               /* Inner spacing */
    color: white;                   /* White text */
    border-radius: 4px;             /* Rounded corners */
    font-size: 12px;                /* Small font size */
}

/* Style for subject badge (e.g. grammar, vocabulary) */
.subject-badge {
    background: #FF9800;            /* Orange background */
    color: white;
    border-radius: 4px;
    font-size: 10px;
    padding: 3px 7px;
    margin-right: 5px;              /* Space between badges */
}

/* Style for TOEIC part badge (e.g. Part 1, Part 2) */
.part-badge {
    background: #2196F3;            /* Blue background */
    color: white;
    border-radius: 4px;
    font-size: 10px;
    padding: 3px 7px;
    margin-right: 5px;
}
</style>
""", unsafe_allow_html=True)  # Allow raw HTML/CSS injection

# =========================================================
# 4. Backend API Helper Functions
# =========================================================

def check_api_health():
    """Ping the backend to verify it's reachable and responsive."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200  # True if healthy
    except requests.RequestException as e:
        st.error(f"API health check failed: {str(e)}")
        return False

def fetch_users():
    """Retrieve a list of all registered users from the backend."""
    try:
        response = requests.get(f"{API_URL}/users", timeout=5)
        if response.status_code == 200:
            return response.json().get("users", [])  # Return user list or empty
        else:
            st.warning(f"Failed to fetch users: {response.status_code} {response.text}")
            return []
    except requests.RequestException as e:
        st.error(f"Error fetching users: {str(e)}")
        return []

def fetch_user_history(user_id):
    """Get interaction history for a specific user by ID."""
    try:
        response = requests.get(f"{API_URL}/user/{user_id}/history", timeout=5)
        if response.status_code == 200:
            return response.json() # Expecting keys: history, total_interactions
        else:
            st.warning(f"Failed to fetch history for {user_id}: {response.status_code} {response.text}")
            return {"history": [], "total_interactions": 0}
    except requests.RequestException as e:
        st.error(f"Error fetching history for {user_id}: {str(e)}")
        return {"history": [], "total_interactions": 0}

def fetch_recommendations(user_id, rec_type, n):
    """
    Request personalized recommendations for a user.

    Parameters:
    - user_id: str, the target user
    - rec_type: str, type of recommendation (e.g. 'toeic', 'grammar')
    - n: int, number of recommendations to fetch
    """
    try:
        response = requests.post(f"{API_URL}/recommendations", json={
            "user_id": user_id,
            "n_recommendations": n,
            "recommendation_type": rec_type
        }, timeout=30)  # Long timeout for heavy models
        if response.status_code == 200:
            return response.json().get("recommendations", [])
        else:
            st.warning(f"Failed to fetch {rec_type} recommendations: {response.status_code} {response.text}")
            return []
    except requests.RequestException as e:
        st.error(f"Error fetching {rec_type} recommendations: {str(e)}")
        return []

# =========================================================
# 5. User Metrics & Visualizations
# =========================================================

def display_user_metrics(history):
    """Visualize key performance metrics and interaction patterns for a selected user."""
    
    # Convert history list to DataFrame
    df = pd.DataFrame(history.get("history", []))
    if df.empty:
        st.info("No user interaction data available.")
        return

    # -------------------------
    # Display Key Summary Stats
    # -------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Interactions", len(df))  # Total number of attempts
    col2.metric("Accuracy", f"{df['is_correct'].mean() * 100:.1f}%")  # Correctness rate
    col3.metric("Avg Response Time", f"{df['elapsed_time'].mean() / 1000:.1f}s")  # Time in seconds
    col4.metric("TOEIC Parts Practiced", df['part'].nunique())  # Diversity of parts attempted

    # -------------------------
    # Accuracy by TOEIC Part
    # -------------------------
    st.subheader("📈 Accuracy by TOEIC Part")
    acc = df.groupby('part')['is_correct'].mean() * 100  # Grouped accuracy
    fig1 = px.bar(
        x=acc.index,
        y=acc.values,
        labels={'x': 'Part', 'y': 'Accuracy (%)'},
        color=acc.index,
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    st.plotly_chart(fig1, use_container_width=True)

    # -------------------------
    # Response Time Distribution
    # -------------------------
    st.subheader("⏱️ Response Time Distribution")
    fig2 = px.histogram(
        df,
        x=df['elapsed_time'] / 1000,  # Convert ms to seconds
        nbins=20,
        labels={'x': 'Seconds', 'y': 'Count'},
        color_discrete_sequence=['#636EFA']
    )
    st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# 6. Recommendation Display Helpers
# =========================================================

def render_recommendation_card(rec, idx):
    """Render a single styled recommendation card with metadata and score."""
    with st.container():
        # Begin custom-styled card block
        st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
        col1, col2 = st.columns([4, 1])  # Layout: main info vs score badge

        # -------------------------
        # Left Column: Title, Tags, Duration
        # -------------------------
        with col1:
            # Title with index
            st.markdown(f"**{idx + 1}. {rec.get('title', 'Untitled')}**")

            # TOEIC part badge
            st.markdown(
                f"<span class='part-badge'>{rec.get('part', 'N/A')}</span>",
                unsafe_allow_html=True
            )

            # Subject tags (e.g. grammar, listening)
            if rec.get("subjects"):
                tags_html = "".join(
                    f"<span class='subject-badge'>{s}</span>" for s in rec["subjects"]
                )
                st.markdown(tags_html, unsafe_allow_html=True)

            # Estimated duration
            duration = rec.get('duration_minutes')
            duration_display = f"{duration:.1f} mins" if isinstance(duration, (int, float)) else "N/A"
            st.caption(f"⏱️ Estimated Time: {duration_display}")

        # -------------------------
        # Right Column: Score Badge
        # -------------------------
        with col2:
            score = rec.get("score")
            score_display = f"{score:.3f}" if isinstance(score, (int, float)) else "N/A"
            st.markdown(
                f"<span class='score-badge'>Score: {score_display}</span>",
                unsafe_allow_html=True
            )

        # Close card block
        st.markdown('</div>', unsafe_allow_html=True)

def display_recommendation_list(title, recs):
    """Render a list of recommendation cards under a section title."""
    st.subheader(title)

    if not recs:
        st.info(f"No {title.lower()} available. Click 'Generate' to fetch recommendations.")
        return

    for idx, rec in enumerate(recs):
        render_recommendation_card(rec, idx)

# =========================================================
# 7. Overlap & Part Distribution Analysis
# =========================================================

def display_overlap_distribution(hybrid, content, collab, advanced):
    """Show overlap counts and TOEIC part distribution across recommendation strategies."""

    # -------------------------
    # Helper: Extract normalized unique ID from a recommendation
    # -------------------------
    def extract_id(rec):
        """
        Extract and normalize a unique identifier from a recommendation.
        Prioritizes 'item_id', then falls back to 'bundle_id' or 'lecture_id'.
        """
        return str(
            rec.get("item_id") or
            rec.get("bundle_id") or
            rec.get("lecture_id") or ""
        ).strip().lower()

    # -------------------------
    # Helper: Get set of unique IDs from a list of recommendations
    # -------------------------
    def get_unique_ids(recs):
        return set(extract_id(r) for r in recs if extract_id(r))

    # -------------------------
    # Compute Overlap Between Strategies
    # -------------------------
    h_ids = get_unique_ids(hybrid)
    c_ids = get_unique_ids(content)
    b_ids = get_unique_ids(collab)
    a_ids = get_unique_ids(advanced)

    # Create overlap summary table
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

    # Display overlap table
    st.subheader("🔗 Recommendation Overlap")
    st.dataframe(overlap_df, use_container_width=True)

    # -------------------------
    # Visualize Part Distribution by Strategy
    # -------------------------
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
        fig = px.histogram(
            df_parts,
            x="Part",
            color="Type",
            barmode="group",
            title="TOEIC Part Distribution by Strategy"
        )
        st.subheader("📊 Part Distribution")
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 8. Main App
# =========================================================

def main():
    """Main Streamlit app for TOEIC recommendation dashboard."""

    # -------------------------
    # App Title
    # -------------------------
    st.title("📚 Data-Driven Personalized Educational Content Recommendation System for TOEIC Preparation")

    # -------------------------
    # API Health Check
    # -------------------------
    if not check_api_health():
        st.error("⚠️ Could not connect to the API. Please ensure the API server is running (`uvicorn api.main:app --reload`).")
        st.stop()

    # -------------------------
    # Sidebar: User & Settings
    # -------------------------
    with st.sidebar:
        st.header("⚙️ Settings")

        # Fetch available users
        users = fetch_users()
        if not users:
            st.error("No users available. Ensure data is loaded in the API.")
            st.stop()

        # User selection and recommendation count
        selected_user = st.selectbox("Select User", users)
        top_k = st.slider("Number of Recommendations", min_value=5, max_value=20, value=10)

        # Reset recommendations if user or slider value changes
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
            st.session_state.top_k = top_k  # Update only if unchanged

    # -------------------------
    # Exit Early if No User Selected
    # -------------------------
    if not selected_user:
        return

    # -------------------------
    # User Metrics Section
    # -------------------------
    st.header(f"👤 Insights for {selected_user}")
    user_history = fetch_user_history(selected_user)
    display_user_metrics(user_history)

    st.markdown("---")

    # -------------------------
    # Recommendation Tabs
    # -------------------------
    st.header(f"🎓 Personalized Recommendations for {selected_user}")
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔀 Hybrid",
        "📋 Content-Based",
        "👥 Collaborative",
        "🌟 Advanced Hybrid"
    ])

    # Tab 1: Hybrid Recommendations
    with tab1:
        if st.button("Generate Hybrid Recommendations"):
            with st.spinner("Fetching hybrid recommendations..."):
                st.session_state.recommendations['hybrid'] = fetch_recommendations(
                    selected_user, "hybrid", st.session_state.top_k
                )
        display_recommendation_list("Hybrid Recommendations", st.session_state.recommendations['hybrid'])

    # Tab 2: Content-Based Recommendations
    with tab2:
        if st.button("Generate Content-Based Recommendations"):
            with st.spinner("Fetching content-based recommendations..."):
                st.session_state.recommendations['content'] = fetch_recommendations(
                    selected_user, "content", st.session_state.top_k
                )
        display_recommendation_list("Content-Based Recommendations", st.session_state.recommendations['content'])

    # Tab 3: Collaborative Recommendations
    with tab3:
        if st.button("Generate Collaborative Recommendations"):
            with st.spinner("Fetching collaborative recommendations..."):
                st.session_state.recommendations['collaborative'] = fetch_recommendations(
                    selected_user, "collaborative", st.session_state.top_k
                )
        display_recommendation_list("Collaborative Recommendations", st.session_state.recommendations['collaborative'])

    # Tab 4: Advanced Hybrid Recommendations
    with tab4:
        if st.button("Generate Advanced Hybrid Recommendations"):
            with st.spinner("Fetching advanced hybrid recommendations..."):
                st.session_state.recommendations['advanced_hybrid'] = fetch_recommendations(
                    selected_user, "advanced_hybrid", st.session_state.top_k
                )
        display_recommendation_list("Advanced Hybrid Recommendations", st.session_state.recommendations['advanced_hybrid'])

    # -------------------------
    # Overlap & Distribution Summary
    # -------------------------
    st.markdown("---")
    display_overlap_distribution(
        st.session_state.recommendations['hybrid'],
        st.session_state.recommendations['content'],
        st.session_state.recommendations['collaborative'],
        st.session_state.recommendations['advanced_hybrid']
    )

# =========================================================
# 9. Entry Point
# =========================================================

# Execute the main function only if this script is run directly
if __name__ == "__main__":
    main()