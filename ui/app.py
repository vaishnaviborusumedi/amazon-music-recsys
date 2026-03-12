import streamlit as st
import requests
import pandas as pd

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title = "🎸 Music Gear Recommender",
    page_icon  = "🎸",
    layout     = "wide"
)

API_URL = "http://localhost:8000"

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f0f0f; }
    .product-card {
        background: #1a1a2e;
        border: 1px solid #e94560;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        transition: transform 0.2s;
    }
    .product-title {
        color: #e94560;
        font-size: 15px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .product-brand {
        color: #a8a8b3;
        font-size: 12px;
    }
    .product-price {
        color: #00b4d8;
        font-size: 18px;
        font-weight: bold;
    }
    .score-badge {
        background: #e94560;
        color: white;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 12px;
    }
    .strategy-badge {
        background: #16213e;
        color: #00b4d8;
        border: 1px solid #00b4d8;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 12px;
    }
    .metric-box {
        background: #16213e;
        border-left: 4px solid #e94560;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────
st.markdown("# 🎸 Music Gear Recommendation System")
st.markdown("*Hybrid AI powered by ALS Collaborative Filtering + TF-IDF Content-Based*")
st.divider()

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # Get sample users from API
    try:
        users_resp = requests.get(f"{API_URL}/users?limit=50", timeout=5)
        sample_users = users_resp.json()['sample']
        total_users  = users_resp.json()['total_users']
    except:
        sample_users = []
        total_users  = 0
        st.error("❌ API not reachable. Make sure uvicorn is running.")

    st.markdown(f"**Total Users:** {total_users:,}")

    # User selection
    st.markdown("### 👤 Select User")
    user_input = st.text_input(
        "Enter User ID",
        value = sample_users[0] if sample_users else "",
        help  = "Paste any user ID from the list below"
    )

    st.markdown("**Quick Select:**")
    for u in sample_users[:8]:
        if st.button(u[:20] + "..." if len(u) > 20 else u,
                     key=u, use_container_width=True):
            user_input = u
            st.session_state['selected_user'] = u

    if 'selected_user' in st.session_state:
        user_input = st.session_state['selected_user']

    st.divider()
    n_recs = st.slider("Number of recommendations", 5, 20, 10)
    st.markdown("### 📊 Model Info")
    st.markdown("""
    - **Retrieval:** ALS Matrix Factorization
    - **Content:** TF-IDF + Cosine Similarity
    - **Hybrid:** Weighted combination
    - **NDCG@10:** 0.156
    - **Recall@10:** 0.195
    """)

# ── Main content ──────────────────────────────────────────
if user_input:
    # Fetch recommendations
    with st.spinner("🔍 Finding recommendations..."):
        try:
            resp = requests.get(
                f"{API_URL}/recommend/{user_input}?n={n_recs}",
                timeout=60
            )

            if resp.status_code == 404:
                st.error(f"❌ User '{user_input}' not found in dataset")
                st.stop()
            elif resp.status_code != 200:
                st.error(f"❌ API error: {resp.status_code}")
                st.stop()

            data     = resp.json()
            products = data['products']
            strategy = data['strategy']
            count    = data['count']

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Make sure uvicorn is running on port 8000.")
            st.stop()

    # ── User info bar ──────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class='metric-box'>
            <h4 style='color:#e94560'>User ID</h4>
            <p style='color:white; font-size:11px'>{user_input[:25]}...</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-box'>
            <h4 style='color:#e94560'>Strategy</h4>
            <p style='color:#00b4d8'>{strategy}</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='metric-box'>
            <h4 style='color:#e94560'>Results</h4>
            <p style='color:white'>{count} products</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class='metric-box'>
            <h4 style='color:#e94560'>Model</h4>
            <p style='color:#00b4d8'>Hybrid ALS + TF-IDF</p>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Recommendations grid ───────────────────────────────
    st.markdown(f"### 🎯 Top {count} Recommendations")

    for i in range(0, len(products), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j >= len(products):
                break
            p = products[i + j]
            with col:
                with st.container():
                    # Image + details side by side
                    img_col, info_col = st.columns([1, 2])

                    with img_col:
                        if p['image_url']:
                            st.image(p['image_url'], width=120)
                        else:
                            st.markdown("🎵")

                    with info_col:
                        st.markdown(f"""
                        <div class='product-title'>
                            {i+j+1}. {p['title']}
                        </div>
                        <div class='product-brand'>
                            🏷️ {p['brand']}
                        </div>
                        <div class='product-price'>
                            {p['price']}
                        </div>
                        <span class='score-badge'>
                            Score: {p['score']}
                        </span>
                        """, unsafe_allow_html=True)

                    if p['features']:
                        with st.expander("📋 Features"):
                            for feat in p['features'].split(' | '):
                                st.markdown(f"• {feat}")

                    if p['description'] and p['description'] != 'No description available':
                        with st.expander("📖 Description"):
                            st.markdown(p['description'])

                    st.divider()

    # ── Similar products section ───────────────────────────
    st.markdown("### 🔍 Explore Similar Products")
    if products:
        product_options = {
            f"{p['title'][:50]}": p['product_id']
            for p in products
            if p['title'] != 'Unknown Product'
        }
        selected_title = st.selectbox(
            "Pick a product to find similar items:",
            options = list(product_options.keys())
        )

        if selected_title:
            selected_pid = product_options[selected_title]
            with st.spinner("Finding similar products..."):
                sim_resp = requests.get(
                    f"{API_URL}/similar/{selected_pid}?n=6",
                    timeout=30
                )
                if sim_resp.status_code == 200:
                    sim_data = sim_resp.json()
                    sim_prods = sim_data['similar']

                    st.markdown(f"**Similar to:** {sim_data['title']}")
                    sim_cols = st.columns(3)
                    for idx, sp in enumerate(sim_prods[:6]):
                        with sim_cols[idx % 3]:
                            if sp['image_url']:
                                st.image(sp['image_url'], width=100)
                            st.markdown(f"**{sp['title'][:40]}**")
                            st.markdown(f"*{sp['brand']}*")
                            st.markdown(f"**{sp['price']}**")
                            st.markdown(
                                f"Similarity: `{sp['similarity']}`"
                            )
                            st.divider()

else:
    # ── Empty state ────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding:60px; color:#a8a8b3'>
        <h2>👈 Select a user from the sidebar</h2>
        <p>Choose a user ID to see their personalized music gear recommendations</p>
    </div>
    """, unsafe_allow_html=True)