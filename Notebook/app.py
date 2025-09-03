import json
import difflib
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title='Shopper Spectrum', page_icon='🛒', layout='wide')

st.title('🛒 Shopper Spectrum — Simple App')
st.caption('Customer Segmentation (RFM + KMeans) & Item-based Recommendations — no functions, no classes')

# ===============================
# Load artifacts
# ===============================
scaler = joblib.load('rfm_scaler.pkl')
kmeans = joblib.load('rfm_kmeans.pkl')
with open('cluster_label_map.json','r') as f:
    cluster_label_map = json.load(f)

clean_df = pd.read_csv('data_preprocessed.csv', parse_dates=['InvoiceDate'])
product_master = pd.read_csv('product_master.csv')
item_item_sim = np.load('item_item_sim.npy')
item_list = np.load('item_list.npy', allow_pickle=True)
rfm_segments = pd.read_csv('rfm_with_segments.csv')  # ✅ added

# ===============================
# Tabs
# ===============================
rec_tab, seg_tab = st.tabs(["🔎 Product Recommendations", "👥 Customer Segmentation"])

# -------------------------------
# Product Recommendation Tab
# -------------------------------
with rec_tab:
    st.subheader('Find similar products (item-based collaborative filtering)')
    all_names = product_master['Description'].fillna('').astype(str).tolist()
    user_input = st.text_input('Enter a product name', '')
    topn = st.number_input('How many similar products?', min_value=1, max_value=20, value=5, step=1)

    if st.button('Get Recommendations'):
        if user_input.strip() == '':
            st.warning('Please type a product name.')
        else:
            matches = difflib.get_close_matches(user_input, all_names, n=5, cutoff=0.4)
            if len(matches) == 0:
                st.error('No close product names found. Try another keyword.')
            else:
                st.write('Closest matches:', matches)
                chosen_name = matches[0]
                chosen_row = product_master[product_master['Description'] == chosen_name].head(1)
                if len(chosen_row) == 0:
                    st.error('Matched description not found in product master.')
                else:
                    chosen_code = chosen_row['StockCode'].iloc[0]
                    try:
                        idx = np.where(item_list == chosen_code)[0][0]
                    except Exception:
                        st.error('Item not present in similarity index.')
                        idx = None

                    if idx is not None:
                        sims = item_item_sim[idx]
                        top_idx = np.argsort(-sims)
                        top_idx = [i for i in top_idx if i != idx][:int(topn)]
                        rec_codes = item_list[top_idx]
                        recs = product_master[product_master['StockCode'].isin(rec_codes)][['StockCode','Description','popularity_qty']]
                        st.success('Recommended products:')
                        st.dataframe(recs.reset_index(drop=True))

# -------------------------------
# Customer Segmentation Tab
# -------------------------------
with seg_tab:
    st.subheader('Predict customer segment from RFM')
    col1, col2, col3 = st.columns(3)
    with col1:
        r_in = st.number_input('Recency (days since last purchase)', min_value=0, value=30, step=1)
    with col2:
        f_in = st.number_input('Frequency (unique invoices)', min_value=0, value=5, step=1)
    with col3:
        m_in = st.number_input('Monetary (total spend)', min_value=0.0, value=100.0, step=10.0)

    if st.button('Predict Cluster'):
        X = np.array([[r_in, f_in, m_in]], dtype=float)
        X_scaled = scaler.transform(X)
        cluster_id = int(kmeans.predict(X_scaled)[0])
        label = cluster_label_map.get(str(cluster_id), f'Cluster {cluster_id}')
        st.success(f'Segment: {label} (Cluster {cluster_id})')

    st.markdown("---")
    st.subheader("📊 Segment Overview")

    seg_counts = rfm_segments['Segment'].value_counts().reset_index()
    seg_counts.columns = ['Segment', 'CustomerCount']
    st.bar_chart(seg_counts.set_index('Segment'))

    st.markdown("### 🔎 Look up a CustomerID")
    cust_id = st.text_input("Enter CustomerID to find their segment")
    if cust_id.strip() != "":
        try:
            cust_id = int(cust_id)
            cust_row = rfm_segments[rfm_segments['CustomerID'] == cust_id]
            if cust_row.empty:
                st.error("CustomerID not found.")
            else:
                seg = cust_row['Segment'].iloc[0]
                st.success(f"Customer {cust_id} belongs to segment: **{seg}**")
                st.dataframe(cust_row)
        except ValueError:
            st.error("Please enter a numeric CustomerID.")

st.caption('© Shopper Spectrum — Demo app for learning purposes')
