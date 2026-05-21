import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from inference import load_default_row, predict_load

BURGUNDY = '#3D1515'
CRIMSON = '#8B2942'
CREAM = '#F5F0E1'
BEIGE = '#D4C4A8'

UI_FIELDS = [
    ('lag_1', 'Лаг 1 ч', 'number', 0.0, 2500.0),
    ('lag_24', 'Лаг 24 ч', 'number', 0.0, 2500.0),
    ('rolling_mean_24', 'Средняя 24 ч', 'number', 0.0, 2500.0),
    ('hour', 'Час', 'int', 0, 23),
    ('day_of_week', 'День недели', 'int', 0, 6),
    ('month', 'Месяц', 'int', 1, 12),
    ('is_weekend', 'Выходной', 'weekend', 0, 1),
    ('T2M_toc', 'Температура', 'number', -10.0, 45.0),
    ('W2M_toc', 'Ветер', 'number', 0.0, 40.0),
    ('Holiday_ID', 'Праздник ID', 'int', 0, 10),
    ('holiday', 'Праздник', 'int', 0, 1),
    ('school', 'Школа', 'int', 0, 1),
]

st.set_page_config(page_title='Прогноз нагрузки', layout='wide')

st.markdown(
    f"""
    <style>
    .stApp {{ background: linear-gradient(180deg, {CREAM} 0%, {BEIGE} 100%); }}
    h1 {{ color: {BURGUNDY}; }}
    .result-box {{
        background: white;
        border-left: 5px solid {CRIMSON};
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin-top: 1rem;
        text-align: center;
    }}
    .result-value {{ color: {BURGUNDY}; font-size: 2.2rem; font-weight: 700; }}
    .stButton>button[kind="primary"] {{ background: {CRIMSON}; border-color: {BURGUNDY}; }}
    .stButton>button[kind="primary"]:hover {{ background: {BURGUNDY}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

if 'features' not in st.session_state:
    st.session_state.features = load_default_row()
if 'last_pred' not in st.session_state:
    st.session_state.last_pred = None

st.title('Прогноз нагрузки')

_, col_btn = st.columns([4, 1])
with col_btn:
    if st.button('Пример', use_container_width=True):
        st.session_state.features = load_default_row()
        st.session_state.last_pred = None
        st.rerun()

with st.form('predict_form', border=False):
    c1, c2, c3 = st.columns(3)
    edited = dict(st.session_state.features)
    for idx, (key, label, kind, lo, hi) in enumerate(UI_FIELDS):
        col = (c1, c2, c3)[idx % 3]
        val = float(st.session_state.features.get(key, 0))
        with col:
            if kind == 'weekend':
                edited[key] = float(
                    st.selectbox(label, options=[0, 1], index=int(val), format_func=lambda x: 'нет' if x == 0 else 'да')
                )
            elif kind == 'int':
                edited[key] = float(
                    st.number_input(label, min_value=int(lo), max_value=int(hi), value=int(val), step=1)
                )
            else:
                edited[key] = st.number_input(label, min_value=float(lo), max_value=float(hi), value=val, format='%.2f')
    submitted = st.form_submit_button('Спрогнозировать', type='primary', use_container_width=True)

if submitted:
    st.session_state.features = edited
    try:
        st.session_state.last_pred = predict_load(edited)
    except FileNotFoundError as exc:
        st.error(str(exc))

if st.session_state.last_pred:
    val = st.session_state.last_pred['predicted_load']
    st.markdown(
        f'<div class="result-box"><div class="result-value">{val:,.2f}</div></div>',
        unsafe_allow_html=True,
    )
