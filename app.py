"""실행: streamlit run app.py"""

import streamlit as st
from dotenv import load_dotenv

from src.data import load_draws
from src.frontend import render_frontend
from src.planner import make_planner
from src.settings import ROOT, load_settings, resolve_path
from src.workflow import make_workflow

st.set_page_config(page_title="LOTTORY", page_icon="🎱", layout="wide")
load_dotenv(ROOT / ".env")


@st.cache_data
def read_draws(path, modified_ns, minimum):
    return load_draws(path, minimum=minimum)


@st.cache_resource
def rag_engine(settings, _planner):
    return make_workflow(settings, planner=_planner)


try:
    settings = load_settings()
    path = resolve_path(settings["data"]["csv_path"])
    draws = read_draws(str(path), path.stat().st_mtime_ns, settings["data"]["minimum_rounds"])
except (OSError, ValueError):
    st.warning("검증된 회차 데이터가 없습니다. 터미널에서 아래 명령을 실행하세요.")
    st.code("python -m scripts.prepare_data", language="powershell")
    st.stop()

render_frontend(settings, draws, lambda config: rag_engine(config, make_planner(config)))
