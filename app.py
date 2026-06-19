import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import os
import sys
import cv2
import base64
import warnings

warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import time

sys.path.insert(0, 'C:/python_workspace/churn_project')

# ─── 페이지 설정 ──────────────────────────────────────────────────────────────
"""
OTT 서비스 고객 이탈 예측 시스템
Streamlit 현업 수준 대시보드
페이스 로그인 + EDA + 모델 예측 + 결과 시각화
"""

st.set_page_config(
    page_title="고객 이탈 예측 시스템 | OTT Analytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS 스타일 ───────────────────────────────────────────────────────────────
st.markdown("""
<style>

    .stApp { background-color: #0F1117; }

    /* 사이드바 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #0d1117 100%);
        border-right: 1px solid #30363d;
    }

    /* 메트릭 카드 */
    .metric-card {
        background: linear-gradient(135deg, #1e2433 0%, #252b3b 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-2px);
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #58a6ff;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #8b949e;
        margin-top: 5px;
    }

    .metric-delta {
        font-size: 0.8rem;
        margin-top: 4px;
    }

    .delta-up { color: #f85149; }
    .delta-down { color: #3fb950; }

    /* 섹션 헤더 */
    .section-header {
        background: linear-gradient(90deg, #1f6feb 0%, #388bfd 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 15px 0 10px 0;
    }

    /* 로그인 카드 */
    .login-container {
        background: white;
        border: 1px solid #d0d7de;
        border-radius: 16px;
        padding: 40px;
        max-width: 500px;
        margin: 0 auto;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
    }

    /* 버튼 */
    .stButton > button {
        background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
    }

    /* 탭 */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f1f5f9;
        border-radius: 8px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #333;
        border-radius: 6px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1f6feb !important;
        color: white !important;
    }

    /* 로그인 화면 오른쪽 글자 */
    .stTextInput label,
    .stSelectbox label,
    .stCheckbox label {
        color: #222 !important;
    }

    .stTextInput input {
        background: white !important;
        color: black !important;
    }

    /* 로그인 화면 왼쪽 제목 */
    .login-title {
        color: white !important;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
    }

    /* 숨기기 */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

</style>
""", unsafe_allow_html=True)


# ─── 데이터 및 모델 로드 ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('C:/python_workspace/churn_project/data/korea_telecom_churn.csv', encoding='utf-8-sig')
    # OTT 데이터 컬럼명에 맞춰 수정
    if '월시청시간' in df.columns:
        df['월시청시간'].fillna(df['월시청시간'].median(), inplace=True)
    if '만족도점수' in df.columns:
        df['만족도점수'].fillna(df['만족도점수'].median(), inplace=True)
    if '월평균접속횟수' in df.columns:
        df['월평균접속횟수'].fillna(df['월평균접속횟수'].median(), inplace=True)

    df['월평균요금'] = df['월정액'] * (1 - df['할인율'] / 100)
    df['연령대'] = pd.cut(df['나이'], bins=[0, 29, 39, 49, 59, 100],
                       labels=['20대', '30대', '40대', '50대', '60대이상'])
    df['고객가치점수'] = (df['가입기간_개월'] * 0.5 + df['부가서비스수'] * 15 + df['만족도점수'] * 10)
    return df


@st.cache_resource
def load_models():
    scaler = joblib.load('C:/python_workspace/churn_project/models/scaler.pkl')
    le_dict = joblib.load('C:/python_workspace/churn_project/models/label_encoders.pkl')
    feature_cols = joblib.load('C:/python_workspace/churn_project/models/feature_cols.pkl')
    best_model = joblib.load('C:/python_workspace/churn_project/models/best_model.pkl')
    with open('C:/python_workspace/churn_project/data/model_results.json', 'r', encoding='utf-8') as f:
        model_results = json.load(f)
    return scaler, le_dict, feature_cols, best_model, model_results


# ─── 세션 상태 초기화 ─────────────────────────────────────────────────────────
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'role' not in st.session_state:
    st.session_state.role = None
if 'login_time' not in st.session_state:
    st.session_state.login_time = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = '대시보드'

if 'password_verified' not in st.session_state:
    st.session_state.password_verified = False

if 'pending_user' not in st.session_state:
    st.session_state.pending_user = None

if 'pending_role' not in st.session_state:
    st.session_state.pending_role = None

if 'auth_step' not in st.session_state:
    st.session_state.auth_step = "login"


# ─── 얼굴 인증 함수 ───────────────────────────────────────────────────────────
def process_face_login(image_data, username_input=None):
    """이미지 데이터로 얼굴 인증 처리"""
    try:
        from face_auth import authenticate_user, register_user, load_user_db
        import numpy as np

        # base64 또는 numpy 배열 처리
        if isinstance(image_data, np.ndarray):
            img_array = image_data
        else:
            return False, None, None, 0.0, "이미지 형식 오류"

        success, username, role, similarity, message = authenticate_user(img_array, threshold=0.60)
        return success, username, role, similarity, message
    except Exception as e:
        return False, None, None, 0.0, f"인증 오류: {str(e)}"


# ─── 로그인 페이지 ────────────────────────────────────────────────────────────
def show_login_page():
    """로그인 페이지 - 왼쪽 OTT/SNS 선택, 오른쪽 로그인 UI"""

    left_col, right_col = st.columns([1, 1])

    # ── 왼쪽 영역: 빈 박스 제거 + 서비스 선택만 표시 ──
    with left_col:

        st.markdown("<br><br><br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.markdown("""
            <h2 style="
                color:white;
                text-align:center;
                font-size:2rem;
                margin-bottom:30px;
                font-weight:700;
            ">
                어떤 분야의 관리자 이신가요?
            </h2>
            """, unsafe_allow_html=True)

            ott_checked = st.checkbox("🎬 OTT", key="select_ott")
            sns_checked = st.checkbox("📱 SNS", key="select_sns")

            selected_count = int(ott_checked) + int(sns_checked)

            if selected_count == 0:
                st.warning("OTT 또는 SNS 중 하나를 선택하세요.")
            elif selected_count > 1:
                st.error("하나만 선택할 수 있습니다.")
            else:
                selected_service = "OTT" if ott_checked else "SNS"
                st.session_state.selected_service = selected_service
                st.success(f"선택된 서비스: {selected_service}")

    can_login = selected_count == 1

    # ── 오른쪽 영역: 로그인 UI ──
    with right_col:
        tab1 = st.tabs(["🔑 계정 로그인"])[0]

    with tab1:
        with st.form("login_form"):
            username = st.text_input("사용자 ID", placeholder="admin")
            password = st.text_input(
                "비밀번호",
                type="password",
                placeholder="admin123"
            )

            submit = st.form_submit_button(
                "로그인",
                use_container_width=True,
                disabled=not can_login
            )

            if submit:
                demo_accounts = {
                    'admin': {'password': 'admin123', 'role': 'admin'},
                    'analyst1': {'password': 'analyst123', 'role': 'analyst'},
                    'viewer1': {'password': 'viewer123', 'role': 'viewer'},
                }

                if username in demo_accounts and demo_accounts[username]['password'] == password:
                    st.session_state.password_verified = True
                    st.session_state.pending_user = username
                    st.session_state.pending_role = demo_accounts[username]['role']
                    st.session_state.auth_step = "face_auth"

                    st.success("1차 인증 성공")
                    time.sleep(0.5)
                    st.rerun()

        st.info("admin / admin123")
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="
            text-align:center;
            color:#8b949e;
            font-size:0.9rem;
            margin:15px 0 10px 0;
        ">
            또는 소셜 계정으로 로그인
        </div>
        """,
        unsafe_allow_html=True
    )

    social_col1, social_col2, social_col3 = st.columns(3)

    with social_col1:
        if st.button("💬 Kakao", use_container_width=True, key="kakao_login"):
            st.info("카카오 로그인 연동 예정")

    with social_col2:
        if st.button("🟢 Naver", use_container_width=True, key="naver_login"):
            st.info("네이버 로그인 연동 예정")

    with social_col3:
        if st.button("🌐 Google", use_container_width=True, key="google_login"):
            st.info("구글 로그인 연동 예정")

    # ---------------------------------------------------------------------------


# -----------------------------------------------------------------------------
def show_face_auth_page():
    st.markdown("<br><br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])

    with col2:
        st.markdown("""
        <div style="
            background:#1e2433;
            border:1px solid #30363d;
            border-radius:16px;
            padding:35px;
            text-align:center;
        ">
            <h2 style="color:white;">2차 인증: 얼굴 인식</h2>
            <p style="color:#8b949e;">
                ID/PW 인증이 완료되었습니다.<br>
                얼굴 인증을 진행해주세요.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.info(f"1차 인증 계정: {st.session_state.pending_user}")

        camera_img = st.camera_input(
            "얼굴 인증 촬영",
            key="second_face_auth_page"
        )

        auth_col1, auth_col2 = st.columns(2)

        with auth_col1:
            if st.button("얼굴 인증하기", use_container_width=True):
                if camera_img is None:
                    st.error("얼굴을 촬영해주세요.")
                else:
                    from PIL import Image
                    import numpy as np
                    import cv2

                    img = Image.open(camera_img)
                    img_array = np.array(img)

                    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                    elif len(img_array.shape) == 3:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                    with st.spinner("얼굴 인증 중..."):
                        success, username, role, similarity, message = process_face_login(img_array)

                    if success:
                        st.session_state.authenticated = True
                        st.session_state.username = st.session_state.pending_user
                        st.session_state.role = st.session_state.pending_role
                        st.session_state.login_time = datetime.now()

                        st.session_state.password_verified = False
                        st.session_state.pending_user = None
                        st.session_state.pending_role = None
                        st.session_state.auth_step = "login"

                        st.success("2차 얼굴 인증 성공")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(f"얼굴 인증 실패: {message}")

        with auth_col2:
            if st.button("이전으로", use_container_width=True):
                st.session_state.password_verified = False
                st.session_state.pending_user = None
                st.session_state.pending_role = None
                st.session_state.auth_step = "login"
                st.rerun()


# ─── 사이드바 ──────────────────────────────────────────────────────────────
def show_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style='text-align: center; padding: 20px 0;'>
            <div style='font-size: 3.5rem;'>🎬</div>
            <h2 style='color: #58a6ff; margin: 10px 0 0 0;'>OTT Analytics</h2>
            <p style='color: #8b949e; font-size: 0.8rem;'>v2.0 (Stable)</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # 메뉴
        menu_options = {
            "대시보드": "📊",
            "EDA 분석": "🔍",
            "모델 성능": "🤖",
            "이탈 예측": "🔮",
            "고객 관리": "👥",
            "시스템 설정": "⚙️"
        }

        for option, icon in menu_options.items():
            if st.button(f"{icon} {option}", use_container_width=True,
                         type="primary" if st.session_state.current_page == option else "secondary"):
                st.session_state.current_page = option
                st.rerun()

        st.markdown("---")

        # 사용자 정보
        st.markdown(f"""
        <div style='background: rgba(48, 54, 61, 0.3); border-radius: 8px; padding: 15px;'>
            <p style='color: #8b949e; font-size: 0.75rem; margin: 0;'>Logged in as</p>
            <p style='color: #e6edf3; font-weight: 600; margin: 0;'>{st.session_state.username} ({st.session_state.role})</p>
            <p style='color: #8b949e; font-size: 0.7rem; margin-top: 5px;'>Since: {st.session_state.login_time.strftime('%H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("로그아웃", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.role = None
            st.rerun()


# ─── 대시보드 페이지 ──────────────────────────────────────────────────────────
def show_dashboard(df):
    st.markdown(
        "<div style='font-size: 1.8rem; font-weight: 700; color: #e6edf3; margin-bottom: 20px;'>📊 고객 이탈 분석 대시보드</div>",
        unsafe_allow_html=True)

    # KPI 카드
    col1, col2, col3, col4 = st.columns(4)

    total_customers = len(df)
    churn_rate = df['이탈여부'].mean() * 100
    avg_revenue = df['월정액'].mean()
    avg_tenure = df['가입기간_개월'].mean()

    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>전체 고객 수</div>
            <div class='metric-value'>{total_customers:,}</div>
            <div class='metric-delta delta-up'>▲ 1.2% vs last month</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>평균 이탈률</div>
            <div class='metric-value'>{churn_rate:.2f}%</div>
            <div class='metric-delta delta-down'>▼ 0.5% vs last month</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>평균 월 매출 (ARPU)</div>
            <div class='metric-value'>₩{avg_revenue:,.0f}</div>
            <div class='metric-delta delta-up'>▲ ₩450 vs last month</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>평균 유지 기간</div>
            <div class='metric-value'>{avg_tenure:.1f}개월</div>
            <div class='metric-delta delta-up'>▲ 0.2개월 vs last month</div>
        </div>
        """, unsafe_allow_html=True)

    carrier_col = '서비스' if '서비스' in df.columns else '통신사'
    # 차트 행 1
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<div class='section-header'>📈 서비스별 월별 이탈 트렌드</div>", unsafe_allow_html=True)
        # 가상 트렌드 데이터 (서비스별)
        months = ['23.07', '23.08', '23.09', '23.10', '23.11', '23.12', '24.01', '24.02', '24.03', '24.04', '24.05',
                  '24.06']

        fig = go.Figure()
        services = ['Netflix', 'Tving', 'Disney+']
        colors = ['#E50914', '#00A0E9', '#0063E5']

        # 가상 데이터 생성 로직 (시각화용)
        np.random.seed(42)
        for svc, color in zip(services, colors):
            base_rate = 12 if svc == 'Netflix' else 15 if svc == 'Tving' else 18
            trend = base_rate + np.random.normal(0, 1, len(months))
            fig.add_trace(go.Scatter(
                x=months, y=trend, mode='lines+markers',
                name=svc, line=dict(color=color, width=3),
                marker=dict(size=8)
            ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c9d1d9'), height=350,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(gridcolor='#21262d'),
            yaxis=dict(gridcolor='#21262d', title='이탈률 (%)'),
            legend=dict(orientation='h', y=1.1, x=0.5, xanchor='center')
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>⏱️ 이용 시간 분포 (시청 시간)</div>", unsafe_allow_html=True)
        # 연령대별 이탈률 대신 이용시간 분포
        fig = px.histogram(df, x='월시청시간', color='이탈여부',
                           color_discrete_map={0: '#3fb950', 1: '#f85149'},
                           marginal="box",  # 상단에 박스플롯 추가
                           nbins=50, opacity=0.7)

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c9d1d9'), height=350,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(gridcolor='#21262d', title='월 시청 시간 (시간)'),
            yaxis=dict(gridcolor='#21262d', title='고객 수'),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    # 차트 행 2
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<div class='section-header'>📊 요금제별 이탈률</div>", unsafe_allow_html=True)
        plan_churn = df.groupby('요금제유형')['이탈여부'].mean().reset_index()
        plan_churn.columns = ['요금제', '이탈률']
        plan_churn['이탈률'] *= 100
        plan_churn = plan_churn.sort_values('이탈률', ascending=True)

        fig = go.Figure(go.Bar(
            x=plan_churn['이탈률'], y=plan_churn['요금제'],
            orientation='h',
            marker=dict(
                color=plan_churn['이탈률'],
                colorscale='RdYlGn_r',
                showscale=False
            ),
            text=plan_churn['이탈률'].round(1),
            texttemplate='%{text}%', textposition='outside'
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c9d1d9'), height=350,
            margin=dict(l=0, r=40, t=10, b=0),
            xaxis=dict(gridcolor='#21262d')
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>⭕ 다중 OTT 서비스 이용 현황</div>", unsafe_allow_html=True)
        # 만족도별 이탈률 대신 벤다이어그램 형태의 다중 이용 현황
        # 실제 벤다이어그램은 plotly에서 직접 지원하지 않으므로 히트맵이나 막대 그래프로 대체 가능하지만
        # 여기서는 '다중 이용 수'를 기반으로 한 분포 시각화로 구현
        if 'Netflix_사용' in df.columns:
            df['다중이용수'] = df['Netflix_사용'] + df['Tving_사용'] + df['Disney_사용']
            multi_use = df['다중이용수'].value_counts().sort_index()

            fig = px.pie(values=multi_use.values, names=[f"{i}개 서비스 이용" for i in multi_use.index],
                         hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
        else:
            # 데이터가 없는 경우 가상 데이터
            fig = px.pie(values=[2500, 1500, 1000], names=["1개 이용", "2개 이용", "3개 이용"],
                         hole=0.4)

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c9d1d9'), height=350,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation='v', y=0.5, x=1.0)
        )
        st.plotly_chart(fig, use_container_width=True)
    # 고위험 고객 테이블
    st.markdown("<div class='section-header'>🚨 고위험 이탈 예상 고객 TOP 10</div>", unsafe_allow_html=True)
    high_risk_df = df[
        (df['만족도점수'] <= 2) & (df['CS문의횟수_6개월'] >= 3)
        ].head(10).copy()

    display_cols = ['고객ID', '나이', carrier_col, '요금제유형', '가입기간_개월', '만족도점수', 'CS문의횟수_6개월', '이탈여부']
    high_risk_df = high_risk_df[display_cols]

    if len(high_risk_df) > 0:
        high_risk_df.columns = ['고객ID', '나이', '서비스', '요금제', '가입기간(월)', '만족도', 'CS문의', '이탈여부']
        high_risk_df['이탈여부'] = high_risk_df['이탈여부'].map({1: '🔴 이탈', 0: '🟢 유지'})
        st.dataframe(high_risk_df, use_container_width=True, hide_index=True)


# ─── EDA 페이지 ───────────────────────────────────────────────────────────────
def show_eda(df):
    st.markdown(
        "<div style='font-size: 1.8rem; font-weight: 700; color: #e6edf3; margin-bottom: 20px;'>🔍 탐색적 데이터 분석 (EDA)</div>",
        unsafe_allow_html=True)

    carrier_col = '서비스' if '서비스' in df.columns else '통신사'

    # 필터
    with st.expander("🔧 데이터 필터", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            carrier_filter = st.multiselect("서비스", df[carrier_col].unique(), default=df[carrier_col].unique())
        with col2:
            plan_filter = st.multiselect("요금제", df['요금제유형'].unique(), default=df['요금제유형'].unique())
        with col3:
            churn_filter = st.multiselect("이탈 여부", [0, 1], default=[0, 1],
                                          format_func=lambda x: '이탈' if x == 1 else '유지')

    df_filtered = df[
        df[carrier_col].isin(carrier_filter) &
        df['요금제유형'].isin(plan_filter) &
        df['이탈여부'].isin(churn_filter)
        ]
    st.info(f"📌 필터 적용 결과: **{len(df_filtered):,}명** (전체 {len(df):,}명)")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 분포 분석", "🔗 상관관계", "📈 세그먼트 분석", "📋 데이터 요약"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("X축 변수", ['월정액', '가입기간_개월', '나이', '월시청시간',
                                           '월평균접속횟수', '만족도점수', '고객가치점수'], key='eda_x')
        with col2:
            color_var = st.selectbox("색상 구분", ['이탈여부', carrier_col, '요금제유형', '구독유형'], key='eda_color')

        col1, col2 = st.columns(2)
        with col1:
            if color_var == '이탈여부':
                df_plot = df_filtered.copy()
                df_plot['이탈여부_str'] = df_plot['이탈여부'].map({0: '유지', 1: '이탈'})
                fig = px.histogram(df_plot, x=x_var, color='이탈여부_str',
                                   color_discrete_map={'유지': '#3fb950', '이탈': '#f85149'},
                                   barmode='overlay', opacity=0.7,
                                   title=f'{x_var} 분포 (이탈 vs 유지)')
            else:
                fig = px.histogram(df_filtered, x=x_var, color=color_var,
                                   barmode='overlay', opacity=0.7,
                                   title=f'{x_var} 분포')
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#c9d1d9'), height=320,
                xaxis=dict(gridcolor='#21262d'),
                yaxis=dict(gridcolor='#21262d')
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            y_var = st.selectbox("Y축 변수 (박스플롯)", ['월정액', '가입기간_개월', '나이',
                                                  '월시청시간', '만족도점수'], key='eda_y')
            df_plot2 = df_filtered.copy()
            df_plot2['이탈여부_str'] = df_plot2['이탈여부'].map({0: '유지', 1: '이탈'})
            fig2 = px.box(df_plot2, x='이탈여부_str', y=y_var,
                          color='이탈여부_str',
                          color_discrete_map={'유지': '#3fb950', '이탈': '#f85149'},
                          title=f'{y_var} 박스플롯')
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#c9d1d9'), height=320,
                yaxis=dict(gridcolor='#21262d'), showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        corr = df_filtered[numeric_cols].corr()

        fig = px.imshow(corr, text_auto='.2f', aspect='auto',
                        color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                        title='변수 간 상관관계 히트맵')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c9d1d9'), height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            seg_x = st.selectbox("X축", [carrier_col, '요금제유형', '구독유형', '지역', '연령대'], key='seg_x')
        with col2:
            seg_metric = st.selectbox("지표", ['이탈률', '평균 월정액', '평균 만족도', '평균 가입기간'], key='seg_m')

        metric_map = {
            '이탈률': ('이탈여부', 'mean', lambda x: x * 100),
            '평균 월정액': ('월정액', 'mean', lambda x: x),
            '평균 만족도': ('만족도점수', 'mean', lambda x: x),
            '평균 가입기간': ('가입기간_개월', 'mean', lambda x: x),
        }
        col_name, agg_func, transform = metric_map[seg_metric]

        seg_data = df_filtered.groupby(seg_x)[col_name].agg(agg_func).reset_index()
        seg_data[col_name] = seg_data[col_name].apply(transform)
        seg_data = seg_data.sort_values(col_name, ascending=False)

        fig = px.bar(seg_data, x=seg_x, y=col_name,
                     color=col_name, color_continuous_scale='RdYlGn_r',
                     title=f'{seg_x}별 {seg_metric}',
                     text=seg_data[col_name].round(2))
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c9d1d9'), height=400,
            xaxis=dict(gridcolor='#21262d'),
            yaxis=dict(gridcolor='#21262d'),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📊 수치형 변수 통계**")
            st.dataframe(df_filtered.describe().round(2), use_container_width=True)
        with col2:
            st.markdown("**📋 범주형 변수 분포**")
            for col in [carrier_col, '요금제유형', '구독유형']:
                st.markdown(f"*{col}*")
                vc = df_filtered[col].value_counts()
                for val, cnt in vc.items():
                    pct = cnt / len(df_filtered) * 100
                    st.markdown(f"""
                    <div style='display:flex; justify-content:space-between; font-size:0.85rem; padding:2px 0;'>
                        <span style='color:#c9d1d9;'>{val}</span>
                        <span style='color:#8b949e;'>{cnt:,}명 ({pct:.1f}%)</span>
                    </div>
                    """, unsafe_allow_html=True)


# ─── 모델 성능 페이지 ─────────────────────────────────────────────────────────
def show_model_performance(model_results):
    st.markdown(
        "<div style='font-size: 1.8rem; font-weight: 700; color: #e6edf3; margin-bottom: 20px;'>🤖 모델 성능 비교</div>",
        unsafe_allow_html=True)

    results_df = pd.DataFrame(model_results).T.reset_index()
    results_df.columns = ['모델명', '정확도', '정밀도', '재현율', 'F1 점수', 'AUC-ROC', 'CV F1 평균', 'CV F1 표준편차']

    # 지표 선택
    metric = st.selectbox("비교 지표 선택", ['AUC-ROC', 'F1 점수', '정확도', '재현율'])

    fig = px.bar(results_df.sort_values(metric, ascending=False),
                 x='모델명', y=metric, color=metric,
                 color_continuous_scale='Blues',
                 title=f'모델별 {metric} 비교',
                 text=results_df.sort_values(metric, ascending=False)[metric].round(4))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#c9d1d9'), height=400,
        xaxis=dict(gridcolor='#21262d'),
        yaxis=dict(gridcolor='#21262d'),
        coloraxis_showscale=False
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'>📋 상세 성능 지표</div>", unsafe_allow_html=True)
    st.dataframe(results_df, use_container_width=True, hide_index=True)


# ─── 이탈 예측 페이지 ─────────────────────────────────────────────────────────
def show_prediction(df, scaler, le_dict, feature_cols, best_model):
    st.markdown(
        "<div style='font-size: 1.8rem; font-weight: 700; color: #e6edf3; margin-bottom: 20px;'>🔮 개별 고객 이탈 예측</div>",
        unsafe_allow_html=True)

    carrier_col = '서비스' if '서비스' in df.columns else '통신사'

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 👤 인구통계")
            age = st.slider("나이", 18, 80, 35)
            gender = st.selectbox("성별", df['성별'].unique())
            region = st.selectbox("지역", df['지역'].unique())

        with col2:
            st.markdown("### 📋 서비스 정보")
            carrier = st.selectbox("서비스", df[carrier_col].unique())
            plan = st.selectbox("요금제", df['요금제유형'].unique())
            contract = st.selectbox("구독유형", df['구독유형'].unique())
            tenure = st.number_input("가입기간 (개월)", 1, 120, 12)

        with col3:
            st.markdown("### 📱 이용 패턴")
            usage = st.number_input("월 시청시간 (시간)", 0.0, 500.0, 50.0)
            access = st.number_input("월 접속횟수", 0, 300, 20)
            profiles = st.number_input("프로필 수", 1, 5, 1)
            sat = st.slider("만족도 점수", 1, 5, 4)

        submit = st.form_submit_button("이탈 위험도 예측하기", use_container_width=True)

        if submit:
            # 예측 로직 (간소화)
            prob = 0.15  # 데모용 고정값

            st.markdown("---")
            col1, col2 = st.columns([1, 1])

            with col1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    title={'text': "이탈 위험도 (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#f85149" if prob > 0.5 else "#3fb950"},
                        'steps': [
                            {'range': [0, 30], 'color': "rgba(63, 185, 80, 0.2)"},
                            {'range': [30, 70], 'color': "rgba(210, 153, 34, 0.2)"},
                            {'range': [70, 100], 'color': "rgba(248, 81, 73, 0.2)"}
                        ]
                    }
                ))
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#c9d1d9"})
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                risk_level = "고위험" if prob > 0.7 else "중위험" if prob > 0.3 else "저위험"
                st.markdown(
                    f"### 예측 결과: <span style='color:{'#f85149' if risk_level == '고위험' else '#d29922' if risk_level == '중위험' else '#3fb950'}'>{risk_level}</span>",
                    unsafe_allow_html=True)
                st.markdown("""
                **💡 권장 대응 전략:**
                - 장기 구독 할인 혜택 안내
                - 독점 신작 콘텐츠 알림 발송
                - 만족도 개선을 위한 설문 및 포인트 지급
                """)


# ─── 고객 관리 페이지 ─────────────────────────────────────────────────────────
def show_customer_mgmt(df):
    st.markdown(
        "<div style='font-size: 1.8rem; font-weight: 700; color: #e6edf3; margin-bottom: 20px;'>👥 고객 리스트 관리</div>",
        unsafe_allow_html=True)
    st.dataframe(df.head(100), use_container_width=True)


# ─── 메인 실행부 ──────────────────────────────────────────────────────────────
def main():
    if not st.session_state.authenticated:
        if st.session_state.auth_step == "face_auth":
            show_face_auth_page()
        else:
            show_login_page()
    else:
        scaler, le_dict, feature_cols, best_model, model_results = load_models()

        show_sidebar()

        if st.session_state.current_page == "대시보드":
            show_dashboard(df)
        elif st.session_state.current_page == "EDA 분석":
            show_eda(df)
        elif st.session_state.current_page == "모델 성능":
            show_model_performance(model_results)
        elif st.session_state.current_page == "이탈 예측":
            show_prediction(df, scaler, le_dict, feature_cols, best_model)
        elif st.session_state.current_page == "고객 관리":
            show_customer_mgmt(df)
        elif st.session_state.current_page == "시스템 설정":
            st.info("시스템 설정 페이지입니다. (준비 중)")


if __name__ == "__main__":
    main()
