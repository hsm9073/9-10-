import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# =========================================================
# 기본 설정
# =========================================================

st.set_page_config(
    page_title="영화 흥행 예측기",
    page_icon="🎬",
    layout="wide"
)

DAILY_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_daily.csv"
MOVIES_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(255,80,80,0.12), transparent 25%),
            radial-gradient(circle at 85% 20%, rgba(120,80,255,0.12), transparent 25%),
            linear-gradient(135deg, #080808 0%, #111111 45%, #080808 100%);
    }

    .main-title {
        font-size: 48px;
        font-weight: 900;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 5px;
        letter-spacing: -2px;
    }

    .sub-title {
        text-align: center;
        color: #aaaaaa;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .info-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 20px;
        text-align: center;
        height: 120px;
    }

    .info-number {
        font-size: 30px;
        font-weight: 800;
    }

    .info-label {
        color: #aaaaaa;
        font-size: 14px;
        margin-top: 5px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 800;
        margin-top: 35px;
        margin-bottom: 15px;
    }

    .warning-box {
        padding: 15px;
        border-radius: 12px;
        background: rgba(255,180,0,0.10);
        border: 1px solid rgba(255,180,0,0.30);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 데이터 불러오기
# =========================================================

@st.cache_data
def load_data():

    daily = pd.read_csv(
        DAILY_URL,
        encoding="utf-8"
    )

    movies = pd.read_csv(
        MOVIES_URL,
        encoding="utf-8"
    )

    return daily, movies


# =========================================================
# 숫자 변환
# =========================================================

def numeric_convert(df, columns):

    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    return df


# =========================================================
# 데이터 전처리 + 두 데이터 결합
# =========================================================

@st.cache_data
def prepare_data(daily, movies):

    daily = daily.copy()
    movies = movies.copy()

    # ---------------------------------------------
    # 영화코드 통일
    # ---------------------------------------------

    daily["movieCd"] = daily["영화코드"].astype(str).str.strip()
    movies["movieCd"] = movies["movieCd"].astype(str).str.strip()

    # ---------------------------------------------
    # 일별 데이터 숫자화
    # ---------------------------------------------

    daily = numeric_convert(
        daily,
        [
            "일관객",
            "누적관객",
            "스크린수",
            "상영횟수"
        ]
    )

    # ---------------------------------------------
    # 영화 정보 숫자화
    # ---------------------------------------------

    movies = numeric_convert(
        movies,
        [
            "first_scrn",
            "first_show",
            "peak",
            "first_week_audi",
            "total_audi",
            "days_in_top10"
        ]
    )

    # ---------------------------------------------
    # 날짜 변환
    # ---------------------------------------------

    daily["date"] = pd.to_datetime(
        daily["날짜"].astype(str),
        format="%Y%m%d",
        errors="coerce"
    )

    movies["open_date"] = pd.to_datetime(
        movies["openDt"].astype(str),
        format="%Y%m%d",
        errors="coerce"
    )

    movies["first_date_dt"] = pd.to_datetime(
        movies["first_date"].astype(str),
        format="%Y%m%d",
        errors="coerce"
    )

    # ---------------------------------------------
    # daily 데이터에서 영화별 통계 생성
    #
    # target(total_audi)와 직접 같은 값을 쓰지 않고
    # 일별 관객/스크린/상영횟수의 통계량을 생성
    # ---------------------------------------------

    daily_features = (
        daily
        .groupby("movieCd")
        .agg(
            daily_avg_audience=("일관객", "mean"),
            daily_max_audience=("일관객", "max"),
            daily_avg_screens=("스크린수", "mean"),
            daily_max_screens=("스크린수", "max"),
            daily_avg_shows=("상영횟수", "mean"),
            daily_max_shows=("상영횟수", "max"),
            daily_observations=("일관객", "count")
        )
        .reset_index()
    )

    # ---------------------------------------------
    # 첫 등장일 기준으로 데이터 기간 계산용
    # ---------------------------------------------

    daily_min = (
        daily
        .groupby("movieCd")["date"]
        .min()
        .reset_index()
        .rename(columns={"date": "daily_first_date"})
    )

    daily_max = (
        daily
        .groupby("movieCd")["date"]
        .max()
        .reset_index()
        .rename(columns={"date": "daily_last_date"})
    )

    daily_features = daily_features.merge(
        daily_min,
        on="movieCd",
        how="left"
    )

    daily_features = daily_features.merge(
        daily_max,
        on="movieCd",
        how="left"
    )

    # ---------------------------------------------
    # 영화 정보 + daily 통계 결합
    # ---------------------------------------------

    df = movies.merge(
        daily_features,
        on="movieCd",
        how="left"
    )

    # ---------------------------------------------
    # 날짜 기반 변수 생성
    # ---------------------------------------------

    df["open_year"] = df["open_date"].dt.year
    df["open_month"] = df["open_date"].dt.month

    df["first_year"] = df["first_date_dt"].dt.year
    df["first_month"] = df["first_date_dt"].dt.month

    # ---------------------------------------------
    # 장르 여러 개면 첫 장르만 사용
    # ---------------------------------------------

    df["genre_first"] = (
        df["genre"]
        .fillna("미상")
        .astype(str)
        .str.split(r"[|/]")
        .str[0]
        .str.strip()
    )

    df["nation_first"] = (
        df["nation"]
        .fillna("미상")
        .astype(str)
        .str.split(r"[|/]")
        .str[0]
        .str.strip()
    )

    # ---------------------------------------------
    # target이 없는 영화 제거
    # ---------------------------------------------

    df = df.dropna(
        subset=["total_audi"]
    ).copy()

    # ---------------------------------------------
    # 영화코드 순 정렬
    # ---------------------------------------------

    df["movieCd_num"] = pd.to_numeric(
        df["movieCd"],
        errors="coerce"
    )

    df = df.sort_values(
        by=["movieCd_num", "movieCd"]
    ).reset_index(drop=True)

    return df


# =========================================================
# 시험/학습 데이터 분리
# =========================================================

def split_movies(df):

    df = df.copy()

    test_indices = []

    # 10편 단위로 앞 3편을 시험용으로 지정
    for start in range(0, len(df), 10):

        block = df.iloc[start:start + 10]

        test_part = block.head(3)

        test_indices.extend(
            test_part.index.tolist()
        )

    df["데이터구분"] = "학습"

    df.loc[
        test_indices,
        "데이터구분"
    ] = "시험"

    train_df = df[
        df["데이터구분"] == "학습"
    ].copy()

    test_df = df[
        df["데이터구분"] == "시험"
    ].copy()

    return train_df, test_df


# =========================================================
# 예측 변수 설정
# =========================================================

FEATURE_INFO = {
    "첫 관측일 스크린수": ("first_scrn", "numeric"),
    "첫 관측일 상영횟수": ("first_show", "numeric"),
    "첫 주 관객수": ("first_week_audi", "numeric"),
    "TOP10 일수": ("days_in_top10", "numeric"),
    "성수기 개봉 여부": ("peak", "numeric"),
    "개봉 연도": ("open_year", "numeric"),
    "개봉 월": ("open_month", "numeric"),
    "TOP10 첫 등장 연도": ("first_year", "numeric"),
    "TOP10 첫 등장 월": ("first_month", "numeric"),
    "장르": ("genre_first", "categorical"),
    "국가": ("nation_first", "categorical"),

    "일별 평균 관객수": ("daily_avg_audience", "numeric"),
    "일별 최대 관객수": ("daily_max_audience", "numeric"),
    "일별 평균 스크린수": ("daily_avg_screens", "numeric"),
    "일별 최대 스크린수": ("daily_max_screens", "numeric"),
    "일별 평균 상영횟수": ("daily_avg_shows", "numeric"),
    "일별 최대 상영횟수": ("daily_max_shows", "numeric"),
    "일별 데이터 관측일수": ("daily_observations", "numeric")
}


# =========================================================
# 메인
# =========================================================

st.markdown(
    '<div class="main-title">🎬 영화 흥행 예측기</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">KOBIS 영화 데이터를 이용한 다중 회귀 기반 총 관객수 예측</div>',
    unsafe_allow_html=True
)


# =========================================================
# 데이터 로딩
# =========================================================

try:

    with st.spinner("🎞️ 영화 데이터를 불러오는 중..."):

        daily_df, movies_df = load_data()

        df = prepare_data(
            daily_df,
            movies_df
        )

except Exception as e:

    st.error(
        "데이터를 불러오는 과정에서 문제가 발생했습니다."
    )

    st.code(str(e))

    st.stop()


# =========================================================
# 데이터 기간
# =========================================================

all_dates = pd.concat(
    [
        daily_df["날짜"].astype(str),
        movies_df["openDt"].astype(str),
        movies_df["first_date"].astype(str)
    ],
    ignore_index=True
)

all_dates = pd.to_datetime(
    all_dates,
    format="%Y%m%d",
    errors="coerce"
)

valid_dates = all_dates.dropna()

if len(valid_dates) > 0:

    period_start = valid_dates.min()
    period_end = valid_dates.max()

else:

    period_start = None
    period_end = None


# =========================================================
# 기본 통계
# =========================================================

train_df, test_df = split_movies(df)

total_movies = len(df)
train_count = len(train_df)
test_count = len(test_df)


c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-number">{total_movies:,}</div>
            <div class="info-label">전체 영화 편수</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-number">{train_count:,}</div>
            <div class="info-label">학습 영화 편수</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-number">{test_count:,}</div>
            <div class="info-label">시험 영화 편수</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c4:
    if period_start is not None:
        period_text = (
            f"{period_start.strftime('%Y.%m.%d')} ~ "
            f"{period_end.strftime('%Y.%m.%d')}"
        )
    else:
        period_text = "확인 불가"

    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-number" style="font-size:18px;">
                {period_text}
            </div>
            <div class="info-label">분석 기준 기간</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# 변수 선택
# =========================================================

st.markdown(
    '<div class="section-title">🧠 예측에 사용할 변수 선택</div>',
    unsafe_allow_html=True
)

st.caption(
    "체크한 변수만 다중 회귀 모델의 입력값으로 사용됩니다."
)


selected_features = []

feature_columns = list(FEATURE_INFO.keys())

col1, col2, col3 = st.columns(3)

for i, label in enumerate(feature_columns):

    if i % 3 == 0:
        container = col1
    elif i % 3 == 1:
        container = col2
    else:
        container = col3

    default_value = label in [
        "첫 관측일 스크린수",
        "첫 관측일 상영횟수",
        "첫 주 관객수",
        "TOP10 일수",
        "성수기 개봉 여부",
        "장르",
        "국가",
        "일별 평균 관객수",
        "일별 평균 스크린수",
        "일별 평균 상영횟수"
    ]

    with container:

        checked = st.checkbox(
            label,
            value=default_value,
            key=f"feature_{i}"
        )

        if checked:
            selected_features.append(label)


# =========================================================
# 선택 변수 표시
# =========================================================

if len(selected_features) == 0:

    st.warning(
        "⚠️ 예측 변수를 하나 이상 선택해주세요."
    )

    st.stop()


# =========================================================
# 모델 데이터 준비
# =========================================================

feature_names = [
    FEATURE_INFO[x][0]
    for x in selected_features
]

categorical_features = [
    FEATURE_INFO[x][0]
    for x in selected_features
    if FEATURE_INFO[x][1] == "categorical"
]

numeric_features = [
    FEATURE_INFO[x][0]
    for x in selected_features
    if FEATURE_INFO[x][1] == "numeric"
]


X_train = train_df[feature_names].copy()
X_test = test_df[feature_names].copy()

y_train = train_df["total_audi"].copy()
y_test = test_df["total_audi"].copy()


# =========================================================
# 전처리
# =========================================================

transformers = []

if len(numeric_features) > 0:

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    transformers.append(
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        )
    )


if len(categorical_features) > 0:

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    transformers.append(
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    )


preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop"
)


# =========================================================
# 다중 회귀 모델
# =========================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regression",
            LinearRegression()
        )
    ]
)


# =========================================================
# 학습
# =========================================================

try:

    with st.spinner("🧠 회귀 모델을 학습하고 있습니다..."):

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

except Exception as e:

    st.error(
        "모델 학습 중 문제가 발생했습니다."
    )

    st.code(str(e))

    st.stop()


# =========================================================
# 음수 예측 방지
# =========================================================

predictions = np.maximum(
    predictions,
    0
)


# =========================================================
# 평가
# =========================================================

r2 = r2_score(
    y_test,
    predictions
)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)


# 평균 오차율
nonzero_mask = y_test > 0

if nonzero_mask.sum() > 0:

    mape = np.mean(
        np.abs(
            (
                y_test[nonzero_mask].values
                -
                predictions[nonzero_mask]
            )
            /
            y_test[nonzero_mask].values
        )
    ) * 100

else:

    mape = np.nan


# =========================================================
# 결과 표시
# =========================================================

st.markdown(
    '<div class="section-title">📊 시험용 데이터 예측 점수</div>',
    unsafe_allow_html=True
)


m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "R² 결정계수",
        f"{r2:.3f}"
    )

with m2:
    st.metric(
        "MAE 평균 절대 오차",
        f"{mae:,.0f}명"
    )

with m3:
    st.metric(
        "RMSE",
        f"{rmse:,.0f}명"
    )

with m4:
    if np.isnan(mape):
        mape_text = "계산 불가"
    else:
        mape_text = f"{mape:.1f}%"

    st.metric(
        "평균 오차율",
        mape_text
    )


st.caption(
    "R²는 시험용 영화에서 실제 총 관객 수의 변동을 모델이 얼마나 설명하는지를 나타냅니다. "
    "MAE와 RMSE는 예측값이 실제값에서 평균적으로 얼마나 떨어져 있는지를 나타냅니다."
)


# =========================================================
# 결과 데이터프레임
# =========================================================

result_df = test_df[
    [
        "movieCd",
        "movieNm",
        "total_audi"
    ]
].copy()

result_df["예측 총 관객수"] = predictions

result_df["실제 총 관객수"] = (
    result_df["total_audi"]
)

result_df["절대 오차"] = np.abs(
    result_df["실제 총 관객수"]
    -
    result_df["예측 총 관객수"]
)

result_df["오차율"] = np.where(
    result_df["실제 총 관객수"] > 0,
    (
        result_df["절대 오차"]
        /
        result_df["실제 총 관객수"]
    ) * 100,
    np.nan
)

result_df["예측 방향"] = np.where(
    result_df["예측 총 관객수"]
    >
    result_df["실제 총 관객수"],
    "과대 예측",
    "과소 예측"
)


# =========================================================
# 1,000명 미만 예측 영화
# =========================================================

low_prediction_mask = (
    result_df["예측 총 관객수"] < 1000
)

low_prediction_count = int(
    low_prediction_mask.sum()
)


# =========================================================
# 산점도
# =========================================================

st.markdown(
    '<div class="section-title">🎯 실제 관객수 vs 예측 관객수</div>',
    unsafe_allow_html=True
)

fig = go.Figure()


# ---------------------------------------------
# 일반 시험 영화
# ---------------------------------------------

normal_mask = (
    result_df["예측 총 관객수"] >= 1000
)

normal_df = result_df[
    normal_mask
].copy()


if len(normal_df) > 0:

    fig.add_trace(
        go.Scatter(
            x=normal_df["실제 총 관객수"],
            y=normal_df["예측 총 관객수"],
            mode="markers",
            name="시험 영화",
            text=normal_df["movieNm"],
            customdata=np.stack(
                [
                    normal_df["movieCd"],
                    normal_df["실제 총 관객수"],
                    normal_df["예측 총 관객수"],
                    normal_df["절대 오차"],
                    normal_df["오차율"]
                ],
                axis=1
            ),
            hovertemplate=
                "<b>%{text}</b><br>"
                "영화코드: %{customdata[0]}<br>"
                "실제 총 관객: %{customdata[1]:,.0f}명<br>"
                "예측 총 관객: %{customdata[2]:,.0f}명<br>"
                "절대 오차: %{customdata[3]:,.0f}명<br>"
                "오차율: %{customdata[4]:.1f}%"
                "<extra></extra>",
            marker=dict(
                size=10,
                opacity=0.8
            )
        )
    )


# ---------------------------------------------
# 1,000명 미만 예측값
#
# 로그축에서는 0을 표시할 수 없으므로
# 그래프 아래쪽에 100명 근처의 위치로 표시
# ---------------------------------------------

low_df = result_df[
    low_prediction_mask
].copy()


if len(low_df) > 0:

    low_y = np.full(
        len(low_df),
        100
    )

    fig.add_trace(
        go.Scatter(
            x=low_df["실제 총 관객수"],
            y=low_y,
            mode="markers",
            name="예측 1,000명 미만",
            text=low_df["movieNm"],
            customdata=np.stack(
                [
                    low_df["movieCd"],
                    low_df["실제 총 관객수"],
                    low_df["예측 총 관객수"],
                    low_df["절대 오차"],
                    low_df["오차율"]
                ],
                axis=1
            ),
            hovertemplate=
                "<b>%{text}</b><br>"
                "영화코드: %{customdata[0]}<br>"
                "실제 총 관객: %{customdata[1]:,.0f}명<br>"
                "예측 총 관객: %{customdata[2]:,.0f}명<br>"
                "절대 오차: %{customdata[3]:,.0f}명<br>"
                "오차율: %{customdata[4]:.1f}%"
                "<extra></extra>",
            marker=dict(
                size=11,
                symbol="diamond",
                opacity=0.9
            )
        )
    )


# ---------------------------------------------
# y = x 기준선
# ---------------------------------------------

all_values = np.concatenate(
    [
        result_df["실제 총 관객수"].values,
        result_df["예측 총 관객수"].values
    ]
)

positive_values = all_values[
    all_values > 0
]

if len(positive_values) > 0:

    line_min = max(
        positive_values.min() * 0.7,
        1
    )

    line_max = (
        positive_values.max() * 1.5
    )

else:

    line_min = 1
    line_max = 1000000


fig.add_trace(
    go.Scatter(
        x=[line_min, line_max],
        y=[line_min, line_max],
        mode="lines",
        name="완벽한 예측 기준선",
        line=dict(
            dash="dash",
            width=2
        ),
        hoverinfo="skip"
    )
)


fig.update_layout(
    height=650,
    xaxis=dict(
        title="실제 총 관객 수",
        type="log",
        showgrid=True
    ),
    yaxis=dict(
        title="예측한 총 관객 수",
        type="log",
        showgrid=True
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    ),
    margin=dict(
        l=40,
        r=40,
        t=50,
        b=40
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# 1,000명 미만 안내
# =========================================================

if low_prediction_count > 0:

    st.info(
        f"🔻 예측 총 관객수가 1,000명보다 작게 나온 영화는 "
        f"**{low_prediction_count}편**입니다. "
        f"그래프에서는 로그축 특성상 바닥 부분에 따로 표시했습니다."
    )

else:

    st.success(
        "🔻 예측 총 관객수가 1,000명보다 작게 나온 영화는 없습니다."
    )


# =========================================================
# 영화별 결과
# =========================================================

st.markdown(
    '<div class="section-title">🎞️ 시험용 영화별 예측 결과</div>',
    unsafe_allow_html=True
)

display_result = result_df[
    [
        "movieCd",
        "movieNm",
        "실제 총 관객수",
        "예측 총 관객수",
        "절대 오차",
        "오차율",
        "예측 방향"
    ]
].copy()

display_result["실제 총 관객수"] = (
    display_result["실제 총 관객수"]
    .round(0)
    .astype(int)
)

display_result["예측 총 관객수"] = (
    display_result["예측 총 관객수"]
    .round(0)
    .astype(int)
)

display_result["절대 오차"] = (
    display_result["절대 오차"]
    .round(0)
    .astype(int)
)

display_result["오차율"] = (
    display_result["오차율"]
    .round(1)
)

display_result = display_result.rename(
    columns={
        "movieCd": "영화코드",
        "movieNm": "영화명",
        "실제 총 관객수": "실제 총 관객수",
        "예측 총 관객수": "예측 총 관객수",
        "절대 오차": "절대 오차",
        "오차율": "오차율(%)",
        "예측 방향": "예측 방향"
    }
)

st.dataframe(
    display_result,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 분석 정보
# =========================================================

st.markdown(
    '<div class="section-title">⚙️ 모델 및 데이터 정보</div>',
    unsafe_allow_html=True
)

info_col1, info_col2 = st.columns(2)

with info_col1:

    st.write(
        f"**학습에 사용한 영화:** {train_count:,}편"
    )

    st.write(
        f"**점수를 측정한 영화:** {test_count:,}편"
    )

    st.write(
        f"**전체 사용 영화:** {total_movies:,}편"
    )

    st.write(
        "**분할 방법:** 영화코드 순으로 정렬 → 10편마다 앞 3편 시험용"
    )

with info_col2:

    if period_start is not None:

        st.write(
            "**분석 기준 기간:** "
            f"{period_start.strftime('%Y-%m-%d')} ~ "
            f"{period_end.strftime('%Y-%m-%d')}"
        )

    st.write(
        "**모델:** 다중 선형 회귀"
    )

    st.write(
        "**범주형 변수:** One-Hot Encoding"
    )

    st.write(
        f"**선택한 변수:** {len(selected_features)}개"
    )


# =========================================================
# 선택 변수 목록
# =========================================================

with st.expander("🔎 현재 모델에 사용된 변수 보기"):

    for feature in selected_features:

        st.write(
            f"• {feature}"
        )


# =========================================================
# 데이터 원본 정보
# =========================================================

with st.expander("📁 데이터 구조 보기"):

    st.write(
        f"일별 박스오피스 데이터: {len(daily_df):,}행"
    )

    st.write(
        f"영화 정보 데이터: {len(movies_df):,}편"
    )

    st.write(
        f"두 데이터는 **movieCd(영화코드)**를 기준으로 결합했습니다."
    )
