import json
import html
import random
import sqlite3
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import streamlit as st


# ====================
# 配置区
# ====================

APP_TITLE = "AI 协作任务研究"
APP_SUBTITLE = "AI-Assisted Task Workspace Study"
DEBUG = False

DEFAULT_MODEL_NAME = "deepseek-v4-flash"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
TEMPERATURE = 0.3
MAX_TOKENS = 700
MAX_TURNS_PER_TASK = 6

CONDITIONS = ["control", "static_feedback", "dynamic_feedback"]

ENERGY_PER_TOKEN_WH = 0.00005
LED_POWER_W = 0.6
TASK_COPY_THRESHOLD = 0.60

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "experiment.db"

TASK_KNOWN_NOTICE = (
    "AI 已经可以看到本任务的基础背景材料，但看不到你的最终提交要求、评分标准和个人偏好。"
    "你不需要复制背景材料全文，但需要告诉 AI 你希望它如何帮助你，以及最终答案需要满足哪些要求。"
)

SYSTEM_PROMPT = (
    "你是一个实验系统中的 AI 助手。你的任务是帮助用户完成当前任务。\n"
    "请根据用户提出的问题和当前任务材料，给出清晰、简洁、实用的回答。\n"
    "如果用户的问题信息不足，可以提醒用户补充必要信息。\n"
    "不要提及实验设计、研究目的、能源消耗、碳排放、环保提示或用户所在实验组。\n"
    "不要主动评价用户的提问方式。\n"
    "回答应尽量简洁。除非用户明确要求展开，否则每次回复控制在 300 个中文字符以内。"
    "对于代码任务，可以给出必要代码和简短解释，但不要写长篇教程。"
)

TASKS = [
    {
        "id": "writing",
        "title": "任务一：撰写延期邮件",
        "ai_visible_context": (
            "你正在帮助一名学生处理延期提交作业的邮件。\n\n"
            "基础背景：\n"
            "1. 作业原定今晚 23:59 截止。\n"
            "2. 学生明天上午需要参加学院创新项目展示。\n"
            "3. 学生负责展示 PPT 的最后整合。\n"
            "4. 学生希望延期 3 天。\n"
            "5. 学生已经完成了大约 70% 的作业。\n"
            "6. 课程老师平时比较重视提前沟通和具体计划。"
        ),
        "participant_only_requirements": (
            "你的最终邮件需要满足以下要求：\n\n"
            "1. 邮件长度控制在 150-200 字；\n"
            "2. 语气礼貌、简洁、可信；\n"
            "3. 不要显得像在找借口；\n"
            "4. 不要过度卑微；\n"
            "5. 不要提家庭原因；\n"
            "6. 不要提身体不舒服；\n"
            "7. 不要承诺“以后绝不再犯”；\n"
            "8. 要说明你已经完成约 70%；\n"
            "9. 要提出明确的新提交时间；\n"
            "10. 不希望邮件听起来太像 AI 生成。\n\n"
            "AI 只能看到基础背景材料，看不到以上最终提交要求。你需要通过提问告诉 AI 你的具体要求。"
        ),
        "final_answer_label": "请提交你认为最适合发给老师的一版邮件。",
    },
    {
        "id": "summary",
        "title": "任务二：总结高校 AI 使用材料",
        "ai_visible_context": (
            "你正在帮助学校整理一段关于“生成式 AI 进入高校学习场景”的材料。"
            "AI 已经可以看到下面这段原始材料。\n\n"
            "近年来，生成式人工智能逐渐进入高校学习场景。学生可以使用 AI 进行资料整理、文章润色、代码调试和观点生成。"
            "一方面，AI 工具提高了学习效率，帮助学生更快理解复杂内容；另一方面，过度依赖 AI 也可能削弱学生的独立思考能力，"
            "并带来学术诚信问题。部分教师主张完全禁止 AI 参与作业，认为这会破坏公平性；也有教师认为，与其禁止，"
            "不如引导学生合理使用 AI，并要求他们披露使用方式。高校未来需要在效率、诚信、学习能力培养之间取得平衡。"
        ),
        "participant_only_requirements": (
            "你的最终答案需要包括三部分：\n"
            "1. 面向普通学生的 120 字以内摘要；\n"
            "2. 面向教师或管理者的三条风险提示；\n"
            "3. 一条你自己的判断或建议。\n\n"
            "其他要求：\n"
            "- 语言要通俗，不要像政策文件；\n"
            "- 风险提示不能只重复“学术诚信”，至少要覆盖学习能力、公平性或教学管理中的两个方面。"
        ),
        "final_answer_label": "请提交你的摘要、三条风险提示和一条判断或建议。",
    },
    {
        "id": "code",
        "title": "任务三：解释并修正 Python 平均分代码",
        "ai_visible_context": (
            "下面这段 Python 代码本来想计算平均分，但结果不正确。\n\n"
            "```python\n"
            "scores = [78, 85, 92, 66, 88]\n"
            "total = 0\n"
            "for i in range(len(scores)):\n"
            "    total = scores[i]\n"
            "average = total / len(scores)\n"
            "print(average)\n"
            "```"
        ),
        "participant_only_requirements": (
            "最终你需要提交：\n"
            "1. 错误原因；\n"
            "2. 修正后的代码；\n"
            "3. 一句适合 Python 初学者理解的解释；\n"
            "4. 修正代码最好保留 for 循环写法，不要只给 sum(scores) 的简写；\n"
            "5. 语言要简洁，不要写成长篇教程。\n\n"
            "AI 只能看到代码和基础问题，看不到以上最终提交要求。你需要通过提问告诉 AI 的具体输出要求。"
        ),
        "final_answer_label": "请提交错误原因、修正代码和一句适合初学者理解的解释。",
    },
]

TASK_REQUIREMENTS = {
    "writing": [
        ("length_150_200", "邮件长度控制在 150-200 字"),
        ("polite_concise_credible", "语气礼貌、简洁、可信"),
        ("not_excuse_like", "不显得像在找借口"),
        ("not_overly_submissive", "不过度卑微"),
        ("no_family_reason", "不提家庭原因"),
        ("no_health_reason", "不提身体不舒服"),
        ("no_never_again_promise", "不承诺“以后绝不再犯”"),
        ("mention_70_percent_done", "说明已经完成约 70%"),
        ("specific_new_deadline", "提出明确的新提交时间"),
        ("not_ai_like", "不希望邮件听起来太像 AI 生成"),
    ],
    "summary": [
        ("student_facing_summary_under_120", "给普通学生看的 120 字以内摘要"),
        ("three_risk_points", "包含三条风险提示"),
        ("teacher_or_manager_audience", "风险提示面向教师或管理者"),
        ("own_judgment_or_suggestion", "包含自己的判断或建议"),
        ("plain_language", "语言通俗"),
        ("not_policy_like", "不要像政策文件"),
        ("risk_beyond_academic_integrity", "风险提示不能只重复学术诚信"),
        ("covers_learning_ability_or_fairness_or_management", "至少覆盖学习能力、公平性或教学管理中的两个方面"),
    ],
    "code": [
        ("identify_error_reason", "说明错误原因"),
        ("provide_corrected_code", "提供修正后的代码"),
        ("beginner_friendly_explanation", "包含适合 Python 初学者理解的解释"),
        ("prefer_for_loop_fix", "修正代码最好保留 for 循环写法"),
        ("avoid_only_sum_shortcut", "不要只给 sum(scores) 的简写"),
        ("concise_explanation", "语言简洁，不写成长篇教程"),
    ],
}


# ====================
# 通用工具
# ====================

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def get_query_param(name: str, default=None):
    value = st.query_params.get(name, default)
    if isinstance(value, list):
        return value[0] if value else default
    return value


def rerun():
    st.rerun()


def get_secret(name: str, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def clean_secret(value) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    placeholder_markers = ["your_", "你的", "change-this"]
    if any(marker in value for marker in placeholder_markers):
        return ""
    return value


def likert(label: str, key: str, value: int | None = 4) -> int | None:
    return st.radio(
        label,
        options=[1, 2, 3, 4, 5, 6, 7],
        index=None if value is None else value - 1,
        horizontal=True,
        key=key,
        help="1 = 非常不同意，7 = 非常同意",
    )


def show_estimation_notice():
    st.info("本实验中的能耗数值为研究目的下的估算反馈，不代表 DeepSeek 或任何具体 AI 平台的真实能耗。")


def render_likert_instruction():
    st.caption("1 = 非常不同意，7 = 非常同意。请根据你的真实想法选择。")


CARD_PALETTES = {
    "default": {"bg": "#FFFFFF", "border": "#E5E7EB", "title": "#111827"},
    "info": {"bg": "#EFF6FF", "border": "#BFDBFE", "title": "#1E40AF"},
    "warning": {"bg": "#FFFBEB", "border": "#FCD34D", "title": "#92400E"},
    "energy": {"bg": "#EEF2FF", "border": "#C7D2FE", "title": "#3730A3"},
    "success": {"bg": "#ECFDF5", "border": "#A7F3D0", "title": "#065F46"},
}


def escape_html(value) -> str:
    return html.escape(str(value or ""), quote=True)


def render_card(title: str, body: str, tone: str = "default", max_height: int | None = None):
    p = CARD_PALETTES.get(tone, CARD_PALETTES["default"])
    title_html = escape_html(title)
    body_html = escape_html(body).replace("\n", "<br>")
    card_styles = [
        f"background-color: {p['bg']}",
        f"border: 1px solid {p['border']}",
        "border-radius: 12px",
        "padding: 18px 20px",
        "margin: 14px 0",
        "line-height: 1.65",
    ]
    if max_height:
        card_styles.extend([f"max-height: {int(max_height)}px", "overflow-y: auto"])
    title_styles = [
        "font-weight: 700",
        f"color: {p['title']}",
        "font-size: 1.05rem",
        "margin-bottom: 8px",
    ]
    body_styles = ["color: #111827", "font-size: 0.98rem"]
    st.markdown(
        (
            f'<div style="{"; ".join(card_styles)};">'
            f'<div style="{"; ".join(title_styles)};">{title_html}</div>'
            f'<div style="{"; ".join(body_styles)};">{body_html}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_bullet_card(
    title: str,
    items: list[str],
    tone: str = "default",
    intro: str | None = None,
    max_height: int | None = None,
):
    p = CARD_PALETTES.get(tone, CARD_PALETTES["default"])
    title_html = escape_html(title)
    intro_html = f'<div style="margin-bottom: 8px;">{escape_html(intro)}</div>' if intro else ""
    items_html = "".join(f"<li>{escape_html(item)}</li>" for item in items)
    card_styles = [
        f"background-color: {p['bg']}",
        f"border: 1px solid {p['border']}",
        "border-radius: 12px",
        "padding: 18px 20px",
        "margin: 14px 0",
        "line-height: 1.65",
    ]
    if max_height:
        card_styles.extend([f"max-height: {int(max_height)}px", "overflow-y: auto"])
    title_styles = [
        "font-weight: 700",
        f"color: {p['title']}",
        "font-size: 1.05rem",
        "margin-bottom: 8px",
    ]
    st.markdown(
        (
            f'<div style="{"; ".join(card_styles)};">'
            f'<div style="{"; ".join(title_styles)};">{title_html}</div>'
            f'<div style="color: #111827; font-size: 0.98rem;">'
            f"{intro_html}"
            '<ul style="margin: 0 0 0 1.2rem; padding: 0;">'
            f"{items_html}"
            "</ul>"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_energy_summary_card(cumulative_wh: float, cumulative_led: float):
    card_styles = (
        "background-color: #EEF2FF; border: 1px solid #C7D2FE; border-radius: 12px; "
        "padding: 12px 14px; margin: 0 0 12px 0; line-height: 1.5"
    )
    st.markdown(
        (
            f'<div style="{card_styles};">'
            '<div style="font-weight: 700; color: #3730A3; margin-bottom: 4px;">本任务累计能耗</div>'
            '<div style="color: #374151;">'
            f"本任务累计估算能耗：<strong>{cumulative_wh:.4f} Wh</strong>｜"
            f"约等于 LED 灯点亮 <strong>{cumulative_led:.1f} 分钟</strong>"
            "</div></div>"
        ),
        unsafe_allow_html=True,
    )


def render_compact_energy_feedback(energy_wh: float, led_minutes: float, cumulative_wh: float, cumulative_led: float):
    card_styles = (
        "background-color: #EEF2FF; border: 1px solid #C7D2FE; border-radius: 10px; "
        "padding: 8px 12px; margin: 6px 0 12px 0; color: #3730A3; "
        "font-size: 0.9rem; line-height: 1.45"
    )
    st.markdown(
        (
            f'<div style="{card_styles};">'
            f"⚡ 本次估算：<strong>{energy_wh:.4f} Wh</strong> · LED <strong>{led_minutes:.1f} 分钟</strong>｜"
            f"累计：<strong>{cumulative_wh:.4f} Wh</strong> · LED <strong>{cumulative_led:.1f} 分钟</strong>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_ai_visible_context(task: dict):
    context = task["ai_visible_context"]
    if task["id"] != "code" or "```python" not in context:
        render_card("以下背景材料已经提供给 AI。你不需要重复复制。", context, "info", max_height=520)
        return

    before, _, rest = context.partition("```python")
    code, _, after = rest.partition("```")
    render_card("AI 已知信息", before.strip(), "info", max_height=260)
    st.code(code.strip(), language="python")
    if after.strip():
        render_card("补充说明", after.strip(), "info", max_height=220)


def render_workspace_styles():
    st.markdown(
        (
            "<style>"
            ".block-container{max-width:1320px;padding-top:2rem;padding-bottom:3rem;margin-left:auto;margin-right:auto;}"
            "h1{font-size:2.05rem!important;line-height:1.2!important;}"
            "[data-testid='stMarkdownContainer'] p{line-height:1.65;}"
            "</style>"
        ),
        unsafe_allow_html=True,
    )


def task_context_message(ai_visible_context: str) -> dict:
    return {
        "role": "user",
        "content": (
            "当前任务的基础背景材料如下。用户接下来会请求你帮助完成该任务。"
            "请基于这些基础材料回答用户的问题，但你看不到用户的最终提交要求、评分标准或个人偏好，"
            "除非用户在对话中主动告诉你。\n\n"
            f"{ai_visible_context}"
        ),
    }


# ====================
# DeepSeek 调用
# ====================

def get_deepseek_api_key() -> str:
    return clean_secret(get_secret("DEEPSEEK_API_KEY", "")) or clean_secret(get_secret("AI_API_KEY", ""))


def get_model_name() -> str:
    configured_model = clean_secret(get_secret("MODEL_NAME", "")) or DEFAULT_MODEL_NAME
    if configured_model == "deepseek-v3":
        return DEFAULT_MODEL_NAME
    return configured_model


def get_deepseek_base_url() -> str:
    configured_url = (
        clean_secret(get_secret("DEEPSEEK_BASE_URL", ""))
        or clean_secret(get_secret("AI_BASE_URL", ""))
        or DEFAULT_DEEPSEEK_BASE_URL
    ).rstrip("/")

    # 用户有时会误填 DeepSeek 官网地址；API endpoint 应使用 api.deepseek.com。
    if configured_url in {"https://www.deepseek.com", "https://deepseek.com"}:
        return DEFAULT_DEEPSEEK_BASE_URL
    return configured_url


def get_deepseek_client():
    from openai import OpenAI

    return OpenAI(api_key=get_deepseek_api_key(), base_url=get_deepseek_base_url())


def call_deepseek_with_http(messages: list[dict]) -> tuple[str, dict, float]:
    payload = {
        "model": get_model_name(),
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url=f"{get_deepseek_base_url()}/chat/completions",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {get_deepseek_api_key()}",
            "Content-Type": "application/json",
        },
    )

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"DeepSeek HTTP {exc.code}: {error_body[:300]}") from exc
    latency = time.perf_counter() - start

    ai_response = data["choices"][0]["message"].get("content", "")
    usage = data.get("usage") or {}
    return ai_response, usage, latency


def call_deepseek(ai_visible_context: str, visible_history: list[dict]) -> tuple[str, object, float]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append(task_context_message(ai_visible_context))
    messages.extend(visible_history)

    try:
        start = time.perf_counter()
        response = get_deepseek_client().chat.completions.create(
            model=get_model_name(),
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        latency = time.perf_counter() - start
        ai_response = response.choices[0].message.content or ""
        return ai_response, response.usage, latency
    except ImportError:
        return call_deepseek_with_http(messages)


# ====================
# 能耗估算
# ====================

def estimate_energy(total_tokens=None, user_prompt=None, ai_response=None) -> tuple[float, float, int]:
    if total_tokens is None:
        user_prompt = user_prompt or ""
        ai_response = ai_response or ""
        total_tokens = round((len(user_prompt) + len(ai_response)) / 2)

    total_tokens = int(total_tokens or 0)
    estimated_energy_wh = total_tokens * ENERGY_PER_TOKEN_WH
    estimated_led_minutes = estimated_energy_wh / LED_POWER_W * 60
    return estimated_energy_wh, estimated_led_minutes, total_tokens


def usage_to_tokens(usage, user_prompt: str, ai_response: str) -> tuple[int, int, int]:
    if isinstance(usage, dict) and usage.get("total_tokens"):
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or 0)
        return input_tokens, output_tokens, total_tokens

    if usage and getattr(usage, "total_tokens", None):
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        return input_tokens, output_tokens, total_tokens

    estimated_total_tokens = round((len(user_prompt) + len(ai_response)) / 2)
    estimated_input_tokens = round(len(user_prompt) / 2)
    estimated_output_tokens = max(estimated_total_tokens - estimated_input_tokens, 0)
    return estimated_input_tokens, estimated_output_tokens, estimated_total_tokens


def build_feedback_text(energy_wh: float, led_minutes: float, cumulative_wh: float, cumulative_led: float) -> str:
    return (
        f"本次 AI 回复估算能耗：{energy_wh:.4f} Wh；约等于小型 LED 灯点亮 {led_minutes:.1f} 分钟。"
        f"本任务累计估算能耗：{cumulative_wh:.4f} Wh；约等于小型 LED 灯累计点亮 {cumulative_led:.1f} 分钟。"
        "注：以上数值为实验估算，不代表真实平台能耗。"
    )


# ====================
# 复制任务检测
# ====================

def calculate_task_copy_similarity(user_prompt: str, task_instruction: str) -> float:
    user_prompt = (user_prompt or "").strip()
    task_instruction = (task_instruction or "").strip()
    if not user_prompt:
        return 0.0
    return SequenceMatcher(None, user_prompt, task_instruction).ratio()


# ====================
# 表单验证
# ====================

def birth_year_options() -> list:
    return ["请选择年份"] + list(range(1950, datetime.now().year + 1))


def birth_month_options() -> list:
    return ["请选择月份"] + list(range(1, 13))


def get_task_index(task_id: str) -> int:
    for index, task in enumerate(TASKS, start=1):
        if task["id"] == task_id:
            return index
    return 0


def validate_pre_survey(gender, education, ai_usage_frequency, birth_year, birth_month) -> tuple[bool, list[str]]:
    errors = []
    if not isinstance(birth_year, int):
        errors.append("出生年份")
    if not isinstance(birth_month, int):
        errors.append("出生月份")

    if gender == "请选择":
        errors.append("性别")
    if education == "请选择":
        errors.append("教育程度")
    if ai_usage_frequency == "请选择":
        errors.append("AI 使用频率")

    return len(errors) == 0, errors


def validate_task_submission(final_answer) -> tuple[bool, str | None]:
    final_answer = final_answer or ""
    if not final_answer.strip():
        return False, "请填写本任务的最终答案。"
    if len(final_answer.strip()) < 20:
        return False, "最终答案过短，请填写更完整的答案。"
    return True, None


def validate_post_survey(
    open_awareness,
    open_behavior_change,
    open_feedback_preference,
    open_productive_iteration,
    open_feedback_fatigue,
    open_design_suggestion,
) -> tuple[bool, list[str]]:
    errors = []
    if len((open_behavior_change or "").strip()) < 5:
        errors.append("提问方式变化说明")
    if len((open_feedback_preference or "").strip()) < 5:
        errors.append("能耗提示偏好说明")
    if len((open_productive_iteration or "").strip()) < 5:
        errors.append("有价值追问放弃说明")
    return len(errors) == 0, errors


# ====================
# 数据库初始化和保存
# ====================

def get_db() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_columns(conn: sqlite3.Connection, table_name: str, required_columns: dict[str, str]):
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
    for column_name, column_type in required_columns.items():
        if column_name not in existing:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")


def init_db():
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS participants (
                participant_id TEXT PRIMARY KEY,
                condition TEXT,
                created_at TEXT,
                completed INTEGER DEFAULT 0,
                age TEXT,
                birth_year INTEGER,
                birth_month INTEGER,
                gender TEXT,
                education TEXT,
                ai_usage_frequency TEXT,
                baseline_energy_awareness REAL,
                environmental_attitude REAL,
                attention_check_pass INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                participant_id TEXT,
                condition TEXT,
                task_id TEXT,
                task_index INTEGER,
                turn_id INTEGER,
                user_prompt TEXT,
                ai_response TEXT,
                prompt_timestamp TEXT,
                response_timestamp TEXT,
                latency_sec REAL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER,
                estimated_energy_wh REAL,
                estimated_led_minutes REAL,
                cumulative_energy_wh REAL,
                cumulative_led_minutes REAL,
                task_copy_similarity REAL,
                is_task_copying INTEGER,
                context_copy_similarity REAL,
                requirements_copy_similarity REAL,
                is_context_copying INTEGER,
                is_requirements_copying INTEGER,
                feedback_displayed INTEGER,
                feedback_text TEXT,
                model_name TEXT,
                temperature REAL,
                max_tokens INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS task_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                participant_id TEXT,
                condition TEXT,
                task_id TEXT,
                task_index INTEGER,
                task_start_time TEXT,
                task_end_time TEXT,
                task_duration_sec REAL,
                request_count INTEGER,
                final_answer TEXT,
                satisfaction INTEGER,
                perceived_efficiency INTEGER,
                cognitive_load INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS questionnaires (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                participant_id TEXT,
                condition TEXT,
                submitted_at TEXT,
                q_ai_use_1 INTEGER,
                q_ai_use_2 INTEGER,
                q_energy_awareness_1 INTEGER,
                q_energy_awareness_2 INTEGER,
                q_energy_awareness_3 INTEGER,
                q_environment_1 INTEGER,
                q_environment_2 INTEGER,
                q_low_energy_wait INTEGER,
                q_low_energy_quality INTEGER,
                q_default_low_energy INTEGER,
                q_feedback_attention INTEGER,
                q_feedback_understandability INTEGER,
                q_feedback_helpfulness INTEGER,
                q_feedback_intrusiveness INTEGER,
                q_feedback_pressure INTEGER,
                q_feedback_guilt INTEGER,
                q_interaction_penalty INTEGER,
                q_feedback_fatigue INTEGER,
                q_long_term_acceptance INTEGER,
                q_task_dependent_feedback INTEGER,
                open_awareness TEXT,
                open_behavior_change TEXT,
                open_feedback_preference TEXT,
                open_productive_iteration TEXT,
                open_feedback_fatigue TEXT,
                open_design_suggestion TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                participant_id TEXT,
                condition TEXT,
                task_id TEXT,
                event_type TEXT,
                event_time TEXT,
                event_value TEXT
            )
            """
        )
        ensure_columns(
            conn,
            "messages",
            {
                "task_copy_similarity": "REAL",
                "is_task_copying": "INTEGER",
                "context_copy_similarity": "REAL",
                "requirements_copy_similarity": "REAL",
                "is_context_copying": "INTEGER",
                "is_requirements_copying": "INTEGER",
                "task_index": "INTEGER",
            },
        )
        ensure_columns(conn, "task_sessions", {"task_index": "INTEGER"})
        ensure_columns(
            conn,
            "questionnaires",
            {
                "q_feedback_attention": "INTEGER",
                "q_feedback_understandability": "INTEGER",
                "q_feedback_helpfulness": "INTEGER",
                "q_feedback_intrusiveness": "INTEGER",
                "q_feedback_pressure": "INTEGER",
                "q_feedback_guilt": "INTEGER",
                "q_interaction_penalty": "INTEGER",
                "q_feedback_fatigue": "INTEGER",
                "q_long_term_acceptance": "INTEGER",
                "q_task_dependent_feedback": "INTEGER",
                "open_productive_iteration": "TEXT",
                "open_feedback_fatigue": "TEXT",
                "open_design_suggestion": "TEXT",
            },
        )
        ensure_columns(
            conn,
            "participants",
            {
                "birth_year": "INTEGER",
                "birth_month": "INTEGER",
            },
        )
        conn.commit()


def get_existing_participant(participant_id: str):
    with get_db() as conn:
        return conn.execute(
            "SELECT * FROM participants WHERE participant_id = ?",
            (participant_id,),
        ).fetchone()


def create_participant(participant_id: str, condition: str):
    with get_db() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO participants (participant_id, condition, created_at)
            VALUES (?, ?, ?)
            """,
            (participant_id, condition, now_iso()),
        )
        conn.commit()


def set_participant_condition(participant_id: str, condition: str):
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO participants (participant_id, condition, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(participant_id) DO UPDATE SET condition = excluded.condition
            """,
            (participant_id, condition, now_iso()),
        )
        conn.commit()


def update_pre_survey(data: dict):
    with get_db() as conn:
        conn.execute(
            """
            UPDATE participants
            SET birth_year = ?,
                birth_month = ?,
                gender = ?,
                education = ?,
                ai_usage_frequency = ?,
                baseline_energy_awareness = ?,
                environmental_attitude = ?,
                attention_check_pass = ?
            WHERE participant_id = ?
            """,
            (
                data["birth_year"],
                data["birth_month"],
                data["gender"],
                data["education"],
                data["ai_usage_frequency"],
                data["baseline_energy_awareness"],
                data["environmental_attitude"],
                data["attention_check_pass"],
                st.session_state.participant_id,
            ),
        )
        existing = conn.execute(
            """
            SELECT id FROM questionnaires
            WHERE participant_id = ? AND q_ai_use_1 IS NOT NULL
            ORDER BY id DESC LIMIT 1
            """,
            (st.session_state.participant_id,),
        ).fetchone()
        questionnaire_values = (
            st.session_state.participant_id,
            st.session_state.condition,
            now_iso(),
            data["q_ai_use_1"],
            data["q_ai_use_2"],
            data["q_energy_awareness_1"],
            data["q_energy_awareness_2"],
            data["q_energy_awareness_3"],
            data["q_environment_1"],
            data["q_environment_2"],
        )
        if existing:
            conn.execute(
                """
                UPDATE questionnaires
                SET participant_id = ?, condition = ?, submitted_at = ?,
                    q_ai_use_1 = ?, q_ai_use_2 = ?,
                    q_energy_awareness_1 = ?, q_energy_awareness_2 = ?, q_energy_awareness_3 = ?,
                    q_environment_1 = ?, q_environment_2 = ?
                WHERE id = ?
                """,
                questionnaire_values + (existing["id"],),
            )
        else:
            conn.execute(
                """
                INSERT INTO questionnaires (
                    participant_id, condition, submitted_at,
                    q_ai_use_1, q_ai_use_2,
                    q_energy_awareness_1, q_energy_awareness_2, q_energy_awareness_3,
                    q_environment_1, q_environment_2
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                questionnaire_values,
            )
        conn.commit()


def save_message(record: dict):
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO messages (
                participant_id, condition, task_id, task_index, turn_id, user_prompt, ai_response,
                prompt_timestamp, response_timestamp, latency_sec,
                input_tokens, output_tokens, total_tokens,
                estimated_energy_wh, estimated_led_minutes,
                cumulative_energy_wh, cumulative_led_minutes,
                task_copy_similarity, is_task_copying,
                context_copy_similarity, requirements_copy_similarity,
                is_context_copying, is_requirements_copying,
                feedback_displayed, feedback_text, model_name, temperature, max_tokens
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["participant_id"],
                record["condition"],
                record["task_id"],
                record["task_index"],
                record["turn_id"],
                record["user_prompt"],
                record["ai_response"],
                record["prompt_timestamp"],
                record["response_timestamp"],
                record["latency_sec"],
                record["input_tokens"],
                record["output_tokens"],
                record["total_tokens"],
                record["estimated_energy_wh"],
                record["estimated_led_minutes"],
                record["cumulative_energy_wh"],
                record["cumulative_led_minutes"],
                record["task_copy_similarity"],
                record["is_task_copying"],
                record.get("context_copy_similarity"),
                record.get("requirements_copy_similarity"),
                record.get("is_context_copying"),
                record.get("is_requirements_copying"),
                record["feedback_displayed"],
                record["feedback_text"],
                record["model_name"],
                record["temperature"],
                record["max_tokens"],
            ),
        )
        conn.commit()


def save_task_session(task_id: str, final_answer: str, satisfaction: int, perceived_efficiency: int, cognitive_load: int):
    start_time = st.session_state.task_start_times.get(task_id, now_iso())
    end_time = now_iso()
    try:
        start_dt = datetime.fromisoformat(start_time)
        end_dt = datetime.fromisoformat(end_time)
        duration = (end_dt - start_dt).total_seconds()
    except ValueError:
        duration = None

    request_count = st.session_state.turn_counts.get(task_id, 0)
    task_index = get_task_index(task_id)
    with get_db() as conn:
        existing = conn.execute(
            """
            SELECT id FROM task_sessions
            WHERE participant_id = ? AND task_id = ?
            ORDER BY id DESC LIMIT 1
            """,
            (st.session_state.participant_id, task_id),
        ).fetchone()
        task_values = (
            st.session_state.participant_id,
            st.session_state.condition,
            task_id,
            task_index,
            start_time,
            end_time,
            duration,
            request_count,
            final_answer,
            satisfaction,
            perceived_efficiency,
            cognitive_load,
        )
        if existing:
            conn.execute(
                """
                UPDATE task_sessions
                SET participant_id = ?, condition = ?, task_id = ?, task_index = ?,
                    task_start_time = ?, task_end_time = ?, task_duration_sec = ?,
                    request_count = ?, final_answer = ?,
                    satisfaction = ?, perceived_efficiency = ?, cognitive_load = ?
                WHERE id = ?
                """,
                task_values + (existing["id"],),
            )
        else:
            conn.execute(
                """
                INSERT INTO task_sessions (
                    participant_id, condition, task_id, task_index,
                    task_start_time, task_end_time, task_duration_sec,
                    request_count, final_answer,
                    satisfaction, perceived_efficiency, cognitive_load
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                task_values,
            )
        conn.commit()


def save_post_survey(data: dict):
    with get_db() as conn:
        existing = conn.execute(
            """
            SELECT id FROM questionnaires
            WHERE participant_id = ? AND q_low_energy_wait IS NOT NULL
            ORDER BY id DESC LIMIT 1
            """,
            (st.session_state.participant_id,),
        ).fetchone()
        questionnaire_values = (
            st.session_state.participant_id,
            st.session_state.condition,
            now_iso(),
            data["q_low_energy_wait"],
            data["q_low_energy_quality"],
            data["q_default_low_energy"],
            data["q_feedback_attention"],
            data["q_feedback_understandability"],
            data["q_feedback_helpfulness"],
            data["q_feedback_intrusiveness"],
            data["q_feedback_pressure"],
            data["q_feedback_guilt"],
            data["q_interaction_penalty"],
            data["q_feedback_fatigue"],
            data["q_long_term_acceptance"],
            data["q_task_dependent_feedback"],
            data["open_awareness"],
            data["open_behavior_change"],
            data["open_feedback_preference"],
            data["open_productive_iteration"],
            data["open_feedback_fatigue"],
            data["open_design_suggestion"],
        )
        if existing:
            conn.execute(
                """
                UPDATE questionnaires
                SET participant_id = ?, condition = ?, submitted_at = ?,
                    q_low_energy_wait = ?, q_low_energy_quality = ?, q_default_low_energy = ?,
                    q_feedback_attention = ?, q_feedback_understandability = ?, q_feedback_helpfulness = ?,
                    q_feedback_intrusiveness = ?, q_feedback_pressure = ?, q_feedback_guilt = ?,
                    q_interaction_penalty = ?, q_feedback_fatigue = ?, q_long_term_acceptance = ?,
                    q_task_dependent_feedback = ?,
                    open_awareness = ?, open_behavior_change = ?, open_feedback_preference = ?,
                    open_productive_iteration = ?, open_feedback_fatigue = ?, open_design_suggestion = ?
                WHERE id = ?
                """,
                questionnaire_values + (existing["id"],),
            )
        else:
            conn.execute(
                """
                INSERT INTO questionnaires (
                    participant_id, condition, submitted_at,
                    q_low_energy_wait, q_low_energy_quality, q_default_low_energy,
                    q_feedback_attention, q_feedback_understandability, q_feedback_helpfulness,
                    q_feedback_intrusiveness, q_feedback_pressure, q_feedback_guilt,
                    q_interaction_penalty, q_feedback_fatigue, q_long_term_acceptance,
                    q_task_dependent_feedback,
                    open_awareness, open_behavior_change, open_feedback_preference,
                    open_productive_iteration, open_feedback_fatigue, open_design_suggestion
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                questionnaire_values,
            )
        conn.execute(
            "UPDATE participants SET completed = 1 WHERE participant_id = ?",
            (st.session_state.participant_id,),
        )
        conn.commit()


def log_event(event_type: str, event_value: str = "", task_id: str | None = None):
    participant_id = st.session_state.get("participant_id", "")
    condition = st.session_state.get("condition", "")
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO events (participant_id, condition, task_id, event_type, event_time, event_value)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (participant_id, condition, task_id, event_type, now_iso(), str(event_value)),
        )
        conn.commit()


# ====================
# 派生导出
# ====================

def make_requirement_checklist_df() -> pd.DataFrame:
    rows = []
    for task_id, requirements in TASK_REQUIREMENTS.items():
        for key, description in requirements:
            rows.append(
                {
                    "task_id": task_id,
                    "requirement_key": key,
                    "requirement_description": description,
                }
            )
    return pd.DataFrame(rows)


def make_prompt_coding_file() -> pd.DataFrame:
    with get_db() as conn:
        df = pd.read_sql_query(
            """
            SELECT
                participant_id, condition, task_id, task_index, turn_id,
                user_prompt, ai_response AS next_ai_response,
                task_copy_similarity, is_task_copying
            FROM messages
            ORDER BY participant_id, task_index, turn_id
            """,
            conn,
        )
    if df.empty:
        return df

    df["previous_ai_response"] = (
        df.sort_values(["participant_id", "task_index", "turn_id"])
        .groupby(["participant_id", "task_id"])["next_ai_response"]
        .shift(1)
        .fillna("")
    )
    df = df.sample(frac=1, random_state=20260507).reset_index(drop=True)
    df.insert(0, "blind_prompt_id", [f"BP{i:06d}" for i in range(1, len(df) + 1)])
    return df[
        [
            "blind_prompt_id",
            "participant_id",
            "task_id",
            "task_index",
            "turn_id",
            "user_prompt",
            "previous_ai_response",
            "next_ai_response",
            "task_copy_similarity",
            "is_task_copying",
            "condition",
        ]
    ]


def make_prompt_coding_template() -> pd.DataFrame:
    source = make_prompt_coding_file()
    if source.empty:
        return source
    template = source[
        [
            "blind_prompt_id",
            "task_id",
            "task_index",
            "turn_id",
            "user_prompt",
            "previous_ai_response",
            "next_ai_response",
        ]
    ].copy()
    for column in [
        "prompt_type",
        "requirement_transmission_count",
        "productive_iteration",
        "low_value_prompt",
        "mechanical_copying",
        "strategic_prompting_score",
        "coder_id",
        "coder_notes",
    ]:
        template[column] = ""
    return template


def make_blind_rating_exports() -> tuple[pd.DataFrame, pd.DataFrame]:
    with get_db() as conn:
        df = pd.read_sql_query(
            """
            SELECT participant_id, condition, task_id, task_index, final_answer
            FROM task_sessions
            ORDER BY participant_id, task_index
            """,
            conn,
        )
    if df.empty:
        return df, df
    df = df.sample(frac=1, random_state=20260507).reset_index(drop=True)
    df.insert(0, "blind_task_id", [f"BT{i:06d}" for i in range(1, len(df) + 1)])
    rating_file = df[["blind_task_id", "task_id", "task_index", "final_answer"]].copy()
    rating_file["overall_quality"] = ""
    rating_file["requirement_hit_rate"] = ""
    rating_file["rater_id"] = ""
    rating_file["rater_notes"] = ""
    key_file = df[["blind_task_id", "participant_id", "condition", "task_id", "task_index"]].copy()
    return rating_file, key_file


def make_ai_scoring_input() -> pd.DataFrame:
    rating_file, key_file = make_blind_rating_exports()
    if rating_file.empty:
        return rating_file
    merged = rating_file[["blind_task_id", "task_id", "task_index", "final_answer"]].copy()
    merged["participant_only_requirements"] = merged["task_id"].map(
        {task["id"]: task["participant_only_requirements"] for task in TASKS}
    )
    return merged


# ====================
# Session state 初始化
# ====================

def init_participant_session():
    if "participant_id" not in st.session_state:
        pid_from_url = get_query_param("pid")
        if pid_from_url:
            participant_id = str(pid_from_url).strip()
        else:
            participant_id = f"P_{uuid.uuid4().hex[:10]}"
        st.session_state.participant_id = participant_id

    forced_condition = get_query_param("condition")
    if forced_condition in CONDITIONS:
        st.session_state.condition = forced_condition
        set_participant_condition(st.session_state.participant_id, forced_condition)
        return

    existing = get_existing_participant(st.session_state.participant_id)
    if existing:
        st.session_state.condition = existing["condition"]
    elif "condition" not in st.session_state:
        st.session_state.condition = random.choice(CONDITIONS)
        create_participant(st.session_state.participant_id, st.session_state.condition)
    else:
        create_participant(st.session_state.participant_id, st.session_state.condition)


def init_session_state():
    init_participant_session()
    defaults = {
        "page": "consent",
        "current_task_index": 0,
        "histories": {},
        "turn_counts": {},
        "cumulative_energy": {},
        "cumulative_led": {},
        "task_start_times": {},
        "last_feedback": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    for task in TASKS:
        task_id = task["id"]
        st.session_state.histories.setdefault(task_id, [])
        st.session_state.turn_counts.setdefault(task_id, 0)
        st.session_state.cumulative_energy.setdefault(task_id, 0.0)
        st.session_state.cumulative_led.setdefault(task_id, 0.0)


def ensure_api_key_for_experiment():
    if not get_deepseek_api_key():
        st.error(
            "未配置 API Key。请在 Streamlit secrets 或本地 .streamlit/secrets.toml 中配置 "
            "DEEPSEEK_API_KEY 或 AI_API_KEY 后再运行实验。"
        )
        st.stop()


def debug_sidebar():
    if DEBUG:
        with st.sidebar:
            st.write("DEBUG")
            st.write("participant_id:", st.session_state.participant_id)
            st.write("condition:", st.session_state.condition)
            st.write("page:", st.session_state.page)


# ====================
# 页面函数
# ====================

def page_consent():
    st.subheader("知情同意")
    st.write("你将完成 3 个简短任务，并可以与 AI 助手对话来帮助完成任务。")

    render_bullet_card(
        "你将做什么",
        [
            "你将依次完成 3 个任务；",
            "每个任务中，你可以与 AI 助手对话获得帮助；",
            f"每个任务最多可以向 AI 提问 {MAX_TURNS_PER_TASK} 轮；",
            "每个任务结束时，你需要提交一个最终答案；",
            "完成全部任务后，你将填写一份简短问卷。",
        ],
        "info",
    )
    render_card(
        "研究数据说明",
        (
            "本研究会记录你的问卷回答、任务操作过程、你输入给 AI 的问题、AI 回复内容、"
            "任务最终答案，以及与任务完成相关的时间和交互信息。"
            "所有数据将以匿名编号保存，仅用于学术研究分析。"
        ),
    )
    render_card(
        "隐私与安全",
        (
            "你的输入内容会发送给第三方大模型 API，用于生成 AI 回复。"
            "请不要输入真实姓名、手机号、身份证号、住址、账号密码、商业机密或其他敏感信息。"
        ),
        "warning",
    )
    render_card(
        "估算反馈说明",
        (
            "部分界面可能包含与 AI 使用过程相关的辅助提示。"
            "如界面中出现任何估算数值，这些数值仅为研究反馈，不代表真实平台测量。"
        ),
        "info",
    )
    agreed = st.checkbox("我已阅读并同意参与本研究")
    if st.button("进入前测问卷", type="primary"):
        if not agreed:
            st.warning("请先勾选知情同意后再进入下一步。")
            return
        log_event("consent_accepted")
        st.session_state.page = "pre_survey"
        rerun()


def page_pre_survey():
    st.header("前测问卷")
    render_card(
        "填写说明",
        "本页用于了解你的基本背景和 AI 使用经验。所有信息仅用于统计分析，不会用于识别个人身份。",
        "info",
    )

    with st.form("pre_survey_form"):
        st.subheader("基本信息")
        birth_col1, birth_col2 = st.columns(2)
        with birth_col1:
            birth_year = st.selectbox("出生年份", birth_year_options())
        with birth_col2:
            birth_month = st.selectbox("出生月份", birth_month_options(), format_func=lambda value: f"{value} 月" if isinstance(value, int) else value)
        gender = st.selectbox("性别", ["请选择", "女", "男", "非二元/其他", "不愿透露"])
        education = st.selectbox("教育程度", ["请选择", "高中及以下", "大专", "本科", "硕士", "博士及以上", "其他"])
        ai_usage_frequency = st.selectbox(
            "AI 使用频率",
            ["请选择", "几乎不用", "每月几次", "每周几次", "每天", "每天多次"],
        )

        st.subheader("AI 使用经验")
        render_likert_instruction()
        q_ai_use_1 = likert("我经常使用生成式 AI 工具完成学习、工作或生活任务。", "q_ai_use_1", value=None)
        q_ai_use_2 = likert("我认为自己比较擅长向 AI 提出清晰的问题。", "q_ai_use_2", value=None)

        st.subheader("AI 资源认知")
        render_likert_instruction()
        q_energy_awareness_1 = likert("我知道生成式 AI 每次回答都需要消耗计算资源。", "q_energy_awareness_1", value=None)
        q_energy_awareness_2 = likert("我知道 AI 数据中心可能消耗大量电力。", "q_energy_awareness_2", value=None)
        q_energy_awareness_3 = likert("在使用 AI 时，我通常会考虑它背后的能源成本。", "q_energy_awareness_3", value=None)

        st.subheader("环保与责任态度")
        render_likert_instruction()
        q_environment_1 = likert("我愿意在日常生活中采取行动减少能源浪费。", "q_environment_1", value=None)
        q_environment_2 = likert("我认为科技产品应该向用户披露环境影响。", "q_environment_2", value=None)

        st.subheader("注意力检查")
        render_likert_instruction()
        attention_check = likert("为了确认你认真阅读，请在本题选择 6。", "attention_check", value=None)

        submitted = st.form_submit_button("提交前测问卷", type="primary")

    if submitted:
        is_valid, errors = validate_pre_survey(gender, education, ai_usage_frequency, birth_year, birth_month)
        likert_answers = {
            "AI 使用经验题 1": q_ai_use_1,
            "AI 使用经验题 2": q_ai_use_2,
            "能源意识题 1": q_energy_awareness_1,
            "能源意识题 2": q_energy_awareness_2,
            "能源意识题 3": q_energy_awareness_3,
            "环境态度题 1": q_environment_1,
            "环境态度题 2": q_environment_2,
            "注意力检查题": attention_check,
        }
        errors.extend([label for label, answer in likert_answers.items() if answer is None])
        if not is_valid or errors:
            st.warning("请完成以下信息：" + "、".join(errors) + "。")
            return

        baseline_energy_awareness = sum(
            [q_energy_awareness_1, q_energy_awareness_2, q_energy_awareness_3]
        ) / 3
        environmental_attitude = sum([q_environment_1, q_environment_2]) / 2
        update_pre_survey(
            {
                "birth_year": birth_year,
                "birth_month": birth_month,
                "gender": gender,
                "education": education,
                "ai_usage_frequency": ai_usage_frequency,
                "q_ai_use_1": q_ai_use_1,
                "q_ai_use_2": q_ai_use_2,
                "q_energy_awareness_1": q_energy_awareness_1,
                "q_energy_awareness_2": q_energy_awareness_2,
                "q_energy_awareness_3": q_energy_awareness_3,
                "q_environment_1": q_environment_1,
                "q_environment_2": q_environment_2,
                "baseline_energy_awareness": baseline_energy_awareness,
                "environmental_attitude": environmental_attitude,
                "attention_check_pass": 1 if attention_check == 6 else 0,
            }
        )
        log_event("pre_survey_submitted", event_value=f"attention_check_pass={1 if attention_check == 6 else 0}")
        st.session_state.page = "instruction"
        rerun()


def page_instruction():
    st.header("实验说明")
    render_bullet_card(
        "任务规则",
        [
            "你将依次完成 3 个任务；",
            f"每个任务最多可以向 AI 提问 {MAX_TURNS_PER_TASK} 轮；",
            "三个任务之间的对话历史不会共享；",
            "每个任务结束时，你需要提交一个最终答案；",
            "AI 的回复不会自动成为最终答案，你需要自己决定最终提交内容。",
        ],
        "info",
    )
    render_card(
        "重要说明：AI 只知道部分任务信息",
        (
            "每个任务中，AI 已经可以看到基础背景材料。但是，AI 看不到你的最终提交要求、"
            "评分标准和个人偏好。如果你希望 AI 满足这些要求，需要在提问中主动告诉 AI。"
        ),
        "warning",
    )

    condition = st.session_state.condition
    if condition == "static_feedback":
        render_card(
            "能耗提示",
            (
                "每次 AI 请求都需要服务器计算，并会消耗一定电力。"
                "本实验中的能耗数值为研究目的下的估算反馈。你仍然可以自由使用 AI 完成任务。"
            ),
            "energy",
        )
    elif condition == "dynamic_feedback":
        render_card(
            "实时能耗反馈已开启",
            (
                "在接下来的任务中，每次 AI 回复后，系统会显示本次请求的估算能耗、"
                "当前任务的累计估算能耗，以及对应的 LED 灯点亮时间类比。"
                "你仍然可以自由使用 AI 完成任务。"
            ),
            "energy",
        )

    if st.button("开始任务", type="primary"):
        log_event("instruction_completed")
        st.session_state.page = "task"
        rerun()


def render_chat(task: dict, energy_summary_slot=None):
    task_id = task["id"]
    history = st.session_state.histories[task_id]

    for message in history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
        if (
            message.get("role") == "assistant"
            and st.session_state.condition == "dynamic_feedback"
            and message.get("estimated_energy_wh") is not None
        ):
            render_compact_energy_feedback(
                message.get("estimated_energy_wh", 0.0),
                message.get("estimated_led_minutes", 0.0),
                message.get("cumulative_energy_wh", 0.0),
                message.get("cumulative_led_minutes", 0.0),
            )

    turns_used = st.session_state.turn_counts[task_id]
    turns_left = MAX_TURNS_PER_TASK - turns_used
    st.caption(f"本任务剩余 AI 对话轮数：{turns_left}")

    if turns_left <= 0:
        st.warning("本任务的 AI 对话轮数已用完。请在下方提交最终答案。")
        return

    prompt = st.chat_input("输入你想问 AI 的问题")
    if not prompt:
        return

    prompt_timestamp = now_iso()
    next_turn = turns_used + 1
    context_similarity = calculate_task_copy_similarity(prompt, task["ai_visible_context"])
    requirements_similarity = calculate_task_copy_similarity(prompt, task["participant_only_requirements"])
    similarity = max(context_similarity, requirements_similarity)
    is_task_copying = 1 if similarity >= TASK_COPY_THRESHOLD else 0
    is_context_copying = 1 if context_similarity >= TASK_COPY_THRESHOLD else 0
    is_requirements_copying = 1 if requirements_similarity >= TASK_COPY_THRESHOLD else 0

    log_event(
        "prompt_submitted",
        event_value=(
            f"turn_id={next_turn};copy_similarity={similarity:.2f};"
            f"context_similarity={context_similarity:.2f};requirements_similarity={requirements_similarity:.2f}"
        ),
        task_id=task_id,
    )
    if is_task_copying:
        log_event("task_copy_detected", event_value=f"{similarity:.2f}", task_id=task_id)

    st.session_state.histories[task_id].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("AI 正在回复..."):
            try:
                ai_response, usage, latency = call_deepseek(task["ai_visible_context"], st.session_state.histories[task_id])
            except Exception as exc:
                st.error("AI 服务暂时不可用，请稍后重试。")
                log_event("api_error", event_value=str(exc)[:300], task_id=task_id)
                st.session_state.histories[task_id].pop()
                return
            st.write(ai_response)

    response_timestamp = now_iso()
    input_tokens, output_tokens, total_tokens = usage_to_tokens(usage, prompt, ai_response)
    energy_wh, led_minutes, total_tokens = estimate_energy(total_tokens, prompt, ai_response)
    st.session_state.cumulative_energy[task_id] += energy_wh
    st.session_state.cumulative_led[task_id] += led_minutes
    cumulative_wh = st.session_state.cumulative_energy[task_id]
    cumulative_led = st.session_state.cumulative_led[task_id]

    feedback_displayed = 1 if st.session_state.condition == "dynamic_feedback" else 0
    feedback_text = ""
    if feedback_displayed:
        feedback_text = build_feedback_text(energy_wh, led_minutes, cumulative_wh, cumulative_led)
        if energy_summary_slot is not None:
            with energy_summary_slot:
                render_energy_summary_card(cumulative_wh, cumulative_led)
        render_compact_energy_feedback(energy_wh, led_minutes, cumulative_wh, cumulative_led)
        st.session_state.last_feedback[task_id] = feedback_text
        log_event("energy_feedback_displayed", event_value=feedback_text, task_id=task_id)

    assistant_message = {"role": "assistant", "content": ai_response}
    if feedback_displayed:
        assistant_message.update(
            {
                "estimated_energy_wh": energy_wh,
                "estimated_led_minutes": led_minutes,
                "cumulative_energy_wh": cumulative_wh,
                "cumulative_led_minutes": cumulative_led,
            }
        )
    st.session_state.histories[task_id].append(assistant_message)
    st.session_state.turn_counts[task_id] = next_turn

    save_message(
        {
            "participant_id": st.session_state.participant_id,
            "condition": st.session_state.condition,
            "task_id": task_id,
            "task_index": get_task_index(task_id),
            "turn_id": next_turn,
            "user_prompt": prompt,
            "ai_response": ai_response,
            "prompt_timestamp": prompt_timestamp,
            "response_timestamp": response_timestamp,
            "latency_sec": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "estimated_energy_wh": energy_wh,
            "estimated_led_minutes": led_minutes,
            "cumulative_energy_wh": cumulative_wh,
            "cumulative_led_minutes": cumulative_led,
            "task_copy_similarity": similarity,
            "is_task_copying": is_task_copying,
            "context_copy_similarity": context_similarity,
            "requirements_copy_similarity": requirements_similarity,
            "is_context_copying": is_context_copying,
            "is_requirements_copying": is_requirements_copying,
            "feedback_displayed": feedback_displayed,
            "feedback_text": feedback_text,
            "model_name": get_model_name(),
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        }
    )
    log_event("ai_response_generated", event_value=f"turn_id={next_turn};total_tokens={total_tokens}", task_id=task_id)


def render_task_submission(task: dict):
    task_id = task["id"]
    st.divider()
    render_card(
        "最终提交答案",
        "请根据 AI 的帮助，提交你最终决定使用的答案。AI 的回复不会自动提交，你可以直接使用 AI 的内容，也可以修改后提交。",
        "default",
    )
    is_last_task = st.session_state.current_task_index >= len(TASKS) - 1
    button_label = "提交本任务并进入实验后问卷" if is_last_task else "提交本任务并进入下一任务"

    with st.form(f"final_form_{task_id}"):
        final_answer = st.text_area(task["final_answer_label"], height=180)
        st.subheader("本任务体验评价")
        render_likert_instruction()
        satisfaction = likert("我对本任务最终结果满意。", f"satisfaction_{task_id}")
        perceived_efficiency = likert("AI 帮助我高效完成了本任务。", f"perceived_efficiency_{task_id}")
        cognitive_load = likert("完成这个任务让我感到费力。", f"cognitive_load_{task_id}")
        submitted = st.form_submit_button(button_label, type="primary")

    if submitted:
        is_valid, error_message = validate_task_submission(final_answer)
        if not is_valid:
            st.warning(error_message or "请填写本任务的最终答案，且答案不能过短。")
            return

        save_task_session(
            task_id=task_id,
            final_answer=(final_answer or "").strip(),
            satisfaction=satisfaction,
            perceived_efficiency=perceived_efficiency,
            cognitive_load=cognitive_load,
        )
        log_event("task_submitted", event_value=f"request_count={st.session_state.turn_counts[task_id]}", task_id=task_id)

        if st.session_state.current_task_index < len(TASKS) - 1:
            st.session_state.current_task_index += 1
            st.session_state.page = "task"
        else:
            st.session_state.page = "post_survey"
        rerun()


def page_task():
    task = TASKS[st.session_state.current_task_index]
    task_id = task["id"]

    if task_id not in st.session_state.task_start_times:
        st.session_state.task_start_times[task_id] = now_iso()
        log_event("task_started", task_id=task_id)

    task_number = st.session_state.current_task_index + 1
    turns_used = st.session_state.turn_counts[task_id]
    turns_left = MAX_TURNS_PER_TASK - turns_used

    st.header(f"任务 {task_number}/{len(TASKS)}：{task['title']}")
    st.progress(task_number / len(TASKS))
    status_col1, status_col2, status_col3 = st.columns(3)
    status_col1.metric("任务进度", f"{task_number}/{len(TASKS)}")
    status_col2.metric("剩余 AI 对话轮次", f"{turns_left}/{MAX_TURNS_PER_TASK}")
    status_col3.metric("当前阶段", "AI 协作生成 → 最终答案提交")

    if st.session_state.condition == "static_feedback":
        render_card(
            "能耗提示",
            "每次 AI 请求都需要服务器计算，并会消耗一定电力。本实验中的能耗数值为研究目的下的估算反馈。",
            "energy",
        )
    energy_summary_slot = None
    if st.session_state.condition == "dynamic_feedback":
        energy_summary_slot = st.empty()
        with energy_summary_slot:
            render_energy_summary_card(
                st.session_state.cumulative_energy[task_id],
                st.session_state.cumulative_led[task_id],
            )

    left_col, middle_col, right_col = st.columns([1.0, 1.8, 1.0])

    with left_col:
        st.subheader("AI 已知信息")
        render_ai_visible_context(task)

    with middle_col:
        st.subheader("AI 对话区")
        render_card(
            "对话提示",
            "请向 AI 说明你希望它如何帮助你。一个有效请求通常包括：目标、输出格式、关键约束和排除项。",
            "default",
        )
        render_chat(task, energy_summary_slot)

    with right_col:
        st.subheader("你的提交要求")
        render_card(
            "AI 默认看不到这些要求。若希望 AI 满足它们，请在提问中主动说明。",
            task["participant_only_requirements"],
            "warning",
            max_height=520,
        )
        render_card("请注意", "这些要求不会自动发送给 AI。", "warning")

    render_task_submission(task)


def page_post_survey():
    st.header("实验后问卷")
    show_estimation_notice()

    with st.form("post_survey_form"):
        render_card("低能耗模式偏好", "请根据你对真实 AI 产品功能设计的接受程度作答。", "info")
        render_likert_instruction()
        q_low_energy_wait = likert(
            "如果 AI 的低能耗模式可以减少约 30% 的估算能耗，但回答时间从 3 秒增加到 8 秒，我愿意开启。",
            "q_low_energy_wait",
        )
        q_low_energy_quality = likert(
            "如果 AI 的低能耗模式可以减少约 30% 的估算能耗，但回答质量可能略低，我愿意在简单任务中开启。",
            "q_low_energy_quality",
        )
        q_default_low_energy = likert(
            "如果 AI 产品默认开启低能耗模式，但允许用户手动切换到高质量模式，我支持这种设计。",
            "q_default_low_energy",
        )

        render_card("能耗反馈体验", "请评价你对实验中提示信息的注意、理解和帮助性感受。", "energy")
        render_likert_instruction()
        q_feedback_attention = likert("我注意到了实验中的能耗反馈或能耗提示。", "q_feedback_attention")
        q_feedback_understandability = likert("我觉得实验中的能耗反馈容易理解。", "q_feedback_understandability")
        q_feedback_helpfulness = likert("我觉得能耗反馈有助于我更有意识地使用 AI。", "q_feedback_helpfulness")

        render_card("压力、内疚与反馈疲劳", "请评价这些提示是否影响了你的追问意愿、任务体验和长期接受度。", "warning")
        render_likert_instruction()
        q_feedback_intrusiveness = likert("我觉得能耗反馈有些打扰我完成任务。", "q_feedback_intrusiveness")
        q_feedback_pressure = likert("能耗反馈让我在使用 AI 时感到压力。", "q_feedback_pressure")
        q_feedback_guilt = likert("能耗反馈让我对继续使用 AI 产生了一些内疚感。", "q_feedback_guilt")
        q_interaction_penalty = likert("我有时因为能耗提示而减少了本来可能有助于提升答案质量的追问。", "q_interaction_penalty")
        q_feedback_fatigue = likert("如果真实 AI 产品长期显示类似能耗提示，我可能会逐渐忽略它。", "q_feedback_fatigue")
        q_long_term_acceptance = likert("如果真实 AI 产品提供类似能耗反馈，我愿意长期使用。", "q_long_term_acceptance")
        q_task_dependent_feedback = likert(
            "我认为能耗提示应该只在长对话、高能耗任务或复杂任务中显示，而不是每次都显示。",
            "q_task_dependent_feedback",
        )

        render_card(
            "开放题",
            (
                "其中带“必填”的 3 个问题需要填写，且每题至少 5 个字符。"
                "其他问题为可选，可根据你的实际想法补充。"
            ),
            "default",
        )
        open_awareness = st.text_area("实验前，你是否意识到 AI 使用可能涉及能源消耗？请简单说明。（可选）")
        open_behavior_change = st.text_area("在实验过程中，你是否改变了自己的提问方式？如果有，是如何改变的？（必填）")
        open_feedback_preference = st.text_area(
            "你认为哪种 AI 能耗提示方式最容易被用户接受？为什么？（必填）"
        )
        open_productive_iteration = st.text_area(
            "在实验中，你有没有因为能耗提示而放弃某些本来可能有助于改善答案的追问？如果有，请举例说明。（必填）"
        )
        open_feedback_fatigue = st.text_area(
            "如果真实 AI 产品长期显示类似能耗提示，你觉得自己会一直注意它，还是会逐渐忽略？为什么？（可选）"
        )
        open_design_suggestion = st.text_area(
            "你认为怎样的 AI 能耗提示设计最合适？例如每次显示、累计显示、任务结束后显示、只在高能耗任务中显示、或者提供低能耗模式开关。（可选）"
        )
        submitted = st.form_submit_button("提交并完成实验", type="primary")

    if submitted:
        is_valid, errors = validate_post_survey(
            open_awareness,
            open_behavior_change,
            open_feedback_preference,
            open_productive_iteration,
            open_feedback_fatigue,
            open_design_suggestion,
        )
        if not is_valid:
            st.warning("请完成必填开放题后再提交。未完成：" + "、".join(errors) + "。")
            return

        save_post_survey(
            {
                "q_low_energy_wait": q_low_energy_wait,
                "q_low_energy_quality": q_low_energy_quality,
                "q_default_low_energy": q_default_low_energy,
                "q_feedback_attention": q_feedback_attention,
                "q_feedback_understandability": q_feedback_understandability,
                "q_feedback_helpfulness": q_feedback_helpfulness,
                "q_feedback_intrusiveness": q_feedback_intrusiveness,
                "q_feedback_pressure": q_feedback_pressure,
                "q_feedback_guilt": q_feedback_guilt,
                "q_interaction_penalty": q_interaction_penalty,
                "q_feedback_fatigue": q_feedback_fatigue,
                "q_long_term_acceptance": q_long_term_acceptance,
                "q_task_dependent_feedback": q_task_dependent_feedback,
                "open_awareness": open_awareness.strip(),
                "open_behavior_change": open_behavior_change.strip(),
                "open_feedback_preference": open_feedback_preference.strip(),
                "open_productive_iteration": open_productive_iteration.strip(),
                "open_feedback_fatigue": open_feedback_fatigue.strip(),
                "open_design_suggestion": open_design_suggestion.strip(),
            }
        )
        log_event("post_survey_submitted")
        log_event("experiment_completed")
        st.session_state.page = "end"
        rerun()


def page_end():
    st.header("实验完成，感谢参与")
    st.success("你已经完成全部任务和问卷。")
    st.write(f"你的匿名 participant_id 是：`{st.session_state.participant_id}`")
    show_estimation_notice()


# ====================
# 管理员导出
# ====================

def admin_page():
    st.title("管理员数据导出")
    password = st.text_input("管理员密码", type="password")
    admin_password = clean_secret(get_secret("ADMIN_PASSWORD", "")) or "admin123"

    if not password:
        st.warning("请输入管理员密码。")
        return

    if password != admin_password:
        st.warning("管理员密码不正确。")
        return

    st.info("如果部署在 Streamlit Community Cloud，请在正式收集数据期间定期导出 CSV 备份。")
    tables = ["participants", "messages", "task_sessions", "questionnaires", "events"]
    selected_table = st.selectbox("选择数据表", tables)

    with get_db() as conn:
        df = pd.read_sql_query(f"SELECT * FROM {selected_table}", conn)

    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label=f"下载 {selected_table}.csv",
        data=csv,
        file_name=f"{selected_table}.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("派生编码与评分文件")
    prompt_coding_file = make_prompt_coding_file()
    st.download_button(
        "导出 Prompt 编码文件 prompt_coding_file.csv",
        data=prompt_coding_file.to_csv(index=False).encode("utf-8-sig"),
        file_name="prompt_coding_file.csv",
        mime="text/csv",
    )

    prompt_coding_template = make_prompt_coding_template()
    st.download_button(
        "导出 Prompt 编码模板 prompt_coding_template.csv",
        data=prompt_coding_template.to_csv(index=False).encode("utf-8-sig"),
        file_name="prompt_coding_template.csv",
        mime="text/csv",
    )

    requirement_checklist = make_requirement_checklist_df()
    st.download_button(
        "导出任务要求清单 requirement_checklist.csv",
        data=requirement_checklist.to_csv(index=False).encode("utf-8-sig"),
        file_name="requirement_checklist.csv",
        mime="text/csv",
    )

    blind_rating_file, blind_rating_key = make_blind_rating_exports()
    st.download_button(
        "导出 blind_rating_file.csv",
        data=blind_rating_file.to_csv(index=False).encode("utf-8-sig"),
        file_name="blind_rating_file.csv",
        mime="text/csv",
    )
    st.download_button(
        "导出 blind_rating_key.csv",
        data=blind_rating_key.to_csv(index=False).encode("utf-8-sig"),
        file_name="blind_rating_key.csv",
        mime="text/csv",
    )

    ai_scoring_input = make_ai_scoring_input()
    st.download_button(
        "导出 ai_scoring_input.csv",
        data=ai_scoring_input.to_csv(index=False).encode("utf-8-sig"),
        file_name="ai_scoring_input.csv",
        mime="text/csv",
    )


# ====================
# main
# ====================

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="AI", layout="wide")
    render_workspace_styles()
    init_db()

    if get_query_param("admin") == "1":
        admin_page()
        return

    ensure_api_key_for_experiment()
    init_session_state()
    debug_sidebar()

    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    page = st.session_state.page
    if page == "consent":
        page_consent()
    elif page == "pre_survey":
        page_pre_survey()
    elif page == "instruction":
        page_instruction()
    elif page == "task":
        page_task()
    elif page == "post_survey":
        page_post_survey()
    elif page == "end":
        page_end()
    else:
        st.session_state.page = "consent"
        rerun()


if __name__ == "__main__":
    main()
