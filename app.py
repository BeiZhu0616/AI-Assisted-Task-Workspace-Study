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
    "AI 已经可以看到本任务的基础背景材料。你可以根据自己的任务目标，自行决定如何向 AI 提问。"
)

SYSTEM_PROMPT = (
    "你是一个实验系统中的 AI 助手。你的任务是帮助用户完成当前任务。\n"
    "请根据用户提出的问题和当前任务材料，给出清晰、简洁、实用的回答。\n"
    "如果用户的问题信息不足，可以提醒用户补充必要信息。\n"
    "不要提及实验设计、研究目的、能源消耗、碳排放、环保提示或用户所在实验组。\n"
    "不要主动评价用户的提问方式。\n"
    "回答应尽量简洁。除非用户明确要求展开，否则每次回复控制在 300 个中文字符以内。"
    "对于写作任务，默认先给一版简洁可用的草稿，不要主动生成多个版本。"
    "对于代码任务，可以给出必要代码和简短解释，但不要写长篇教程。"
    "不要主动给出过多格式化解释、评分分析或长篇说明。优先直接回应用户当前请求。"
)

TASKS = [
    {
        "id": "writing",
        "title": "任务一：撰写延期邮件",
        "ai_visible_context": (
            "下面是学生整理的几条零散情况，准备给老师写一封延期沟通邮件：\n\n"
            "- 今晚有一门课的作业截止；\n"
            "- 学生明天上午需要参加学院创新项目展示；\n"
            "- 学生负责展示 PPT 的最后整合；\n"
            "- 作业已经做了一部分，但还没有完全收尾；\n"
            "- 想和老师沟通能否晚一点提交；\n"
            "- 老师平时比较重视提前沟通。"
        ),
        "participant_only_requirements": (
            "请借助 AI，完成一封适合发给老师的延期申请邮件。\n\n"
            "这封邮件应当自然、礼貌、可信，并适合真实发送。"
            "你可以自行决定如何向 AI 提问、是否让 AI 修改，以及最终采用哪一版内容。\n\n"
            "最终提交：一封你认为最适合发给老师的邮件。"
        ),
        "final_answer_label": "请提交你认为最适合发给老师的一版邮件。",
    },
    {
        "id": "summary",
        "title": "任务二：总结高校 AI 使用材料",
        "ai_visible_context": (
            "以下是一段关于生成式 AI 进入高校学习场景的材料：\n\n"
            "近年来，生成式人工智能逐渐进入高校学习场景。学生可以使用 AI 进行资料整理、文章润色、代码调试和观点生成。"
            "一方面，AI 工具提高了学习效率，帮助学生更快理解复杂内容；另一方面，过度依赖 AI 也可能削弱学生的独立思考能力，"
            "并带来学术诚信问题。部分教师主张完全禁止 AI 参与作业，认为这会破坏公平性；也有教师认为，与其禁止，"
            "不如引导学生合理使用 AI，并要求他们披露使用方式。高校未来需要在效率、诚信、学习能力培养之间取得平衡。"
        ),
        "participant_only_requirements": (
            "请借助 AI，基于左侧材料整理一份简短说明。\n\n"
            "这份说明需要让读者快速理解：生成式 AI 进入高校学习场景后，可能带来哪些好处、风险，"
            "以及你认为学校应该如何应对。\n\n"
            "请你自行决定如何组织内容，使其清楚、通俗、有用。\n\n"
            "最终提交：一份你认为清楚、有用的材料整理。"
        ),
        "final_answer_label": "请提交你认为清楚、有用的一份材料整理。",
    },
    {
        "id": "code",
        "title": "任务三：解释并修正 Python 总收入代码",
        "ai_visible_context": (
            "下面这段 Python 代码本来想计算订单总收入，但结果不正确。\n\n"
            "```python\n"
            "orders = [\n"
            "    {\"item\": \"book\", \"price\": 80, \"quantity\": 2},\n"
            "    {\"item\": \"pen\", \"price\": 5, \"quantity\": 10},\n"
            "    {\"item\": \"bag\", \"price\": 120, \"quantity\": 1}\n"
            "]\n\n"
            "total = 0\n"
            "for order in orders:\n"
            "    total = order[\"price\"]\n\n"
            "print(\"Total revenue:\", total)\n"
            "```"
        ),
        "participant_only_requirements": (
            "请借助 AI，检查这段 Python 代码为什么没有正确计算订单总收入，并整理一份适合初学者理解的修正说明。\n\n"
            "请你自行决定如何向 AI 提问，并最终提交你认为最清楚、最有帮助的解释和修正代码。\n\n"
            "最终提交：问题说明、修正代码和简短解释。"
        ),
        "final_answer_label": "请提交问题说明、修正代码和简短解释。",
    },
]

TASK_REQUIREMENTS = {
    "writing": [
        ("task_completion", "是否清楚表达了延期申请"),
        ("reasonableness", "延期理由是否合理、可信、不过度夸张"),
        ("actionability", "是否让老师清楚知道学生希望如何处理"),
        ("tone_appropriateness", "语气是否礼貌、自然、不过度卑微"),
        ("clarity", "表达是否清楚、简洁、容易理解"),
        ("real_world_usability", "是否适合真实发送给老师"),
        ("overall_quality", "整体质量"),
        ("diagnostic_mostly_done", "诊断项：是否提到已完成大部分作业"),
        ("diagnostic_new_arrangement", "诊断项：是否提出明确新提交安排"),
        ("diagnostic_unnecessary_personal_reason", "诊断项：是否出现不必要的家庭/身体原因"),
        ("diagnostic_ai_template_like", "诊断项：是否明显 AI 模板化"),
    ],
    "summary": [
        ("relevance", "是否围绕左侧材料展开，没有跑题"),
        ("coverage", "是否覆盖主要好处和主要风险"),
        ("faithfulness_consistency", "是否忠实于原材料，没有编造重要事实"),
        ("coherence", "结构是否清楚，逻辑是否连贯"),
        ("usefulness", "是否对读者理解高校 AI 使用议题有帮助"),
        ("practical_judgment", "是否包含合理判断或建议"),
        ("fluency", "语言是否通顺、自然、通俗"),
        ("overall_quality", "整体质量"),
        ("diagnostic_learning_efficiency", "诊断项：是否提到学习效率"),
        ("diagnostic_learning_ability_risk", "诊断项：是否提到学习能力风险"),
        ("diagnostic_academic_integrity", "诊断项：是否提到学术诚信"),
        ("diagnostic_fairness_or_management", "诊断项：是否提到公平性或教学管理"),
        ("diagnostic_suggestion_or_judgment", "诊断项：是否包含建议或判断"),
    ],
    "code": [
        ("bug_identification", "是否准确指出原代码的问题"),
        ("correctness", "修正代码是否能正确计算总收入"),
        ("conceptual_explanation", "是否解释了为什么原代码结果不对"),
        ("beginner_suitability", "解释是否适合初学者理解"),
        ("conciseness", "是否简洁，不写成长篇无关教程"),
        ("overall_usefulness", "作为代码学习说明是否有帮助"),
        ("overall_quality", "整体质量"),
        ("diagnostic_total_overwritten", "诊断项：是否指出 total 被覆盖"),
        ("diagnostic_price_times_quantity", "诊断项：是否指出需要计算 price × quantity"),
        ("diagnostic_accumulation_logic", "诊断项：是否使用累加逻辑"),
        ("diagnostic_final_result_reason", "诊断项：是否解释最后结果为什么不对"),
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
        "background-color: #FFF7ED; border: 1px solid #FDBA74; border-radius: 12px; "
        "padding: 12px 14px; margin: 0 0 12px 0; line-height: 1.5"
    )
    st.markdown(
        (
            f'<div style="{card_styles};">'
            '<div style="font-weight: 700; color: #C2410C; margin-bottom: 4px;">累计估算反馈</div>'
            '<div style="color: #111827;">'
            f"本任务累计：<strong>{cumulative_wh:.4f} Wh</strong><br>"
            f"💡 LED 约 <strong>{cumulative_led:.1f} 分钟</strong>"
            "</div></div>"
        ),
        unsafe_allow_html=True,
    )


def render_compact_energy_feedback(energy_wh: float, led_minutes: float, cumulative_wh: float, cumulative_led: float):
    card_styles = (
        "background-color: #FFF7ED; border: 1px solid #FDBA74; border-radius: 10px; "
        "padding: 8px 12px; margin: 6px 0 12px 0; color: #111827; "
        "font-size: 0.9rem; line-height: 1.45"
    )
    st.markdown(
        (
            f'<div style="{card_styles};">'
            f"⚡ 本次估算：<strong>{energy_wh:.4f} Wh</strong>｜💡 LED 约 <strong>{led_minutes:.1f} 分钟</strong>｜"
            f"累计：<strong>{cumulative_wh:.4f} Wh</strong>｜💡 LED 约 <strong>{cumulative_led:.1f} 分钟</strong>"
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
            ".block-container{max-width:1320px;padding-top:32px;padding-bottom:3rem;margin-left:auto;margin-right:auto;}"
            "h1{font-size:2.05rem!important;line-height:1.2!important;}"
            "h2,h3{line-height:1.25!important;}"
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
            "请基于这些基础材料回答用户的问题，不要假设未出现在对话中的额外要求。\n\n"
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


def get_successful_prompt_count(task_id: str) -> int:
    history = st.session_state.get("histories", {}).get(task_id, [])
    session_count = sum(1 for message in history if message.get("role") == "user")
    if session_count:
        return session_count

    participant_id = st.session_state.get("participant_id")
    if not participant_id:
        return 0
    try:
        with get_db() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS prompt_count
                FROM messages
                WHERE participant_id = ? AND task_id = ?
                """,
                (participant_id, task_id),
            ).fetchone()
            return int(row["prompt_count"] if row else 0)
    except Exception:
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
    open_behavior_change,
    open_feedback_effect=None,
    open_dynamic_iteration_effect=None,
    condition="control",
) -> tuple[bool, list[str]]:
    errors = []
    if len((open_behavior_change or "").strip()) < 5:
        errors.append("提问方式变化说明")
    if condition in {"static_feedback", "dynamic_feedback"} and len((open_feedback_effect or "").strip()) < 5:
        errors.append("提示或反馈影响说明")
    if condition == "dynamic_feedback" and len((open_dynamic_iteration_effect or "").strip()) < 5:
        errors.append("继续追问或修改意愿说明")
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
                open_design_suggestion TEXT,
                q_hypothetical_feedback_helpful INTEGER,
                q_hypothetical_low_energy_mode INTEGER,
                q_hypothetical_wait_acceptance INTEGER,
                q_hypothetical_quality_tradeoff INTEGER,
                q_static_feedback_attention INTEGER,
                q_static_feedback_understandability INTEGER,
                q_static_feedback_behavior_change INTEGER,
                q_static_feedback_pressure INTEGER,
                q_dynamic_feedback_attention INTEGER,
                q_dynamic_feedback_understandability INTEGER,
                q_dynamic_feedback_behavior_change INTEGER,
                q_dynamic_cumulative_salience INTEGER,
                q_dynamic_feedback_pressure INTEGER,
                q_dynamic_feedback_fatigue INTEGER,
                open_feedback_effect TEXT,
                open_dynamic_iteration_effect TEXT
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
                "q_hypothetical_feedback_helpful": "INTEGER",
                "q_hypothetical_low_energy_mode": "INTEGER",
                "q_hypothetical_wait_acceptance": "INTEGER",
                "q_hypothetical_quality_tradeoff": "INTEGER",
                "q_static_feedback_attention": "INTEGER",
                "q_static_feedback_understandability": "INTEGER",
                "q_static_feedback_behavior_change": "INTEGER",
                "q_static_feedback_pressure": "INTEGER",
                "q_dynamic_feedback_attention": "INTEGER",
                "q_dynamic_feedback_understandability": "INTEGER",
                "q_dynamic_feedback_behavior_change": "INTEGER",
                "q_dynamic_cumulative_salience": "INTEGER",
                "q_dynamic_feedback_pressure": "INTEGER",
                "q_dynamic_feedback_fatigue": "INTEGER",
                "open_feedback_effect": "TEXT",
                "open_dynamic_iteration_effect": "TEXT",
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
    columns = [
        "participant_id",
        "condition",
        "submitted_at",
        "q_low_energy_wait",
        "q_low_energy_quality",
        "q_default_low_energy",
        "q_feedback_attention",
        "q_feedback_understandability",
        "q_feedback_helpfulness",
        "q_feedback_intrusiveness",
        "q_feedback_pressure",
        "q_feedback_guilt",
        "q_interaction_penalty",
        "q_feedback_fatigue",
        "q_long_term_acceptance",
        "q_task_dependent_feedback",
        "open_awareness",
        "open_behavior_change",
        "open_feedback_preference",
        "open_productive_iteration",
        "open_feedback_fatigue",
        "open_design_suggestion",
        "q_hypothetical_feedback_helpful",
        "q_hypothetical_low_energy_mode",
        "q_hypothetical_wait_acceptance",
        "q_hypothetical_quality_tradeoff",
        "q_static_feedback_attention",
        "q_static_feedback_understandability",
        "q_static_feedback_behavior_change",
        "q_static_feedback_pressure",
        "q_dynamic_feedback_attention",
        "q_dynamic_feedback_understandability",
        "q_dynamic_feedback_behavior_change",
        "q_dynamic_cumulative_salience",
        "q_dynamic_feedback_pressure",
        "q_dynamic_feedback_fatigue",
        "open_feedback_effect",
        "open_dynamic_iteration_effect",
    ]
    values = {
        "participant_id": st.session_state.participant_id,
        "condition": st.session_state.condition,
        "submitted_at": now_iso(),
    }
    for column in columns:
        if column not in values:
            values[column] = data.get(column)

    with get_db() as conn:
        existing = conn.execute(
            """
            SELECT id FROM questionnaires
            WHERE participant_id = ?
            ORDER BY id DESC LIMIT 1
            """,
            (st.session_state.participant_id,),
        ).fetchone()
        questionnaire_values = tuple(values[column] for column in columns)
        if existing:
            assignments = ", ".join([f"{column} = ?" for column in columns])
            conn.execute(
                f"UPDATE questionnaires SET {assignments} WHERE id = ?",
                questionnaire_values + (existing["id"],),
            )
        else:
            placeholders = ", ".join(["?"] * len(columns))
            column_sql = ", ".join(columns)
            conn.execute(
                f"INSERT INTO questionnaires ({column_sql}) VALUES ({placeholders})",
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
    template_columns = [
        "blind_prompt_id",
        "task_id",
        "task_index",
        "turn_id",
        "user_prompt",
        "previous_ai_response",
        "next_ai_response",
        "task_copy_similarity",
        "is_task_copying",
        "prompt_type",
        "prompt_specificity_score",
        "strategic_prompting_score",
        "productive_iteration",
        "low_value_prompt",
        "mechanical_copying",
        "coder_id",
        "coder_notes",
    ]
    if source.empty:
        return pd.DataFrame(columns=template_columns)
    template = source[
        [
            "blind_prompt_id",
            "task_id",
            "task_index",
            "turn_id",
            "user_prompt",
            "previous_ai_response",
            "next_ai_response",
            "task_copy_similarity",
            "is_task_copying",
        ]
    ].copy()
    for column in [
        "prompt_type",
        "prompt_specificity_score",
        "strategic_prompting_score",
        "productive_iteration",
        "low_value_prompt",
        "mechanical_copying",
        "coder_id",
        "coder_notes",
    ]:
        template[column] = ""
    return template[template_columns]


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
    rating_file["task_completion"] = ""
    rating_file["relevance"] = ""
    rating_file["clarity"] = ""
    rating_file["coherence"] = ""
    rating_file["usefulness"] = ""
    rating_file["correctness_for_code"] = ""
    rating_file["real_world_usability"] = ""
    rating_file["overall_quality"] = ""
    rating_file["diagnostic_notes"] = ""
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
    st.title("知情同意")
    render_card(
        "研究介绍与任务说明",
        (
            "我们是来自西交利物浦大学产业家学院的研究团队，正在开展一项关于 AI 辅助任务完成方式的研究。\n\n"
            "在本研究中，您需要依次完成三个简短任务：邮件撰写、摘要撰写和代码修改。\n\n"
            f"每个任务中，您可以向页面中的 AI 助手提问，最多 {MAX_TURNS_PER_TASK} 轮。"
            "每个任务结束时，您需要提交一份最终答案。\n\n"
            "数据研究说明与隐私安全：\n"
            "本研究会记录您的问卷回答、任务操作过程、输入给 AI 的问题、AI 回复内容和最终提交答案。"
            "所有数据将以匿名编号保存，仅用于学术研究分析。"
            "您的输入内容会发送给第三方大模型 API 用于生成回复，请不要输入真实姓名、手机号、身份证号、住址、"
            "账号密码、商业机密或其他敏感信息。"
        ),
        "info",
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
    if st.button("下一页", type="primary"):
        if not agreed:
            st.warning("请先勾选知情同意后再进入下一步。")
            return
        log_event("consent_accepted")
        st.session_state.page = "pre_survey"
        rerun()


def page_pre_survey():
    st.markdown(
        (
            '<div style="padding-top: 18px; margin-bottom: 18px; '
            'font-size: 1rem; line-height: 1.8; color: #111827;">'
            "请根据您的真实情况填写以下信息。"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    with st.form("pre_survey_form"):
        st.subheader("基本信息")
        birth_col1, birth_col2 = st.columns(2)
        with birth_col1:
            birth_year = st.selectbox("出生年份", birth_year_options())
        with birth_col2:
            birth_month = st.selectbox("出生月份", birth_month_options(), format_func=lambda value: f"{value} 月" if isinstance(value, int) else value)
        gender = st.selectbox("性别", ["请选择", "男", "女"])
        education = st.selectbox("教育程度", ["请选择", "高中及以下", "大专", "本科", "硕士", "博士及以上", "其他"])
        ai_usage_frequency = st.selectbox(
            "AI 使用频率",
            ["请选择", "几乎不用", "每月几次", "每周几次", "每天", "每天多次"],
        )

        st.subheader("基本看法")
        st.caption("以下题目没有正确或错误答案，请根据真实想法选择。1 = 非常不同意，7 = 非常同意。")
        q_ai_use_1 = likert("我经常使用生成式 AI 工具完成学习、工作或生活任务。", "q_ai_use_1", value=None)
        q_ai_use_2 = likert("我认为自己比较擅长向 AI 提出清晰的问题。", "q_ai_use_2", value=None)

        q_energy_awareness_1 = likert("我知道生成式 AI 每次回答都需要消耗计算资源。", "q_energy_awareness_1", value=None)
        q_energy_awareness_2 = likert("我知道 AI 数据中心可能消耗大量电力。", "q_energy_awareness_2", value=None)
        q_energy_awareness_3 = likert("在使用 AI 时，我通常会考虑它背后的能源成本。", "q_energy_awareness_3", value=None)

        q_environment_1 = likert("我愿意在日常生活中采取行动减少能源浪费。", "q_environment_1", value=None)
        q_environment_2 = likert("我认为科技产品应该向用户披露环境影响。", "q_environment_2", value=None)

        st.subheader("注意力检查")
        attention_check = likert("为了确认你认真阅读，请在本题选择 6。", "attention_check", value=None)

        nav_left, back_col, next_col, nav_right = st.columns([1, 0.35, 0.35, 1])
        with back_col:
            go_back = st.form_submit_button("上一页")
        with next_col:
            submitted = st.form_submit_button("下一页", type="primary")

    if go_back:
        st.session_state.page = "consent"
        rerun()

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
    st.header("任务操作说明")
    render_bullet_card(
        "请先阅读这三条规则",
        [
            f"您将依次完成三个任务，每个任务最多可以向 AI 提问 {MAX_TURNS_PER_TASK} 轮。",
            "每个任务中，AI 已经可以看到页面提供的基础材料。您可以根据自己的判断向 AI 提问。",
            "AI 的回复不会自动提交。每个任务结束时，请在最终答案框中提交您决定采用的答案。",
        ],
        "info",
    )
    nav_left, back_col, next_col, nav_right = st.columns([1, 0.35, 0.35, 1])
    with back_col:
        if st.button("上一页"):
            st.session_state.page = "pre_survey"
            rerun()
    with next_col:
        next_clicked = st.button("下一页", type="primary")
    if next_clicked:
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

    turns_used = get_successful_prompt_count(task_id)
    st.session_state.turn_counts[task_id] = turns_used
    turns_left = MAX_TURNS_PER_TASK - turns_used

    if turns_left <= 0:
        st.warning("本任务的 AI 提问次数已用完，请在下方提交最终答案。")
        return

    prompt = st.chat_input("输入你想问 AI 的问题")
    if not prompt:
        return

    prompt_timestamp = now_iso()
    next_turn = turns_used + 1
    context_similarity = calculate_task_copy_similarity(prompt, task["ai_visible_context"])
    right_goal_similarity = calculate_task_copy_similarity(prompt, task["participant_only_requirements"])
    similarity = max(context_similarity, right_goal_similarity)
    is_task_copying = 1 if similarity >= TASK_COPY_THRESHOLD else 0
    is_context_copying = 1 if context_similarity >= TASK_COPY_THRESHOLD else 0
    is_requirements_copying = 1 if right_goal_similarity >= TASK_COPY_THRESHOLD else 0

    log_event(
        "prompt_submitted",
        event_value=(
            f"turn_id={next_turn};copy_similarity={similarity:.2f};"
            f"context_similarity={context_similarity:.2f};right_goal_similarity={right_goal_similarity:.2f}"
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
            "requirements_copy_similarity": right_goal_similarity,
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
    rerun()


def render_task_submission(task: dict):
    task_id = task["id"]
    st.divider()
    render_card(
        "最终提交答案",
        "请根据 AI 的帮助，提交您最终决定采用的答案。AI 的回复不会自动提交。",
        "default",
    )

    with st.form(f"final_form_{task_id}"):
        final_answer = st.text_area(task["final_answer_label"], height=180)
        st.subheader("本任务体验评价")
        render_likert_instruction()
        satisfaction = likert("我对本任务最终结果满意。", f"satisfaction_{task_id}")
        perceived_efficiency = likert("AI 帮助我高效完成了本任务。", f"perceived_efficiency_{task_id}")
        cognitive_load = likert("完成这个任务让我感到费力。", f"cognitive_load_{task_id}")
        submitted = st.form_submit_button("提交本任务", type="primary")

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
    turns_used = get_successful_prompt_count(task_id)
    st.session_state.turn_counts[task_id] = turns_used
    turns_left = MAX_TURNS_PER_TASK - turns_used

    st.header(task["title"])
    st.caption(f"第 {task_number} 个任务，共 {len(TASKS)} 个｜剩余 AI 提问次数：{turns_left}")

    render_card("你的任务", task["participant_only_requirements"], "warning")

    st.subheader("AI 已知材料")
    st.caption("以下材料已经提供给 AI，您不需要重复复制。")
    material_expanded = task["id"] != "summary"
    with st.expander("查看 AI 已知材料", expanded=material_expanded):
        render_ai_visible_context(task)

    chat_col, status_col = st.columns([1.65, 0.75])
    energy_summary_slot = None

    with status_col:
        st.subheader("任务状态")
        render_card("剩余提问次数", f"{turns_left}", "default")
        if st.session_state.condition == "static_feedback":
            render_card(
                "提示",
                "AI 请求会消耗一定计算资源和电力。本实验中的相关数值均为估算。",
                "energy",
            )
        elif st.session_state.condition == "dynamic_feedback":
            energy_summary_slot = st.empty()
            with energy_summary_slot:
                render_energy_summary_card(
                    st.session_state.cumulative_energy[task_id],
                    st.session_state.cumulative_led[task_id],
                )

    with chat_col:
        st.subheader("AI 对话区")
        render_card(
            "对话提示",
            "您可以告诉 AI 想完成什么，也可以根据需要补充格式、语气或限制条件。",
            "default",
        )
        render_chat(task, energy_summary_slot)

    render_task_submission(task)


def page_post_survey():
    st.header("最后几个问题")
    condition = st.session_state.condition

    with st.form("post_survey_form"):
        render_card("未来产品假设题", "以下问题是假设未来 AI 产品可能提供相关功能，请根据真实想法选择。", "info")
        render_likert_instruction()
        q_hypothetical_feedback_helpful = likert(
            "如果未来 AI 产品显示 AI 请求的估算能耗或资源使用情况，我认为这会有帮助。",
            "q_hypothetical_feedback_helpful",
        )
        q_hypothetical_low_energy_mode = likert(
            "如果未来 AI 产品提供低能耗模式，我愿意在简单任务中尝试使用。",
            "q_hypothetical_low_energy_mode",
        )
        q_hypothetical_wait_acceptance = likert(
            "如果低能耗模式会让回答稍慢一些，我在简单任务中可以接受。",
            "q_hypothetical_wait_acceptance",
        )
        q_hypothetical_quality_tradeoff = likert(
            "如果低能耗模式可能让回答质量略低，我只会在不重要或简单任务中使用。",
            "q_hypothetical_quality_tradeoff",
        )

        q_static_feedback_attention = None
        q_static_feedback_understandability = None
        q_static_feedback_behavior_change = None
        q_static_feedback_pressure = None
        q_dynamic_feedback_attention = None
        q_dynamic_feedback_understandability = None
        q_dynamic_feedback_behavior_change = None
        q_dynamic_cumulative_salience = None
        q_dynamic_feedback_pressure = None
        q_dynamic_feedback_fatigue = None

        if condition == "static_feedback":
            render_card("任务提示体验", "请根据您在任务中看到的提示作答。", "energy")
            render_likert_instruction()
            q_static_feedback_attention = likert("我注意到了任务中的能耗提示。", "q_static_feedback_attention")
            q_static_feedback_understandability = likert("我觉得任务中的能耗提示容易理解。", "q_static_feedback_understandability")
            q_static_feedback_behavior_change = likert("任务中的能耗提示影响了我使用 AI 的方式。", "q_static_feedback_behavior_change")
            q_static_feedback_pressure = likert("任务中的能耗提示让我感到有压力。", "q_static_feedback_pressure")
        elif condition == "dynamic_feedback":
            render_card("任务反馈体验", "请根据您在任务中看到的反馈作答。", "energy")
            render_likert_instruction()
            q_dynamic_feedback_attention = likert("我注意到了每次 AI 回复后的能耗反馈。", "q_dynamic_feedback_attention")
            q_dynamic_feedback_understandability = likert("我觉得每次 AI 回复后的能耗反馈容易理解。", "q_dynamic_feedback_understandability")
            q_dynamic_feedback_behavior_change = likert("每次 AI 回复后的能耗反馈影响了我的提问方式。", "q_dynamic_feedback_behavior_change")
            q_dynamic_cumulative_salience = likert("累计能耗反馈让我更注意自己向 AI 提问的次数。", "q_dynamic_cumulative_salience")
            q_dynamic_feedback_pressure = likert("能耗反馈让我在使用 AI 时感到压力。", "q_dynamic_feedback_pressure")
            q_dynamic_feedback_fatigue = likert("如果真实 AI 产品长期显示类似反馈，我可能会逐渐忽略它。", "q_dynamic_feedback_fatigue")

        render_card(
            "开放题",
            "请用简短文字说明您的真实想法。带“必填”的问题需要填写。",
            "default",
        )
        open_behavior_change = st.text_area("在实验过程中，您是否改变了自己向 AI 提问的方式？如果有，是如何改变的？（必填）")
        open_feedback_effect = None
        open_dynamic_iteration_effect = None
        if condition in {"static_feedback", "dynamic_feedback"}:
            open_feedback_effect = st.text_area("您觉得本实验中的能耗提示或反馈对您有什么影响？（必填）")
        if condition == "dynamic_feedback":
            open_dynamic_iteration_effect = st.text_area("每次 AI 回复后的能耗反馈是否影响了您继续追问或修改答案的意愿？请简单说明。（必填）")

        submitted = st.form_submit_button("提交并完成", type="primary")

    if submitted:
        is_valid, errors = validate_post_survey(
            open_behavior_change,
            open_feedback_effect,
            open_dynamic_iteration_effect,
            condition,
        )
        if not is_valid:
            st.warning("请完成必填开放题后再提交。未完成：" + "、".join(errors) + "。")
            return

        save_post_survey(
            {
                "q_hypothetical_feedback_helpful": q_hypothetical_feedback_helpful,
                "q_hypothetical_low_energy_mode": q_hypothetical_low_energy_mode,
                "q_hypothetical_wait_acceptance": q_hypothetical_wait_acceptance,
                "q_hypothetical_quality_tradeoff": q_hypothetical_quality_tradeoff,
                "q_static_feedback_attention": q_static_feedback_attention,
                "q_static_feedback_understandability": q_static_feedback_understandability,
                "q_static_feedback_behavior_change": q_static_feedback_behavior_change,
                "q_static_feedback_pressure": q_static_feedback_pressure,
                "q_dynamic_feedback_attention": q_dynamic_feedback_attention,
                "q_dynamic_feedback_understandability": q_dynamic_feedback_understandability,
                "q_dynamic_feedback_behavior_change": q_dynamic_feedback_behavior_change,
                "q_dynamic_cumulative_salience": q_dynamic_cumulative_salience,
                "q_dynamic_feedback_pressure": q_dynamic_feedback_pressure,
                "q_dynamic_feedback_fatigue": q_dynamic_feedback_fatigue,
                "open_behavior_change": open_behavior_change.strip(),
                "open_feedback_effect": (open_feedback_effect or "").strip() or None,
                "open_dynamic_iteration_effect": (open_dynamic_iteration_effect or "").strip() or None,
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
    st.info("如界面中出现过任何估算数值，这些数值仅用于研究目的，不代表真实平台测量结果。")


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
