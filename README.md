# AI 协作任务研究系统

这是一个使用 Python、Streamlit、DeepSeek API、SQLite 和 pandas 实现的在线实验系统，用于 Paper 1：Energy Feedback as Reflective Friction。系统可本地运行，也可部署到 Streamlit Community Cloud，通过公开链接分享给被试。

本实验中的能耗数值为研究目的下的估算反馈，不代表 DeepSeek 或任何具体 AI 平台的真实能耗。

## 实验设计

实验采用 between-subjects design，参与者随机分配到三组之一：

- `control`：普通 AI 对话界面，不显示能耗信息。
- `static_feedback`：任务开始前显示一次静态能耗提示。
- `dynamic_feedback`：每次 AI 回复后显示本次估算能耗和当前任务累计估算能耗。

三组唯一不同是界面能耗反馈。三组调用同一个 DeepSeek 模型、同一个 system prompt、同样的 temperature 和 max_tokens。

## 界面设计目标

本实验界面被设计为“AI 协作任务工作台”，而不是普通聊天页面。被试在每个任务中同时看到：

- 左栏：AI 已知信息，即已经提供给 AI 的基础背景材料；
- 中栏：AI 对话区，用于向 AI 说明目标、输出格式、关键约束和排除项；
- 右栏：用户独有提交要求，即 AI 默认看不到的最终提交要求、评分标准和个人偏好。

这种信息架构服务于 Paper 1 的核心定位：Energy Feedback as Reflective Friction。动态反馈组的每轮能耗反馈用于形成轻量的反思性摩擦，但它不是强制限制，也不是惩罚。三组除能耗反馈外尽量保持一致，避免把任务材料呈现方式、prompt 指导或最终答案提交区变成混淆变量。

首页和全局标题使用“AI 协作任务研究”，而不是“能源感知实验”，以减少被试在进入任务前过早推测研究目的带来的 demand effect。control 组在任务说明页和任务工作台中不显示能耗、电力、LED、环保、低能耗等干预信息。

页面主容器限制在约 1200-1320px 宽度内，并保持左对齐阅读结构。任务标题采用具体任务名称：`任务一：撰写延期邮件`、`任务二：总结高校 AI 使用材料`、`任务三：解释并修正 Python 平均分代码`。

界面卡片由 [app.py](app.py) 中统一的 `render_card` / `render_bullet_card` 渲染。卡片正文会先进行 HTML escape，再由 `st.markdown(..., unsafe_allow_html=True)` 输出，以避免 `<`、`>`、换行或代码片段破坏 HTML 结构。任务三的 Python 代码单独使用 `st.code(..., language="python")` 显示，不放入 HTML 卡片正文。

## 能耗反馈机制与实验控制

三组的任务材料、双层任务结构、AI 模型、system prompt、temperature、max_tokens、最终答案提交区和问卷流程保持一致。唯一的界面操纵是能耗反馈：

- `control`：任务说明页和任务工作台不显示能耗相关信息；
- `static_feedback`：在实验说明页和每个任务页显示静态能耗提示，但不显示每轮数值；
- `dynamic_feedback`：每次 AI 成功回复后，在对话区下方显示本次估算能耗、当前任务累计估算能耗和 LED 灯点亮时间类比。

dynamic_feedback 的反馈卡片使用浅色中性视觉风格，目的是提升隐藏资源成本的显著性，形成 reflective friction，而不是制造惩罚感或阻止用户继续追问。

任务页中，`dynamic_feedback` 组还会在任务状态区下方显示“本任务累计能耗”小卡片，并在每条 AI 回复下方显示 compact 本次反馈。`static_feedback` 组只显示静态提示，不显示本次或累计数值。

默认模型配置：

```python
MODEL_NAME = "deepseek-v4-flash"
TEMPERATURE = 0.3
MAX_TOKENS = 700
```

如果 `deepseek-v4-flash` 调用失败，可以在 [app.py](app.py) 中将 `MODEL_NAME` 改为 DeepSeek 当前可用的 chat model。当前官方接口报错信息显示支持 `deepseek-v4-pro` 或 `deepseek-v4-flash`；如果误填 `deepseek-v3`，本系统会自动改用默认的 `deepseek-v4-flash`。

## 研究理论逻辑

本研究不是简单研究“能耗提示能否让用户少用 AI”，而是研究当 AI 使用背后的隐藏资源成本被界面显示出来时，用户是否会从随手问、反复试、机械复制，转向更有策略、更高质量、更负责任的提问方式。

理论链条：

```text
Invisible AI Resource Cost
→ Energy Feedback Salience
→ Reflective Friction
→ Prompting Strategy Adjustment
→ Task Quality / Resource Trade-off
→ Design Principles for Energy-Aware GenAI Interfaces
```

中文解释：

```text
AI 资源成本不可见
→ 能源反馈提升显著性
→ 反思性摩擦
→ 提问策略调整
→ 任务质量与资源消耗权衡
→ 能源感知型生成式 AI 界面设计原则
```

## 双层任务结构

本实验采用“双层任务结构”，每个任务被拆成两部分：

- `ai_visible_context`：AI 可见的基础背景材料，会自动加入 DeepSeek 请求历史。
- `participant_only_requirements`：只有被试看见的最终提交要求、评分标准、输出格式、个人偏好和排除项，不会自动传给 DeepSeek。

任务页采用三栏布局明确呈现双层任务结构：

- `AI 已知信息`：展示 `ai_visible_context`，并提示这些背景材料已经提供给 AI；
- `AI 对话区`：展示通用 prompt guidance，提醒用户说明目标、输出格式、关键约束和排除项；
- `你的提交要求`：展示 `participant_only_requirements`，并强调这些要求不会自动发送给 AI。

`participant_only_requirements` 不会通过 system prompt、hidden prompt、task initialization 或任何其他后台消息泄露给 DeepSeek。除非用户自己在 prompt 中告诉 AI，否则 AI 不知道这些最终提交要求。

这样设计是为了避免用户只输入“请基于任务完成”就获得完整答案，并用于区分 generic prompting、mechanical copying 和 strategic prompting。

## 复制任务检测

系统使用 Python 标准库 `difflib.SequenceMatcher` 计算用户 prompt 与任务两层材料的相似度：

```python
context_copy_similarity = SequenceMatcher(None, user_prompt.strip(), ai_visible_context.strip()).ratio()
requirements_copy_similarity = SequenceMatcher(None, user_prompt.strip(), participant_only_requirements.strip()).ratio()
task_copy_similarity = max(context_copy_similarity, requirements_copy_similarity)
```

如果 `task_copy_similarity >= 0.60`，则：

- `task_copy_similarity` 保存相似度
- `is_task_copying = 1`
- 在 `events` 表记录 `task_copy_detected`

系统也会在 `messages` 表中保存 `context_copy_similarity`、`requirements_copy_similarity`、`is_context_copying` 和 `is_requirements_copying`，用于区分复制基础材料和复制最终要求。

该变量用于区分 mechanical copying 和 strategic prompting。

后续人工编码建议：

- mechanical copying：用户直接复制任务全文，或不加筛选地把任务材料塞给 AI。
- strategic prompting：用户主动提炼任务目标、输出格式、质量标准、约束条件和排除项。

建议增加人工编码变量 `strategic_prompting_score`，1-5 分：

- 1 = 复制任务或笼统要求，无主动组织
- 2 = 提出基本目标，但缺少约束
- 3 = 提出目标和部分约束
- 4 = 能筛选关键信息，并明确格式/语气
- 5 = 能综合目标、约束、排除项和质量标准，形成高效 prompt

代码任务的正确答案评分 rubric 可以包含：正确指出 `total = scores[i]` 会覆盖之前的值，而不是累加。该评分点只用于后期评分或 README 说明，不显示给被试，也不会自动传给 DeepSeek。

## Prompt 行为分类

后期编码建议区分以下 prompt 类型：

- `generic_prompt`：笼统请求，例如“请基于任务写一封邮件”。
- `mechanical_copying`：机械复制任务材料或最终要求。
- `requirement_transmission`：用户主动把仅自己可见的最终要求传达给 AI。
- `productive_iteration`：有价值迭代，例如指出 AI 遗漏要求、要求修正具体问题、补充明确约束。
- `low_value_repetition`：低价值重复，例如无具体方向的“再好一点”“继续”“优化一下”。
- `off_task`：离题或无效请求。

本研究关心的不是简单减少所有多轮对话，而是判断能耗反馈是否减少低价值交互，同时不抑制有价值的人机协作。

## 文件结构

```text
energy_ai_experiment/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   ├── secrets.toml.example
│   └── config.toml
└── data/
    └── experiment.db  # 运行时自动创建，不要提交到 GitHub
```

## 安装依赖

建议使用 Python 3.10+。

```bash
cd energy_ai_experiment
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux：

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果本地出现 `typing_extensions` / `Sentinel` 相关错误，请确认正在使用项目独立虚拟环境，并执行：

```bash
python -m pip install --upgrade typing_extensions pydantic pydantic-core openai
python -m pip install -r requirements.txt
```

## 配置 DeepSeek API Key

本地运行时复制示例文件：

```bash
copy .streamlit\secrets.toml.example .streamlit\secrets.toml
```

macOS/Linux：

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

编辑 `.streamlit/secrets.toml`：

```toml
DEEPSEEK_API_KEY = "你的 DeepSeek API Key"
ADMIN_PASSWORD = "请修改为强密码"
```

不要把真实 API key 提交到 GitHub。

## 本地运行

```bash
streamlit run app.py
```

浏览器会打开本地实验页面。SQLite 数据库会自动创建在 `data/experiment.db`。

完整流程为：

```text
consent → pre_survey → instruction → writing task → summary task → code task → post_survey → end
```

管理员导出页面：

```text
http://localhost:8501/?admin=1
```

## Streamlit Community Cloud 部署

1. 将项目上传到 GitHub 仓库。
2. 确认以下文件已提交：
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `.streamlit/config.toml`
   - `.streamlit/secrets.toml.example`
   - `.gitignore`
3. 确认以下文件不要提交：
   - `.streamlit/secrets.toml`
   - `data/experiment.db`
   - `data/*.db`
   - 任何真实 API Key
4. 登录 Streamlit Community Cloud。
5. 选择 GitHub 仓库、分支和 `app.py`。
6. 在 Streamlit Cloud 的 Secrets 管理界面中添加：

```toml
DEEPSEEK_API_KEY = "你的 DeepSeek API Key"
ADMIN_PASSWORD = "强密码"
```

7. 点击 Deploy。
8. 部署完成后获得形如 `https://your-app-name.streamlit.app` 的链接。
9. 将该链接发给被试即可。
10. 如果需要给每个被试指定 ID，可以分享：

```text
https://your-app-name.streamlit.app/?pid=P001
```

11. 管理员导出数据访问：

```text
https://your-app-name.streamlit.app/?admin=1
```

## URL 参数

- `?pid=xxx`：指定匿名 participant_id。如果没有提供，系统自动生成 `P_` + uuid 前 10 位。
- `?admin=1`：进入管理员导出页面。
- `?pid=test_dynamic&condition=dynamic_feedback`：测试时强制指定实验组。`condition` 可取 `control`、`static_feedback`、`dynamic_feedback`。

如果同一个 `participant_id` 已经存在于数据库中，系统会读取原有 `condition`，不会重新随机分组。

## 数据表说明

系统启动时会自动创建以下 SQLite 表：

- `participants`：匿名参与者 ID、实验组、创建时间、完成状态、基本信息、前测指标和注意力检查结果。出生年月使用两个独立字段 `birth_year` 和 `birth_month`，不再使用 `1990-06` 这类年月合并字段；旧的 `age` 字段保留用于兼容历史数据库，但新表单不再填写或更新该字段。
- `messages`：每轮用户 prompt、AI 回复、时间戳、延迟、token 数、估算能耗、累计估算能耗、`task_index`、`task_copy_similarity`、`is_task_copying`、`context_copy_similarity`、`requirements_copy_similarity`、反馈文本和模型参数。
- `task_sessions`：每个任务的开始/结束时间、任务顺序 `task_index`、任务时长、请求次数、最终答案和任务后问卷。
- `questionnaires`：前测量表、后测低能耗偏好量表、feedback attention/pressure/guilt/fatigue 等量表和开放题回答。
- `events`：关键事件日志，包括 `consent_accepted`、`pre_survey_submitted`、`instruction_completed`、`task_started`、`prompt_submitted`、`task_copy_detected`、`ai_response_generated`、`energy_feedback_displayed`、`task_submitted`、`post_survey_submitted`、`experiment_completed`、`api_error`。

管理员页面可以选择导出以上表为 CSV，编码为 `utf-8-sig`，方便用 Excel 打开中文。

管理员页面还提供派生导出：

- `blind_rating_file.csv`
- `blind_rating_key.csv`
- `ai_scoring_input.csv`
- `prompt_coding_file.csv`
- `prompt_coding_template.csv`
- `requirement_checklist.csv`

`prompt_coding_file.csv` 包含 condition，供研究者内部使用；`prompt_coding_template.csv` 不包含 condition，适合给编码员盲评。

## 能耗估算说明

系统不测量 DeepSeek 或任何真实平台的实际能耗。能耗反馈只用于实验操纵和界面研究。

估算规则：

```python
ENERGY_PER_TOKEN_WH = 0.00005
LED_POWER_W = 0.6

estimated_energy_wh = total_tokens * ENERGY_PER_TOKEN_WH
estimated_led_minutes = estimated_energy_wh / LED_POWER_W * 60
```

如果 API 返回 `usage.total_tokens`，系统使用 API 的 token 数；如果 usage 为空，则使用字符数近似估算：

```python
estimated_total_tokens = round((len(user_prompt) + len(ai_response)) / 2)
```

所有界面和文档中都应明确：这些数值为研究目的下的估算反馈，不代表真实平台能耗。

## 隐私和伦理注意事项

- 系统不收集真实姓名、手机号、身份证号等直接身份信息。
- `participant_id` 是匿名 ID，可通过 URL 参数指定，也可由系统自动生成。
- 知情同意页会提醒参与者不要输入真实姓名、手机号、身份证号、住址、账号密码、商业机密或其他敏感信息。
- DeepSeek API 会接收用户输入用于生成回复。研究者需要根据 DeepSeek 平台条款、所在学校或机构要求处理数据合规问题。
- 能耗反馈为实验模拟估算，不代表真实能耗。
- 如果用于正式研究，应先通过所在机构的伦理审批或 IRB 审查。
- 如果使用 Streamlit Community Cloud，研究者需要自行评估数据存储和导出机制是否满足研究伦理和数据管理要求。

## Streamlit Cloud 数据备份提醒

如果部署在 Streamlit Community Cloud，SQLite 文件保存在应用运行环境中。Community Cloud 可能会重启应用，SQLite 数据库文件不适合作为长期唯一数据存储。

正式收集数据期间应定期从 `?admin=1` 管理员页面导出 CSV 备份。如果做正式大样本实验，建议后续升级为外部数据库，例如 Supabase、PostgreSQL 或 Google Sheets。

## 新颖性效应与 Feedback Fatigue

系统在 `messages` 和 `task_sessions` 中保存 `task_index`：

- writing = 1
- summary = 2
- code = 3

这可以初步检验动态能耗反馈是否在三个任务之间出现衰减。建议模型：

```text
Outcome ~ Condition + TaskIndex + Condition × TaskIndex
```

如果 `dynamic_feedback` 在后续任务中效果减弱，可能说明存在 feedback fatigue 或新颖性效应。如果效果持续存在，可以说明短期多任务环境中反馈作用较稳定。

本实验仍然是短期受控实验，不能证明长期真实产品使用中的持续效果。为此，post_survey 增加了 `q_feedback_fatigue`、`q_long_term_acceptance` 和开放题 `open_feedback_fatigue`。

## 后续分析建议

建议分析指标：

- `request_count`
- `total_tokens`
- `estimated_energy_wh`
- `task_copy_similarity`
- `is_task_copying`
- `requirement_transmission_count`
- `productive_iteration rate`
- `low_value_prompt rate`
- `strategic_prompting_score`
- `final_answer overall_quality`
- `requirement_hit_rate`
- `satisfaction`
- `cognitive_load`
- `q_feedback_pressure`
- `q_feedback_guilt`
- `q_interaction_penalty`
- `q_feedback_fatigue`
- `q_low_energy_wait`
- `q_low_energy_quality`
- `q_default_low_energy`

建议核心分析模型：

```text
Outcome ~ Condition + TaskIndex + Condition × TaskIndex + AI_Usage + Baseline_Energy_Awareness + Environmental_Attitude
```

对于多任务数据，可以使用 mixed-effects model：

```text
Outcome ~ Condition + TaskIndex + Condition × TaskIndex + controls + (1 | participant_id)
```

## 本地测试清单

- 不勾选知情同意，不能进入下一步。
- 前测问卷中不应出现 `1990-06` 这类合并选项。
- 前测问卷应分别出现出生年份下拉框和出生月份下拉框。
- 不选择出生年份不能提交前测。
- 不选择出生月份不能提交前测。
- `participants.csv` 中应出现 `birth_year` 和 `birth_month` 两列。
- 性别、教育、AI 使用频率为“请选择”时，不能提交前测。
- 注意力检查不选 6，可以继续，但 `attention_check_pass` 记录为 0。
- 每个任务 `final_answer` 为空，不能进入下一任务。
- 每个任务 `final_answer` 少于 20 字符，不能进入下一任务。
- `post_survey` 三个必填开放题为空，不能完成实验；其他开放题可以留空。
- `admin` 密码为空或错误，不能导出数据。
- 完整填写后，可以顺利走完 `consent → pre_survey → instruction → 三个 task → post_survey → end`。
- `?admin=1` 可以导出 `participants`、`messages`、`task_sessions`、`questionnaires`、`events`。
- 首页和全局标题应显示“AI 协作任务研究”，不应使用“能源感知实验”作为首屏标题。
- 任务页应呈现三栏工作台：左栏“AI 已知信息”、中栏“AI 对话区”、右栏“你的提交要求”。
- `control` 组的 instruction 和 task 页面不应出现能耗、电力、LED、环保、低能耗、资源成本、节能等干预信息。
- `static_feedback` 组在任务页只显示静态能耗提示，不显示本次或累计数值。
- `dynamic_feedback` 组能耗反馈卡片应显示本次估算值、累计估算值和实验估算注释。
- `task_sessions` 表中保存 `task_index`。
- `messages` 表中保存 `task_index`。
- `post_survey` 中出现新增的 feedback pressure / guilt / fatigue 题。
- `questionnaires` 表中能保存新增题项。
- admin 页面能导出 `prompt_coding_file.csv`。
- admin 页面能导出 `prompt_coding_template.csv`。
- admin 页面能导出 `requirement_checklist.csv`。
- `prompt_coding_template.csv` 不包含 `condition`。
- `prompt_coding_file.csv` 包含足够上下文用于编码。
- `dynamic_feedback` 组每次 AI 回复后仍显示能耗卡片。
- `control` 组仍不显示任何能耗信息。
- `static_feedback` 组仍只显示静态能耗提示。
- 出生年份和出生月份仍然分开选择。
- admin 页面仍能导出原始数据库表。

任务双层结构测试：

- 打开任意任务页面，页面应显示“AI 已知信息”和“你的提交要求”。
- 输入“请基于任务写一封邮件”“请基于任务总结”或“请帮我修改代码”时，AI 不会自动获得 `participant_only_requirements`。
- 检查代码中 DeepSeek messages 不包含 `participant_only_requirements`。
- 检查最终答案输入框使用 `task["final_answer_label"]`。

## 正式部署前 checklist

- DeepSeek API Key 已重置且可用
- 真实 API Key 没有提交到 GitHub
- `ADMIN_PASSWORD` 已设置强密码
- `DEBUG = False`
- 本地完整跑通过一次实验流程
- 三个组别都测试过
- `dynamic_feedback` 能显示能耗反馈
- `task_copy_similarity` 和 `is_task_copying` 能写入 `messages` 表
- DeepSeek API 出错时不会导致系统崩溃
- `?admin=1` 能导出 CSV
- 知情同意文本已确认
- 已完成 20-30 人 pilot
- 正式收集期间安排定期导出 CSV 备份
