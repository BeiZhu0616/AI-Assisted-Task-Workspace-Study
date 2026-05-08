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

本实验采用“减法设计”，尽量减少每页信息量，避免被试在实验流程中 lost。每页只保留当前阶段最必要的信息，不重复上一页已经说明过的内容。

首页只用于知情同意，标题为“知情同意”，不使用“能源感知实验”或英文研究标题作为首屏标题。知情同意中采用“估算反馈”的泛化表述，不提前突出能耗、电力、LED 或低能耗主题，以避免污染 control 组。

基本信息页和任务操作说明页同样采用减负设计。基本信息页顶部只显示一句“请根据您的真实情况填写以下信息。”；非任务页的“上一页 / 下一页”按钮居中相邻显示，减少视觉跳跃。

实验说明页只保留任务操作规则，不重复研究团队、数据记录和隐私说明。任务页从三栏改为“上下结合 + 两栏”结构：

- 顶部：任务标题和剩余提问次数；
- 第一块：你的任务；
- 第二块：AI 已知材料；
- 下方左栏：AI 对话区；
- 下方右栏：任务状态、static 提示或 dynamic 累计反馈；
- 底部：最终提交答案。

这种信息架构服务于 Paper 1 的核心定位：Energy Feedback as Reflective Friction。动态反馈组的每轮反馈用于形成轻量的反思性摩擦，但它不是强制限制，也不是惩罚。三组除能耗反馈外尽量保持一致，避免把任务材料呈现方式、prompt 指导或最终答案提交区变成混淆变量。

页面主容器限制在约 1200-1320px 宽度内，并保持左对齐阅读结构。任务标题采用具体任务名称：`任务一：撰写延期邮件`、`任务二：总结高校 AI 使用材料`、`任务三：解释并修正 Python 总收入代码`。

界面卡片由 [app.py](app.py) 中统一的 `render_card` / `render_bullet_card` 渲染。卡片正文会先进行 HTML escape，再由 `st.markdown(..., unsafe_allow_html=True)` 输出，以避免 `<`、`>`、换行或代码片段破坏 HTML 结构。任务三的 Python 代码单独使用 `st.code(..., language="python")` 显示，不放入 HTML 卡片正文。

## 能耗反馈机制与实验控制

三组的任务材料、任务结构、AI 模型、system prompt、temperature、max_tokens、最终答案提交区和问卷流程保持一致。唯一的界面操纵是能耗反馈：

- `control`：任务说明页和任务工作台不显示能耗相关信息；
- `static_feedback`：只在任务页状态区显示静态提示，不显示每轮数值；
- `dynamic_feedback`：每次 AI 成功回复后，在对话区下方显示本次估算能耗、当前任务累计估算能耗和 LED 灯点亮时间类比。

dynamic_feedback 的反馈卡片使用浅色中性视觉风格，目的是提升隐藏资源成本的显著性，形成 reflective friction，而不是制造惩罚感或阻止用户继续追问。

任务页中，`dynamic_feedback` 组会在右侧任务状态区显示“累计估算反馈”小卡片，并在每条 AI 回复下方显示 compact 本次反馈。`static_feedback` 组只显示静态提示，不显示本次或累计数值。

post_survey 按实验组显示题项：control 组只回答未来产品假设题；static_feedback 组额外回答静态提示体验题；dynamic_feedback 组额外回答动态反馈体验题。这样可以避免 control 组被不存在的能耗反馈题污染。

AI 提问次数按照当前任务中用户成功发送并获得 AI 回复的 prompt 数计算，不统计空输入、API 失败请求、AI 回复、最终答案提交或其他任务的 prompt。初始剩余次数为 6，每成功提问一次减少 1，剩余 0 时输入区停止接收新问题。

`dynamic_feedback` 组的反馈使用浅橙提示样式，并用 `💡` 表示 LED 灯类比。每条 AI 回复下方会立即显示 compact 本次反馈，右侧任务状态区同步显示累计估算反馈。

任务一的 AI 可见材料改为碎片化备忘录，而不是结构完整的写作框架。它不直接给出具体截止时间、延期天数或完成比例，避免用户只输入“帮我写一封邮件”就获得过于完整的答案。这有助于观察 `prompt_specificity_score`、`strategic_prompting_score`、`productive_iteration_rate` 和 `low_value_prompt_rate`。

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

## 三层任务结构

本实验采用“三层任务结构”，用于避免右栏成为可直接复制的 prompt 模板：

- `ai_visible_context`：左栏 AI 已知信息，只包含 AI 已经看到的原始背景材料、原始文本或原始代码。
- `participant_only_requirements`：右栏“你的任务目标”，只包含被试看见的笼统任务目标和最终交付物。为了兼容旧数据库字段，代码中仍沿用这个字段名，但前端不再显示详细 checklist、评分标准、禁用项或具体格式要求。
- 后台 broad quality rubric：只写在 README 或评分说明中，用于后期质量评分，不显示给被试，不传给 DeepSeek。

任务页采用三栏布局呈现任务工作台：

- `AI 已知信息`：展示 `ai_visible_context`，并提示这些背景材料已经提供给 AI；
- `AI 对话区`：展示通用 prompt guidance：“你可以告诉 AI 你想完成什么，也可以根据需要补充格式、语气或限制条件。”；
- `你的任务目标`：展示笼统任务目标，提醒被试自行决定如何向 AI 提问、是否让 AI 修改，以及最终采用哪一版答案。

这样设计的原因是：如果右栏显示详细评分标准或 checklist，用户可以直接复制右栏作为 prompt，实验就更容易测到复制行为而不是策略性提问。同时，最终答案质量评分也不应因为被试没有满足未被告知的具体格式要求而受到主评分惩罚。

本研究不再重点考察用户是否“传达了多少隐藏要求”，而是考察在较真实、开放的 AI 协作任务中，能耗反馈是否会促使用户更具体、更策略性地提问，减少低价值交互和机械复制，并在不牺牲任务质量的情况下提高使用效率。

## 复制任务检测

系统使用 Python 标准库 `difflib.SequenceMatcher` 计算用户 prompt 与任务两层材料的相似度：

```python
context_copy_similarity = SequenceMatcher(None, user_prompt.strip(), ai_visible_context.strip()).ratio()
right_goal_similarity = SequenceMatcher(None, user_prompt.strip(), participant_only_requirements.strip()).ratio()
task_copy_similarity = max(context_copy_similarity, right_goal_similarity)
```

如果 `task_copy_similarity >= 0.60`，则：

- `task_copy_similarity` 保存相似度
- `is_task_copying = 1`
- 在 `events` 表记录 `task_copy_detected`

系统也会在 `messages` 表中保存 `context_copy_similarity`、`requirements_copy_similarity`、`is_context_copying` 和 `is_requirements_copying`。其中 `requirements_copy_similarity` 为兼容旧字段名，当前语义是用户 prompt 与右栏笼统任务目标的相似度。

由于右栏现在只显示笼统任务目标，copying 不再表示“复制具体评分要求”，而表示用户是否直接复制界面任务文本。该指标应与 `prompt_specificity_score`、`strategic_prompting_score`、`low_value_prompt_rate` 和 `productive_iteration_rate` 结合解释。

后续人工编码建议：

- mechanical copying：用户直接复制界面任务文本或大段任务材料。
- strategic prompting：用户主动组织目标、限制条件、语气、受众、输出形式或修改方向。

建议增加人工编码变量：

- `prompt_specificity_score`，1-5 分：1 = 极其笼统；2 = 有基本目标但缺少方向；3 = 说明任务目标并有少量具体要求；4 = 目标清楚且包含受众、语气、格式或修改方向；5 = 高度具体，能有效组织任务目标、上下文、限制和输出期望。
- `strategic_prompting_score`，1-5 分：1 = 无策略，随手问或机械复制；2 = 有基本目标但依赖 AI 猜测；3 = 有一定策略，能说明部分重点；4 = 能主动组织目标、限制、受众或评价标准；5 = 能高效组织任务，减少不必要追问，同时保持结果质量。

## Prompt 行为分类

后期编码建议区分以下 prompt 类型：

- `generic_prompt`：笼统请求，例如“请基于任务写一封邮件”。
- `task_goal_copying`：直接复制右栏笼统任务目标。
- `context_copying`：直接复制左栏 AI 已知信息或大段原始材料。
- `strategic_prompting`：主动组织目标、受众、语气、格式、限制或修改方向。
- `mechanical_copying`：机械复制界面任务文本或大段材料。
- `productive_iteration`：有价值迭代，例如指出 AI 遗漏要求、要求修正具体问题、补充明确约束。
- `low_value_repetition`：低价值重复，例如无具体方向的“再好一点”“继续”“优化一下”。
- `off_task`：离题或无效请求。
- `other`：其他无法归类的 prompt。

本研究关心的不是简单减少所有多轮对话，而是判断能耗反馈是否减少低价值交互，同时不抑制有价值的人机协作。

`prompt_coding_template.csv` 建议包含以下编码字段：

- `prompt_type`
- `prompt_specificity_score`
- `strategic_prompting_score`
- `productive_iteration`
- `low_value_prompt`
- `mechanical_copying`
- `coder_id`
- `coder_notes`

`requirement_transmission_count` 不再作为核心指标；如研究者保留旧字段，也只建议作为 optional / not core 变量。

## Broad Quality Rubric

最终答案质量评分应使用 broad quality rubric，而不是隐藏 checklist。主质量评分不应因为被试没有满足未被告知的具体要求而扣分。

不应作为主质量扣分项的未告知要求包括：必须 150-200 字、必须正好三条风险、必须不提家庭原因、必须保留 for 循环、必须使用某个特定句式。这些可以作为 diagnostic coding，但不是主评分。

主评分维度建议包括：`task completion`、`relevance`、`clarity`、`coherence`、`usefulness`、`tone appropriateness`、代码任务中的 `correctness`、`real-world usability` 和 `overall quality`。

Task 1 backend quality rubric:

- Task completion: 是否清楚表达了延期申请；
- Reasonableness: 延期理由是否合理、可信、不过度夸张；
- Actionability: 是否让老师清楚知道学生希望如何处理，例如延期、补交或后续安排；
- Tone appropriateness: 语气是否礼貌、自然、不过度卑微；
- Clarity: 表达是否清楚、简洁、容易理解；
- Real-world usability: 是否适合真实发送给老师；
- Overall quality: 整体质量。

Diagnostic coding only：是否提到已完成大部分作业；是否提出明确新提交安排；是否出现不必要的家庭/身体原因；是否明显 AI 模板化。

Task 2 backend quality rubric:

- Relevance: 是否围绕左侧材料展开，没有跑题；
- Coverage: 是否覆盖主要好处和主要风险；
- Faithfulness / consistency: 是否忠实于原材料，没有编造不在材料中的重要事实；
- Coherence: 结构是否清楚，逻辑是否连贯；
- Usefulness: 是否对读者理解高校 AI 使用议题有帮助；
- Practical judgment: 是否包含合理判断或建议；
- Fluency: 语言是否通顺、自然、通俗；
- Overall quality: 整体质量。

Diagnostic coding only：是否提到学习效率；是否提到学习能力风险；是否提到学术诚信；是否提到公平性或教学管理；是否包含建议或判断。

Task 3 backend quality rubric:

- Bug identification: 是否准确指出原代码的问题；
- Correctness: 修正代码是否能正确计算总收入；
- Conceptual explanation: 是否解释了为什么原代码结果不对；
- Beginner suitability: 解释是否适合初学者理解；
- Conciseness: 是否简洁，不写成长篇无关教程；
- Overall usefulness: 作为代码学习说明是否有帮助；
- Overall quality: 整体质量。

Diagnostic coding only：是否指出 total 被覆盖；是否指出需要计算 price × quantity；是否使用累加逻辑；是否解释最后结果为什么不对。

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
- `prompt_specificity_score`
- `strategic_prompting_score`
- `low_value_prompt_rate`
- `productive_iteration_rate`
- `mechanical_copying_rate`
- `final_answer_quality`
- `quality_per_token = final_answer_quality / total_tokens`
- `quality_per_request = final_answer_quality / request_count`
- `quality_per_energy = final_answer_quality / estimated_energy_wh`
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
- `post_survey` 按组显示开放题：control 组 1 个必填，static_feedback 组 2 个必填，dynamic_feedback 组 3 个必填。
- `admin` 密码为空或错误，不能导出数据。
- 完整填写后，可以顺利走完 `consent → pre_survey → instruction → 三个 task → post_survey → end`。
- `?admin=1` 可以导出 `participants`、`messages`、`task_sessions`、`questionnaires`、`events`。
- 第一页标题应显示“知情同意”，且只保留两个主要信息框。
- 第一页不应出现“能耗”“电力”“LED”“低能耗”等强操纵词。
- 基本信息页顶部说明“请根据您的真实情况填写以下信息。”应完整显示，不被截断。
- 第二页不显示“前测问卷”，性别选项只有“男”“女”。
- 实验说明页只显示任务操作规则，不重复首页内容。
- 基本信息页和任务操作说明页的“上一页 / 下一页”按钮应居中且相邻。
- 任务页应呈现“任务目标 + AI 已知材料 + 两栏对话/状态 + 最终提交”的结构。
- 任务页初始剩余 AI 提问次数为 6，第一次成功提问后为 5，第二次成功提问后为 4，第六次成功提问后为 0。
- 剩余次数为 0 时，AI 输入区停止接收新问题，并提示提交最终答案。
- `control` 组的 instruction 和 task 页面不应出现能耗、电力、LED、环保、低能耗、资源成本、节能等干预信息。
- `static_feedback` 组在任务页只显示静态能耗提示，不显示本次或累计数值。
- `dynamic_feedback` 组累计反馈应为浅橙样式，每条 AI 回复后的 compact 反馈应包含 `💡 LED` 图标。
- `task_sessions` 表中保存 `task_index`。
- `messages` 表中保存 `task_index`。
- `post_survey` 中 control 组不出现实际反馈体验题。
- `post_survey` 中 static_feedback 组出现静态提示体验题。
- `post_survey` 中 dynamic_feedback 组出现动态反馈体验题。
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

任务三层结构测试：

- 打开任意任务页面，页面应显示“AI 已知信息”和“你的任务目标”。
- 输入“请基于任务写一封邮件”“请基于任务总结”或“请帮我修改代码”时，AI 不会自动获得 `participant_only_requirements`。
- 检查代码中 DeepSeek messages 不包含 `participant_only_requirements`。
- 检查最终答案输入框使用 `task["final_answer_label"]`。

Paper 1 任务设计测试：

- 任务一右栏不显示 150-200 字、家庭原因、身体原因等具体要求。
- 任务一 AI 可见材料不直接写“延期 3 天”或“完成 70%”。
- 任务二右栏不显示 120 字、三条风险、一条判断等 checklist。
- 任务三为订单总收入代码任务。
- 任务三前端不显示 total 被覆盖、price × quantity、+= 等答案线索。
- 任务一 AI 可见材料应为碎片化备忘录，不直接给出 `23:59`、延期 3 天或完成 70%。
- `prompt_coding_template.csv` 包含 `prompt_specificity_score` 和 `strategic_prompting_score`，不再将 `requirement_transmission_count` 作为核心字段。

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
