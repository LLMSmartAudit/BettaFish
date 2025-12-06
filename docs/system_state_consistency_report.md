# BettaFish System Report

## How the system works
BettaFish coordinates four specialized agents to deliver public-opinion insights from query to formatted report. The Query, Media, and Insight agents start in parallel after the Flask entrypoint receives a user question, each running a tool-augmented workflow for their modality (web search, multimodal content, or private data). A ForumEngine mediates iterative collaboration cycles between these agents before the Report agent fuses their results, selects templates, plans layout and word budgets, and renders the final HTML/PDF output.【F:README-EN.md†L91-L200】

### Agent pipelines and responsibilities
- **QueryEngine** – Broad news and web search; manages search histories and paragraph-level research state while refining summaries that ultimately feed the report generator.【F:README-EN.md†L127-L139】【F:QueryEngine/state/state.py†L12-L205】
- **MediaEngine** – Multimodal understanding for images and video; mirrors the QueryEngine’s modular structure and contributes synthesized media findings to the report pipeline.【F:README-EN.md†L136-L144】
- **InsightEngine** – Private-database mining with keyword optimization, SQL helpers, and sentiment tools, maintaining the same research/paragraph state contract as QueryEngine for downstream merging.【F:README-EN.md†L145-L167】【F:InsightEngine/state/state.py†L12-L205】
- **ReportEngine** – Orchestrates template selection, layout planning, and chapter generation, then stitches validated IR blocks into interactive HTML and PDF exports.【F:README-EN.md†L168-L200】【F:ReportEngine/state/state.py†L12-L142】

### Detailed agent workflows (subsections)

#### QueryEngine: search-driven news synthesis
- **Initialization & tools** – Binds the configured LLM, a six-tool Tavily search agency, and five reasoning nodes (structure, initial search, reflection, two-stage summarization, report formatting) while bootstrapping an empty state and output directories.【F:QueryEngine/agent.py†L29-L139】
- **Paragraph-plan algorithm** – `research` invokes `ReportStructureNode` to draft paragraph titles/targets, saves them to state, then iterates each paragraph through `_initial_search_and_summary` and `_reflection_loop` to accumulate search histories and refinements before marking completion.【F:QueryEngine/agent.py†L141-L215】 Within each paragraph, the agent derives a tool-specific query, executes the chosen Tavily endpoint (with date-range validation), formats the results for prompting, and produces a first summary; subsequent reflection rounds re-query with adjusted prompts and merge new findings into the paragraph state.【F:QueryEngine/agent.py†L217-L343】
- **Finalization** – After all paragraphs are marked done, `ReportFormattingNode` assembles the latest summaries into a Markdown report, persists it (plus optional JSON state checkpoint), and flags the agent as completed for downstream consumption.【F:QueryEngine/agent.py†L344-L447】

#### MediaEngine: multimodal search-and-refine
- **Initialization & toolchain** – Mirrors QueryEngine’s node layout but swaps in the Bocha multimodal search suite (comprehensive/web-only/structured/recency tools) and media-focused model endpoints, seeding state and directories up front.【F:MediaEngine/agent.py†L29-L135】
- **Iterative pipeline** – Generates report structure, then loops paragraphs through initial search/summarize and reflection rounds that call the configured Bocha tool, track search results in state, and update summaries after each pass before marking completion and computing progress.【F:MediaEngine/agent.py†L136-L335】 The consistent paragraph algorithm keeps media insights aligned with text agents for easy merging.
- **Report handoff** – Uses `ReportFormattingNode` to consolidate paragraph latest states, writes Markdown outputs, and optionally persists intermediate state artifacts for reproducibility.【F:MediaEngine/agent.py†L336-L423】

#### InsightEngine: optimized database mining
- **Initialization & specialty tools** – Couples the LLM with MediaCrawlerDB query helpers, a multilingual sentiment analyzer, and the keyword-optimization middleware to pre-process any search terms before hitting the database endpoints, initializing the same node stack and state shape as other engines.【F:InsightEngine/agent.py†L29-L120】
- **Query optimization algorithm** – Each tool invocation first runs keyword optimization, fans out optimized terms across target queries (global search, date-scoped, platform-specific, comments), aggregates results, and optionally injects sentiment analysis outcomes into the response payload before updating paragraph state.【F:InsightEngine/agent.py†L105-L220】
- **Paragraph refinement loop** – The research cycle mirrors the structure/summarize/reflect pipeline of the other agents, so InsightEngine outputs maintain identical paragraph schemas for merging in the ReportEngine.【F:InsightEngine/agent.py†L141-L260】

#### ReportEngine: template-to-render pipeline
- **Baseline intake & readiness checks** – Initializes file-count baselines for Query/Media/Insight report directories so the web layer can verify fresh Markdown inputs and fetches the latest files for ingestion.【F:ReportEngine/agent.py†L46-L170】
- **Reasoning stages** – Drives four nodes in order: template selection, document layout, word budgeting, and chapter generation (with structured JSON parsing safeguards and retries) before composing sections into an internal representation validated by `IRValidator`.【F:ReportEngine/agent.py†L1-L220】
- **Rendering & persistence** – Uses `HTMLRenderer` to emit interactive reports, stores chapter artifacts via `ChapterStorage`, and records status/metadata in `ReportState`, allowing resumable tasks and deterministic downstream serving.【F:ReportEngine/agent.py†L1-L220】

## State consistency mechanisms
BettaFish solves state consistency by giving every agent explicit, serializable state models and by aligning their progress signals so the ReportEngine can combine outputs deterministically.

### Structured research states per agent
- Query and Insight agents define identical dataclass hierarchies (`Search`, `Research`, `Paragraph`, `State`) to capture search inputs, iterative reflections, completion flags, and timestamped summaries. They provide `to_dict`/`from_dict` plus file save/load helpers, ensuring reproducible checkpoints and consistent schema across agents before results enter the forum or report stages.【F:QueryEngine/state/state.py†L12-L258】【F:InsightEngine/state/state.py†L12-L258】

### Report-side task state and progress
- The ReportEngine keeps a `ReportState` that records the task ID, merged agent reports, selected template, and rendered HTML. Progress estimation is derived from template selection and HTML availability, while metadata such as generation time and template choice are captured in `ReportMetadata`. Serialization intentionally omits bulky HTML bodies to avoid drift between runtime and persisted snapshots, preserving only the fields necessary to resume or audit a run.【F:ReportEngine/state/state.py†L12-L142】

### Cross-agent alignment
- By sharing paragraph-level completion checks and JSON-compatible state serialization, upstream agents deliver consistent intermediate artifacts to the ReportEngine. The report layer’s state model then tracks downstream milestones (processing vs. completed) and metadata, preventing race conditions or partial renders from being treated as finished outputs.【F:QueryEngine/state/state.py†L142-L258】【F:ReportEngine/state/state.py†L37-L142】

## How mathematical methods support the system
BettaFish couples its LLM agents with conventional machine-learning pipelines to ground media understanding, sentiment scoring, and quality control in statistically measurable models.

### Traditional ML sentiment stack
- **Naive Bayes baseline** – Uses bag-of-words counts with a multinomial likelihood to provide a fast, memory-light polarity classifier for Chinese microblogs.【F:SentimentAnalysisModel/WeiboSentiment_MachineLearning/bayes_train.py†L8-L94】 The model persists vectorizers and weights, allowing deterministic reuse during InsightEngine runs.
- **SVM with TF-IDF** – Applies a kernelized support-vector classifier over TF-IDF features to improve margin-based separation between positive and negative samples, exposing kernel/C/gamma tuning for domain adaptation.【F:SentimentAnalysisModel/WeiboSentiment_MachineLearning/svm_train.py†L8-L98】 This balances bias/variance for sparse textual data while keeping calibration via probability estimates.
- **Gradient boosting (XGBoost)** – Trains tree ensembles on capped bag-of-words features with tunable depth, learning rate, and class-weighting, and reports AUC in addition to accuracy/F1 to monitor ranking quality for imbalanced topics.【F:SentimentAnalysisModel/WeiboSentiment_MachineLearning/xgboost_train.py†L9-L158】 The richer metrics surface to evaluators before models are handed to downstream agents.

### Neural sentiment and multilingual coverage
- The ML suite also ships LSTM and BERT classifiers (and multilingual variants) so teams can swap in sequence-aware or pretrained language representations when higher recall is needed; benchmark tables make it easy to compare their accuracy/AUC trade-offs.【F:SentimentAnalysisModel/WeiboSentiment_MachineLearning/README.md†L5-L107】 InsightEngine’s sentiment tool wraps a multilingual transformers classifier, exposing five-level labels and GPU/CPU device selection so agents can invoke high-capacity models when dependencies are available.【F:InsightEngine/tools/sentiment_analyzer.py†L1-L157】

### How agents benefit
- InsightEngine calls the multilingual sentiment analyzer during paragraph refinement, enriching database query results with probability-weighted sentiment labels that can be merged alongside Query/Media findings.【F:InsightEngine/agent.py†L110-L159】 The report pipeline then treats these scores as grounded evidence that complements LLM narratives, improving cross-source consistency during chapter synthesis.

## Key takeaways
- A unified architecture coordinates specialized agents through an iterative forum and a report orchestration pipeline that culminates in HTML/PDF delivery.【F:README-EN.md†L91-L200】
- Identical research-state schemas in Query and Insight agents standardize how searches, reflections, and paragraph completion are tracked and persisted.【F:QueryEngine/state/state.py†L12-L258】【F:InsightEngine/state/state.py†L12-L258】
- The ReportEngine’s task state guards consistency during rendering by explicitly modeling progress, selected templates, and metadata while excluding heavy payloads from persistence.【F:ReportEngine/state/state.py†L12-L142】
