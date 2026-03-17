# Parking Bot (Stage 1) - Interview Preparation Guide

This guide provides a comprehensive overview of your Parking Bot task, the rationale behind your technical decisions, and 20 potential interview questions you might face, complete with strong, well-reasoned answers.

---

## 🏗️ Detailed Overview of the Task

The "Parking Bot" project is an autonomous multi-agent system designed to handle parking availability queries, conversational engagement, and reservation slot-filling. 

### Core Components implemented in Stage 1:

1. **Stateful Orchestration (`chatbot_graph.py`)**: 
   - Built on **LangGraph**, the system utilizes a `StateGraph` which passes a global state object (`AgentState`) containing conversation history (`messages`), extracted `user_info`, and `dialog_stage` variables between distinct nodes.
   - **Router Node**: Analyzes intent using a hybrid approach of exact keyword matching (for fast overrides) and an LLM classification prompt. 
   - **Functional Nodes**:
     - `rag_node`: Contextualizes follow-up questions and queries a Vector Store for rules and facts.
     - `dynamic_info_node`: Fetches real-time pricing and slot availability directly from a SQL Database.
     - `reservation_node`: Utilizes JSON-based structured extraction to perform slot-filling (Name, Plate, Time). It actively prompts the user until all fields are collected.
     - `conversation_node`: Extracts the user's name heuristically and handles general small talk.
     - `check_status_node`: Queries the SQL DB to return a user's reservation state.

2. **Data Privacy & Guardrails (`guardrails.py`)**:
   - Implements a hybrid approach using **GLiNER** (a lightweight Named Entity Recognition NLP model) to scan bot outputs.
   - It redacts highly sensitive PII (Phone numbers, Emails, precise Addresses) while explicitly allowing domain-specific non-sensitive data like "person names" and "license plates" to pass through to preserve the context of a parking application.
   - Contains a robust Regex fallback mechanism to ensure the system gracefully degrades if the local GLiNER model fails to load.

3. **Quantitative Evaluation (`evaluate.py`)**:
   - An offline evaluation pipeline testing against a golden dataset (`evaluation_set.json`).
   - It measures the system entirely programmatically on 4 key metrics: **Answer Accuracy** (End-to-End keyword hit rate), **Retrieval Recall** (Expected truth found in context), **Context Precision** (Relevance of retrieved chunks), and **Latency**.

---

## 🧠 Reasons for Architectural Solutions

1. **LangGraph over Standard Chains / ReAct Agents**: standard LLM agents easily get confused in cyclic tasks, like repeatedly asking a user for missing reservation details. LangGraph explicitly tracks states (`dialog_stage`), enabling deterministic routing and locking users into "slot-filling loops" until the required conditions are met.
2. **Hybrid Routing (Keywords + LLM)**: By prioritizing keywords (e.g., "hours", "cancel", "status"), the system avoids expensive and slower LLM invocations for obvious user intents. LLM intention analysis is kept exclusively for ambiguous edge cases.
3. **Targeted JSON Extraction for Reservations**: Because LLMs are prone to hallucination, asking it to output standard text can mess up internal variables. Dictating a strict JSON schema (`name`, `car_number`, `start_time`, `end_time`) ensures that intermediate variables are perfectly parsable and clean.
4. **GLiNER over OpenAI for Guardrails**: Sending output back to an LLM to check for PII is slow and doubles the token cost. GLiNER runs locally, is lightweight, and is much smarter than Regex at identifying complex addresses without breaking local data rules.
5. **Contextual Query Rewriting in RAG**: If a user asks "Are there any spots?", gets an answer, and follows up with "How much do they cost?", standard RAG fails because "they" lacks context. We pass the last 4 chat messages to an LLM to rewrite the query context (e.g. "How much does a parking spot cost?") *before* doing vector retrieval.
6. **Bypassing the LLM for Dynamic Data**: Calling `get_available_spots()` cleanly via Python in the `dynamic_info` node completely side-steps the vector DB. RAG should only be used for static documents; live data (like prices and spot availability) will become stale immediately in a Vector Store.

---

## ❓ 20 Example Interview Questions & Answers

### Architecture & Frameworks
**1. Why did you choose LangGraph over traditional LangChain Agents?**
*Answer*: LangGraph provides deterministic control flows and cyclic graphs through explicit state machines. It makes slot-filling (asking for missing info back-and-forth) highly reliable, whereas standard ReAct agents often get confused or stuck in loops attempting to use tools.

**2. How do you handle and persist conversation memory?**
*Answer*: Memory is maintained in the `AgentState` object which passes the `messages` array and a structured `user_info` dictionary between nodes. For the RAG logic, the last 4 messages are passed into the LLM context to ensure short-term memory remains relevant without blowing up token limits.

**3. What happens if a user inputs "nevermind" while halfway through making a reservation?**
*Answer*: The intent analyzer checks keyword heuristics first. It explicitly looks for "cancel", "nevermind", or "stop" and forces the router to return to `conversation_flow`, altering the `dialog_stage` and breaking the user out of the reservation loop.

### Routing & RAG
**4. How does the `router_node` decide where to direct the user input?**
*Answer*: It evaluates intent in layers. First, it respects explicit state rules (e.g., if we are actively in the `reservation` stage, loop back). Then, it checks keyword overrides (e.g., "status" -> `check_status_node`). Finally, if it's ambiguous, it falls back to a zero-shot LLM classifier.

**5. Why do you contextualize the query in the `rag_node` prior to retrieval?**
*Answer*: To handle pronouns in follow-up questions. If a user asks "What are the working hours?" and then asks "Is it open on Sundays?", the vector store won't understand "it". Contextualization rewrites the question to a standalone prompt combining history and current input, drastically improving vector retrieval accuracy.

**6. How do you prevent the LLM from confirming a reservation inside the informational RAG flow?**
*Answer*: Through explicit role isolation via prompts. The system prompt for the `rag_node` specifically instructs the model: "Do not confirm reservations here. The context is about parking rules." The reservation node is the *only* place that has access to the tool that natively modifies the SQL DB.

### Logic & Data Management
**7. How does the slot-filling approach in `reservation_node` work?**
*Answer*: It uses a structured LLM extraction prompt to pull `name`, `car_number`, `start_time`, and `end_time` into a strict JSON payload. The node evaluates missing keys. If any are missing, it responds asking for exactly what is missing and retains the `reservation` dialog stage until the payload is fully populated.

**8. If a user says "I want to park from 8 to 16", how do you separate the start and end times?**
*Answer*: The system prompt for the extraction LLM explicitly commands the model to handle time ranges by splitting them into two separate entities, preventing string overlaps.

**9. How does the system inform the user about currently available spots?**
*Answer*: Instead of pushing dynamic status into a Vector Store, we have a specialized `dynamic_info_node`. This completely bypasses the LLM processing and runs a direct query (`get_available_spots()`) on the SQL database to construct a factual, deterministic response.

**10. What is a potential major bottleneck in your `reservation_node` regarding the admin approval?**
*Answer*: Currently, the node uses synchronous active polling (`time.sleep`) to wait for a database change to `confirmed`. In a highly concurrent system, this blocks the server thread. To fix this, we'd use LangGraph's `interrupt_before` (which is standard in Stage 4) to pause graph execution entirely until an external webhook or Admin payload restarts it.

### Guardrails (Security / PII)
**11. What is GLiNER and why use it over Regex for PII detection?**
*Answer*: GLiNER is a highly efficient, locally-run Named Entity Recognition model. It is substantially better than Regex at identifying context-specific PII like complex physical addresses, without requiring the latency or token cost of sending outputs back to GPT-4.

**12. Why do you intentionally skip redacting License Plates and User Names in your guardrails?**
*Answer*: It is a product-specific decision. In a parking application, the bot must repeat the user's name and license plate back to them to confirm the correct reservation. Redacting them would break the conversational context entirely. We only redact strictly dangerous PII (Phones, Emails, standard Addresses).

**13. What happens if the GLiNER model fails to load or the server runs out of RAM?**
*Answer*: The system is built with fault tolerance. The `filter_sensitive_data` block wraps the model in a `try/except` block and cascades down to a `filter_sensitive_data_regex` fallback string analyzer, ensuring basic security is met without crashing the process.

**14. What environment configurations were needed to keep GLiNER running smoothly?**
*Answer*: Setting `HF_HUB_DISABLE_PROGRESS_BARS="1"` and `TOKENIZERS_PARALLELISM="false"`. Without this, HuggingFace spams the console logger with download bars and model weight initializations, which would visually pollute our CLI interactions.

### Evaluation & Metrics
**15. How do you evaluate the reliability of your bot infrastructure?**
*Answer*: I run an offline evaluation script (`evaluate.py`) against a static `evaluation_set.json`. It passes synthetic User messages to the graph and checks the resulting metadata to provide a programmatic report for CI pipelines.

**16. The evaluation calculates "Context Precision". What does that mean?**
*Answer*: Context Precision evaluates the *Relevance of the chunks*. It determines the fraction of retrieved chunks that actually contained the expected answer keywords. This ensures we aren't pulling in excessively large, irrelevant data sets that cost tokens and distract the LLM.

**17. What is the difference between Retrieval Recall and Answer Accuracy?**
*Answer*: Retrieval Recall measures the Vector Database—did it fetch the document containing the truth? Answer Accuracy measures the LLM End-to-End performance—did the generated bot response actually relay that truth accurately to the user?

**18. Why include Latency as a primary tracking metric during script evaluation?**
*Answer*: Chaining complex LLM tasks (Routing -> Rewriting context -> RAG generation -> JSON extraction -> PII filtering) can result in unacceptably slow user wait times. Tracking latency ensures that adding a new LLM call to the graph doesn't push the response time beyond a viable UX limit.

### LLM Configurations
**19. Why use `gpt-3.5-turbo` with `temperature=0` here?**
*Answer*: In routing logic and JSON data extraction, creativity causes catastrophic system failure. `temperature=0` provides the most deterministic, reproducible outputs, which is strictly needed for control flow state switches. 

**20. What is the biggest challenge when passing chat history to an LLM context?**
*Answer*: Accumulating token limits. As the conversation grows, sending the entire history increases latency, costs more API credits, and can cause the LLM to get distracted by earlier subjects. The solution used here is a sliding window approach, passing only the last 4 inputs to contextualizing prompts (`messages[:-1][-4:]`).
