Project: Agentic Debate and Persuasion Between Conspiracy and Scientific Models
Team: SixSeven
________________________________________

Project Output: An end-to-end system that simulates structured debates between two LLM-based agents (a conspiracy proponent and a scientific opponent) moderated by a control agent, runs repeated experiments across multiple conspiracy theories and models, logs all interactions in a unified debate format, and analyzes persuasion dynamics and language use. 

Familiarization with LLM Models
The objective of this task is to familiarize with different LLM backends that will later be used as debating agents.

Models / Platforms:
	Gemini (free daily access via API)
https://aistudio.google.com/api-keys
	Dockerized Ollama with local models (e.g. LLaMA, Qwen, Mistral) 
https://ollama.com/
	OpenAI API (discuss with me if needed).

Possible Steps (Adjustable): 
	Set up access to Gemini API and run simple prompts
	Install and run Ollama locally via Docker
	For your own understanding compare:
	response length
	coherence
	tendency to hedge or assert
	Document limitations (rate limits, latency, hallucinations)

Agentic Debate System (Conspiracy vs Scientific Evidence)
The objective is to design and implement a multi-agent debate system in which two agents with opposing epistemic positions debate a conspiracy claim over multiple turns, under the supervision of a moderator agent.

Roles:
	Proponent Agent: Argues in favor of a conspiracy claim provided at the beginning e.g. COVID-19 vaccines contain microchips.
	Opponent Agent: Argues against the claim using scientific or evidence-based reasoning.
	Moderator Agent: Controls turn-taking, tracks rounds, and decides when the debate ends (either if one of the debating agents change their minds, or after k turns).

The focus is not factual correctness, but persuasion attempts and resistance to persuasion.

Debate Orchestration:
Students may choose one orchestration framework
	CrewAI https://www.crewai.com/
	LangGraph
	AutoGen (or equivalent)

Access to Information:
	Each debating agent may use Tavily for online search during the debate.
https://www.tavily.com/pricing
	The proponent will search fringe communities and biased queries e.g. site:reddit.com/r/conspiracy.
	The opponent will use mainstream sources e.g. fact-checkers, scientific organizations and reports, etc).

Canonical Debate Log Format: All debates (synthetic or human) must be logged in a common JSON schema, enabling later comparison. This information will be extracted by the moderator.
	debate_id
	claim
	round
	speaker_role (proponent | opponent | moderator)
	utterance
	stance (pro | con | neutral)
	confidence (optional, numeric or categorical)
	tool_used (none | tavily)
	tool_query
	reply_to_turn

Sources:
	https://blog.gopenai.com/i-built-an-ai-debate-team-to-settle-the-pineapple-on-pizza-argument-32d69291d3cf
	https://www.digitalocean.com/community/tutorials/local-ai-agents-with-langgraph-and-ollama
	https://microsoft.github.io/autogen/stable//user-guide/core-user-guide/design-patterns/group-chat.html
	https://github.com/nikhil-verma-ml/DebateGraph



Debate Argumentation & Persuasion Analysis
The objective of this task is to analyze argumentation patterns, language use, and persuasion dynamics in debates.

Before synthetic agent debates are available, real debate datasets can be used to set up and validate the analysis pipeline.
	Intelligence Squared Debates (IQ2): https://tisjune.github.io/research/iq2
	So you dont have to wait until the output of the debate system, you can use this dataset to process it and analyze it as if it was the dialogue from the agents.
	design analysis code
	test metrics
	establish human persuasion baselines

Data Preparation: 
	Convert debates into the same canonical debate log format used by agents
	Assign roles (proponent / opponent / moderator)
	Preserve turn order and debate boundaries

Analysis Objectives: Analyze debates (human and synthetic) with respect to:
	Language use:
	uncertainty: https://drive.google.com/drive/folders/1vm1jN69BapA36-JbEsYf5zrb9FBfmii0?usp=drive_link
	moral framing: https://github.com/medianeuroscience/emfd
	modality (strong https://drive.google.com/file/d/1t13LePb_h19Qnez9h5f9qp1u1hRZCxSV/view?usp=drive_link and weak https://drive.google.com/file/d/1r4zSOp-bC7bI0IAnx6Iph7ZP52SN0QZV/view?usp=drive_link)
	emotion https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
	Persuasion outcomes:
	stance stability or change
	confidence shifts over turns
	Comparison:
	human vs LLM debates
	model vs model (Gemini vs Ollama)


