# 🤖 Non-Linear Agent: LangGraph + Mistral + Flask Web App

A modern real-time web application powered by **LangGraph**, **Ollama + Mistral**, and **Flask + WebSockets**. This AI agent intelligently routes user queries (math, translation, summarization, general chat) through specialized logic paths and responds live in a sleek frontend.

---

## 🚀 Features

- 🌐 **Real-time WebSocket communication**
- 🔁 **Non-linear query routing** via LangGraph
- 🧮 **Math solver** (arithmetic + word problems)
- 📝 **Text summarizer** (LLM-based)
- 🌍 **Language translation**
- 💬 **General fallback conversation**
- 🎨 **Responsive modern UI** with animations and metadata display

---

## 🧠 Tech Stack

| Layer        | Technology                      |
|-------------|----------------------------------|
| Backend      | Flask, Flask-SocketIO, LangGraph, Ollama |
| LLM Engine   | [Mistral](https://ollama.com/library/mistral) via Ollama |
| Frontend     | HTML, CSS, Vanilla JS, Socket.IO |
| AI Orchestration | LangGraph (state-based decision routing) |

---

## ⚙️ Requirements

- Python 3.8+
- [Ollama](https://ollama.com/) installed locally with `mistral` model pulled
- Node.js (optional, for running other frontends)
- pip packages (see below)

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/nonlinear-agent-webapp.git
cd nonlinear-agent-webapp
