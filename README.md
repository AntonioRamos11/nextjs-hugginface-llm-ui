<div align="center">
  <img src="huggingface-nextjs-ui.gif">
</div>

<h1 align="center">
  Fully-featured web interface for HuggingFace LLMs
</h1>

<div align="center">
  
![GitHub Repo stars](https://img.shields.io/github/stars/jakobhoeg/nextjs-hugginface-llm-ui)
  
</div>

Get up and running with Large Language Models **quickly**, **locally** and even **offline**.
This project aims to be the easiest way for you to get started with HuggingFace LLMs. No tedious and annoying setup required!

> This is a hobby project that provides a simple way to interact with DeepSeek and other HuggingFace models through a FastAPI backend with an Ollama-compatible API.

# Features ✨

- **Beautiful & intuitive UI:** Inspired by ChatGPT, to enhance similarity in the user experience.
- **HuggingFace models support:** Interface with powerful DeepSeek and other models from HuggingFace.
- **Ollama API compatibility:** Uses a FastAPI backend that implements Ollama-compatible endpoints.
- **Fully local:** Stores chats in localstorage for convenience. No need to run a database.
- **Fully responsive:** Use your phone to chat, with the same ease as on desktop.
- **Easy setup:** No tedious and annoying setup required. Just clone the repo and you're good to go!
- **Code syntax highlighting:** Messages that include code, will be highlighted for easy access.
- **Copy codeblocks easily:** Easily copy the highlighted code with one click.
- **Switch between models:** Switch between HuggingFace models fast with a click.
- **Chat history:** Chats are saved and easily accessed.
- **Light & Dark mode:** Switch between light & dark mode.

# Requisites ⚙️

To use the web interface, these requisites must be met:

1. Python environment for running the FastAPI backend (Python 3.8+ recommended)
2. Node.js (18+) and npm for the frontend. [Download](https://nodejs.org/en/download)

# Backend Setup

**1. Clone the repository:**

```
git clone https://github.com/jakobhoeg/nextjs-hugginface-llm-ui
```

**2. Install Python requirements:**

```
pip install fastapi uvicorn pydantic transformers torch
```

**3. Start the FastAPI backend:**

```
python app.py
```

This will start the FastAPI server on http://localhost:11435, which implements Ollama-compatible endpoints but uses HuggingFace models.

# Frontend Setup

**1. Open the folder:**

```
cd nextjs-hugginface-llm-ui
```

**2. Rename the `.example.env` to `.env`:**

```
mv .example.env .env
```

**3. Update the OLLAMA_URL in the .env file to point to the FastAPI backend:**

```
OLLAMA_URL="http://localhost:11435"
```

**4. Install dependencies:**

```
npm install
```

**5. Start the development server:**

```
npm run dev
```

**6. Go to [localhost:3000](http://localhost:3000) and start chatting with HuggingFace models!**

# Available Models

The FastAPI backend currently supports the following models:
f
- **deepcoder-1.5b**: Optimized for coding tasks

You can add more models by updating the models dictionary in the FastAPI backend.

# Docker Setup

## Running with Docker

Build and run the FastAPI backend:

```
run the api_server.py  - choose your port actually is 11435
```


# Tech stack

[NextJS](https://nextjs.org/) - React Framework for the Web
[FastAPI](https://fastapi.tiangolo.com/) - Modern API framework for Python
[HuggingFace Transformers](https://huggingface.co/docs/transformers/index) - State-of-the-art ML models
[TailwindCSS](https://tailwindcss.com/) - Utility-first CSS framework
[shadcn-ui](https://ui.shadcn.com/) - UI component built using Radix UI and Tailwind CSS
[shadcn-chat](https://github.com/jakobhoeg/shadcn-chat) - Chat components for NextJS/React projects
[Framer Motion](https://www.framer.com/motion/) - Motion/animation library for React
[Lucide Icons](https://lucide.dev/) - Icon library

# Upcoming features

This is a to-do list consisting of upcoming features.

- ✅ Voice input support
- ✅ Code syntax highlighting
- ✅ Ability to send an image in the prompt to utilize vision language models.
- ✅ Ability to regenerate responses
- ✅ Support for HuggingFace models
- ⬜️ Import and export chats