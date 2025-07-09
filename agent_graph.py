#!/usr/bin/env python3
"""
Flask Web Application for Non-Linear Agent with LangGraph + Mistral
Complete web interface with modern UI and real-time interaction
"""

import asyncio
import re
import json
import logging
from datetime import datetime
from typing import Dict, Any, TypedDict
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, END
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")


class AgentState(TypedDict):
    """State definition for the agent graph"""
    input: str
    output: str
    next: str
    intermediate_results: Dict[str, Any]
    processing_time: float
    timestamp: str


class NonLinearAgent:
    """Non-linear agent with multiple specialized nodes"""

    def __init__(self, model_name: str = "mistral"):
        """Initialize the agent with Mistral LLM via Ollama"""
        self.model_name = model_name
        self.is_initialized = False
        self.error_message = None

        try:
            self.llm = Ollama(model=model_name, temperature=0.1)
            # Test the connection
            test_response = self.llm.invoke("Hello")
            self.graph = self._build_graph()
            self.app = self.graph.compile()
            self.is_initialized = True
            logger.info(f"Successfully initialized Ollama with model: {model_name}")
        except Exception as e:
            self.error_message = str(e)
            logger.error(f"Failed to initialize Ollama: {e}")

    def _build_graph(self) -> StateGraph:
        """Build the non-linear agent graph with multiple nodes"""
        graph = StateGraph(AgentState)

        # Add all nodes to the graph
        graph.add_node("router", self._router_node)
        graph.add_node("math_solver", self._math_solver_node)
        graph.add_node("text_summarizer", self._text_summarizer_node)
        graph.add_node("translator", self._translator_node)
        graph.add_node("final_printer", self._final_printer_node)

        # Set entry point
        graph.set_entry_point("router")

        # Add conditional edges from router to specialized nodes
        graph.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "math": "math_solver",
                "summarize": "text_summarizer",
                "translate": "translator",
                "fallback": "final_printer"
            }
        )

        # Connect all specialized nodes to final printer
        graph.add_edge("math_solver", "final_printer")
        graph.add_edge("text_summarizer", "final_printer")
        graph.add_edge("translator", "final_printer")
        graph.add_edge("final_printer", END)

        return graph

    def _router_node(self, state: AgentState) -> AgentState:
        """Router node that analyzes input and determines routing"""
        user_input = state["input"].lower()

        if "intermediate_results" not in state:
            state["intermediate_results"] = {}

        # Enhanced routing logic
        if any(op in user_input for op in ["+", "-", "*", "/", "calculate", "solve", "math", "equation"]):
            next_node = "math"
            state["intermediate_results"]["route_reason"] = "Mathematical operation detected"
        elif any(keyword in user_input for keyword in ["summarize", "summary", "tldr", "brief", "shorten"]):
            next_node = "summarize"
            state["intermediate_results"]["route_reason"] = "Text summarization requested"
        elif any(keyword in user_input for keyword in
                 ["translate", "translation", "convert to", "in spanish", "in french"]):
            next_node = "translate"
            state["intermediate_results"]["route_reason"] = "Translation requested"
        else:
            next_node = "fallback"
            state["intermediate_results"]["route_reason"] = "General query - using fallback"

        state["intermediate_results"]["routing_decision"] = next_node
        state["next"] = next_node

        return state

    def _route_decision(self, state: AgentState) -> str:
        """Helper function to return the routing decision"""
        return state["next"]

    def _math_solver_node(self, state: AgentState) -> AgentState:
        """Math solver node with enhanced capabilities"""
        user_input = state["input"]

        # Try simple arithmetic first
        math_pattern = r'^[\d+\-*/\s().]+$'
        if re.match(math_pattern, user_input.replace(' ', '')):
            try:
                result = eval(user_input)
                state["output"] = f"**Mathematical Result:** {result}"
                state["intermediate_results"]["math_type"] = "simple_arithmetic"
                state["intermediate_results"]["calculation"] = f"{user_input} = {result}"
            except:
                state["output"] = self._solve_with_llm(user_input)
                state["intermediate_results"]["math_type"] = "llm_solved"
        else:
            state["output"] = self._solve_with_llm(user_input)
            state["intermediate_results"]["math_type"] = "word_problem"

        return state

    def _solve_with_llm(self, math_problem: str) -> str:
        """Use Mistral LLM to solve complex math problems"""
        prompt = f"""
        You are an expert mathematician. Solve this problem step by step:

        Problem: {math_problem}

        Provide a clear, step-by-step solution with the final answer clearly marked.
        """

        try:
            response = self.llm.invoke(prompt)
            return f"**Math Solution:**\n\n{response}"
        except Exception as e:
            return f"**Error:** Unable to solve math problem - {str(e)}"

    def _text_summarizer_node(self, state: AgentState) -> AgentState:
        """Enhanced text summarizer node"""
        user_input = state["input"]

        # Extract text to summarize
        text_to_summarize = re.sub(r'\b(summarize|summary|tldr|brief|shorten)\b:?\s*', '', user_input,
                                   flags=re.IGNORECASE)

        if len(text_to_summarize.strip()) < 20:
            state["output"] = "**Note:** Please provide more substantial text to summarize."
            state["intermediate_results"]["summary_type"] = "insufficient_text"
        else:
            prompt = f"""
            Create a concise and informative summary of the following text. 
            Focus on the main points and key insights:

            Text: {text_to_summarize}

            Summary:
            """

            try:
                response = self.llm.invoke(prompt)
                state["output"] = f"**Text Summary:**\n\n{response}"
                state["intermediate_results"]["summary_type"] = "llm_generated"
                state["intermediate_results"]["original_length"] = len(text_to_summarize)
                state["intermediate_results"]["summary_length"] = len(response)
            except Exception as e:
                state["output"] = f"**Error:** Unable to create summary - {str(e)}"
                state["intermediate_results"]["summary_type"] = "error"

        return state

    def _translator_node(self, state: AgentState) -> AgentState:
        """Enhanced translator node"""
        user_input = state["input"]

        # Enhanced translation pattern matching
        patterns = [
            r'translate\s+["\']?(.+?)["\']?\s+to\s+(\w+)',
            r'convert\s+["\']?(.+?)["\']?\s+to\s+(\w+)',
            r'say\s+["\']?(.+?)["\']?\s+in\s+(\w+)'
        ]

        match = None
        for pattern in patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                break

        if match:
            text_to_translate = match.group(1).strip()
            target_language = match.group(2).strip()

            prompt = f"""
            Translate the following text to {target_language}. 
            Provide accurate translation with proper grammar:

            Text: {text_to_translate}
            Target Language: {target_language}

            Translation:
            """

            try:
                response = self.llm.invoke(prompt)
                state["output"] = f"**Translation to {target_language.title()}:**\n\n{response}"
                state["intermediate_results"]["translation_type"] = "successful"
                state["intermediate_results"]["source_text"] = text_to_translate
                state["intermediate_results"]["target_language"] = target_language
            except Exception as e:
                state["output"] = f"**Error:** Translation failed - {str(e)}"
                state["intermediate_results"]["translation_type"] = "error"
        else:
            state["output"] = "**Translation Help:** Please use format like 'translate [text] to [language]'"
            state["intermediate_results"]["translation_type"] = "invalid_format"

        return state

    def _final_printer_node(self, state: AgentState) -> AgentState:
        """Enhanced final printer with better formatting"""
        if not state.get("output"):
            # Fallback for general queries
            prompt = f"""
            You are a helpful AI assistant. Provide a comprehensive and helpful response to:

            Query: {state['input']}

            Response:
            """

            try:
                response = self.llm.invoke(prompt)
                state["output"] = f"**General Response:**\n\n{response}"
                state["intermediate_results"]["response_type"] = "general_fallback"
            except Exception as e:
                state["output"] = f"**Error:** Unable to process request - {str(e)}"
                state["intermediate_results"]["response_type"] = "error"

        # Add metadata
        state["intermediate_results"]["node_path"] = f"router → {state.get('next', 'unknown')} → final_printer"
        state["timestamp"] = datetime.now().isoformat()

        return state

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process query and return detailed results"""
        if not self.is_initialized:
            return {
                "success": False,
                "error": f"Agent not initialized: {self.error_message}",
                "output": "❌ Agent initialization failed. Please check Ollama setup.",
                "route": "error",
                "processing_time": 0
            }

        start_time = time.time()

        try:
            initial_state = {
                "input": query,
                "output": "",
                "next": "",
                "intermediate_results": {},
                "processing_time": 0,
                "timestamp": ""
            }

            result = self.app.invoke(initial_state)
            processing_time = time.time() - start_time

            return {
                "success": True,
                "output": result["output"],
                "route": result["intermediate_results"].get("routing_decision", "unknown"),
                "route_reason": result["intermediate_results"].get("route_reason", ""),
                "node_path": result["intermediate_results"].get("node_path", ""),
                "processing_time": round(processing_time, 2),
                "timestamp": result["timestamp"],
                "metadata": result["intermediate_results"]
            }

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing query: {e}")
            return {
                "success": False,
                "error": str(e),
                "output": f"❌ Error processing your request: {str(e)}",
                "route": "error",
                "processing_time": round(processing_time, 2)
            }


# Initialize the agent
agent = NonLinearAgent()


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/query', methods=['POST'])
def api_query():
    """API endpoint for processing queries"""
    data = request.get_json()
    query = data.get('query', '').strip()

    if not query:
        return jsonify({
            "success": False,
            "error": "Query cannot be empty"
        })

    result = agent.process_query(query)
    return jsonify(result)


@app.route('/api/status')
def api_status():
    """API endpoint for checking agent status"""
    return jsonify({
        "initialized": agent.is_initialized,
        "model": agent.model_name,
        "error": agent.error_message
    })


@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    emit('status', {
        'connected': True,
        'agent_ready': agent.is_initialized,
        'model': agent.model_name
    })


@socketio.on('query')
def handle_query(data):
    """Handle real-time query via WebSocket"""
    query = data.get('query', '').strip()

    if not query:
        emit('error', {'message': 'Query cannot be empty'})
        return

    # Emit processing status
    emit('processing', {'status': 'Processing your query...'})

    # Process the query
    result = agent.process_query(query)

    # Emit the result
    emit('result', result)


if __name__ == '__main__':
    print("🚀 Starting Flask Web Application for LangGraph Agent...")
    print("📱 Access the web interface at: http://localhost:5000")
    print("🔧 Make sure Ollama is running with Mistral model loaded")

    # Run the Flask app
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)