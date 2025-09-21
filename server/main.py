import os
import time
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from fastapi.responses import StreamingResponse
import json
import uuid

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG


class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol, e.g., NVDA")
    date: Optional[str] = Field(None, description="Trade date YYYY-MM-DD; default: today")
    provider: Optional[str] = Field(None, description="LLM provider override (openai|google|anthropic|ollama|openrouter)")
    deep_model: Optional[str] = Field(None, description="Deep thinking LLM model name")
    quick_model: Optional[str] = Field(None, description="Quick thinking LLM model name")
    backend_url: Optional[str] = Field(None, description="Optional custom backend URL for the provider")
    online_tools: Optional[bool] = Field(None, description="Toggle online tool usage")


class AnalyzeResponse(BaseModel):
    ticker: str
    date: str
    decision: str
    decision_processed: str
    reports: Dict[str, Any]
    conv_id: str


class ChatRequest(BaseModel):
    conv_id: str
    message: str


class ChatResponse(BaseModel):
    conv_id: str
    reply: str
    agent: Optional[str] = None


def detect_provider_from_env() -> Optional[str]:
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("GOOGLE_API_KEY"):
        return "google"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    # OpenRouter or Ollama custom backends could be inferred via URLs/keys
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.getenv("OLLAMA_BASE_URL"):
        return "ollama"
    return None


def build_config(req: AnalyzeRequest) -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG.copy()
    if req.provider:
        cfg["llm_provider"] = req.provider
    else:
        detected = detect_provider_from_env()
        if detected:
            cfg["llm_provider"] = detected
    if req.deep_model:
        cfg["deep_think_llm"] = req.deep_model
    if req.quick_model:
        cfg["quick_think_llm"] = req.quick_model
    if req.backend_url:
        cfg["backend_url"] = req.backend_url
    if req.online_tools is not None:
        cfg["online_tools"] = req.online_tools
    return cfg


app = FastAPI(title="TradingAgents API", version="0.1.0")
Conversations: Dict[str, Dict[str, Any]] = {}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    try:
        cfg = build_config(req)
        ta = TradingAgentsGraph(debug=False, config=cfg)
        date = req.date or "2024-05-10"
        final_state, processed = ta.propagate(req.ticker, date)

        reports = {
            "market_report": final_state.get("market_report"),
            "sentiment_report": final_state.get("sentiment_report"),
            "news_report": final_state.get("news_report"),
            "fundamentals_report": final_state.get("fundamentals_report"),
            "investment_plan": final_state.get("investment_plan"),
            "final_trade_decision": final_state.get("final_trade_decision"),
            "trader_investment_plan": final_state.get("trader_investment_plan"),
            "investment_debate_state": final_state.get("investment_debate_state"),
            "risk_debate_state": final_state.get("risk_debate_state"),
        }

        conv_id = uuid.uuid4().hex
        Conversations[conv_id] = {
            "cfg": cfg,
            "ticker": req.ticker,
            "date": date,
            "reports": reports,
            "history": [
                {"role": "system", "content": "TradingAgents conversation started."},
                {"role": "assistant", "content": f"Decision: {processed}"},
            ],
        }

        return AnalyzeResponse(
            ticker=req.ticker,
            date=date,
            decision=final_state.get("final_trade_decision", ""),
            decision_processed=processed,
            reports=reports,
            conv_id=conv_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Use a general-purpose logger; uvicorn.access expects a specific arg tuple
    logger = logging.getLogger("uvicorn.error")
    start = time.time()
    logger.info(f"--> {request.method} {request.url.path}")
    try:
        response = await call_next(request)
    except Exception as e:
        logger.exception(f"xxx {request.method} {request.url.path} error: {e}")
        raise
    finally:
        duration_ms = int((time.time() - start) * 1000)
        logger.info(f"<-- {request.method} {request.url.path} {duration_ms}ms")
    return response


@app.get("/")
def index():
    return {"ok": True, "msg": "TradingAgents API", "endpoints": ["/healthz", "/api/analyze", "/api/stream-analyze"]}


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/api/stream-analyze")
def stream_analyze(ticker: str, date: Optional[str] = None,
                   provider: Optional[str] = None,
                   deep_model: Optional[str] = None,
                   quick_model: Optional[str] = None,
                   backend_url: Optional[str] = None,
                   online_tools: Optional[bool] = None):
    req = AnalyzeRequest(
        ticker=ticker,
        date=date,
        provider=provider,
        deep_model=deep_model,
        quick_model=quick_model,
        backend_url=backend_url,
        online_tools=online_tools,
    )
    cfg = build_config(req)
    ta = TradingAgentsGraph(debug=False, config=cfg)

    init_state = ta.propagator.create_initial_state(ticker, date or "2024-05-10")
    args = ta.propagator.get_graph_args()
    conv_id = uuid.uuid4().hex
    Conversations[conv_id] = {
        "cfg": cfg,
        "ticker": ticker,
        "date": init_state["trade_date"],
        "reports": {},
        "history": [
            {"role": "system", "content": "TradingAgents conversation started (stream)."}
        ],
    }

    def gen():
        try:
            last_vals: Dict[str, Any] = {}
            yield f"data: {json.dumps({'type': 'start', 'ticker': ticker, 'date': init_state['trade_date'], 'conv_id': conv_id})}\n\n"

            last_chunk = None
            # Pylance type mismatch is fine at runtime; graph.stream accepts dict state
            for chunk in ta.graph.stream(init_state, **args):  # type: ignore
                last_chunk = chunk

                agent_for_key = {
                    "market_report": "Market Analyst",
                    "sentiment_report": "Social Media Analyst",
                    "news_report": "News Analyst",
                    "fundamentals_report": "Fundamentals Analyst",
                    "investment_plan": "Trader",
                    "trader_investment_plan": "Trader",
                    "final_trade_decision": "Risk Manager",
                }

                for key in [
                    "market_report",
                    "sentiment_report",
                    "news_report",
                    "fundamentals_report",
                    "investment_plan",
                    "trader_investment_plan",
                    "final_trade_decision",
                ]:
                    val = chunk.get(key)
                    if val and val != last_vals.get(key):
                        last_vals[key] = val
                        Conversations[conv_id]["reports"][key] = val
                        payload = {"type": key, "value": val, "agent": agent_for_key.get(key)}
                        yield f"data: {json.dumps(payload)}\n\n"

                msgs = chunk.get("messages")
                if msgs and len(msgs) > 0:
                    last_msg = msgs[-1]
                    content = None
                    role = None
                    try:
                        content = getattr(last_msg, "content", None)
                        role = getattr(last_msg, "type", None) or getattr(last_msg, "role", None)
                        if not content and isinstance(last_msg, (list, tuple)) and len(last_msg) >= 2:
                            role, content = last_msg[0], last_msg[1]
                    except Exception:
                        pass
                    if content:
                        Conversations[conv_id]["history"].append({"role": role or "assistant", "content": str(content)})
                        payload = {"type": "message", "role": role or "assistant", "content": content}
                        yield f"data: {json.dumps(payload)}\n\n"

            if last_chunk and last_chunk.get("final_trade_decision"):
                processed = ta.process_signal(last_chunk["final_trade_decision"]) or ""
                payload = {
                    "type": "done",
                    "decision": last_chunk.get("final_trade_decision", ""),
                    "decision_processed": processed,
                    "agent": "Risk Manager",
                }
                Conversations[conv_id]["history"].append({"role": "assistant", "content": f"Decision: {processed}"})
                yield f"data: {json.dumps(payload)}\n\n"

            yield "event: end\ndata: {}\n\n"
        except Exception as e:
            err = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(err)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def _build_quick_llm(cfg: Dict[str, Any]):
    provider = (cfg.get("llm_provider") or "openai").lower()
    if provider in ("openai", "ollama", "openrouter"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=cfg["quick_think_llm"], base_url=cfg.get("backend_url"))
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=cfg["quick_think_llm"], base_url=cfg.get("backend_url"))  # type: ignore
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=cfg["quick_think_llm"])
    raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    conv = Conversations.get(req.conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    cfg = conv["cfg"]
    llm = _build_quick_llm(cfg)

    # Build context prompt from reports
    reports = conv.get("reports", {})
    context_parts: List[str] = []
    for k in ["market_report", "sentiment_report", "news_report", "fundamentals_report", "investment_plan", "final_trade_decision"]:
        v = reports.get(k)
        if v:
            context_parts.append(f"[{k}]\n{v}\n")
    context = "\n".join(context_parts) or "No prior reports."

    messages = [
        ("system", "You are a concise trading research assistant. Use the provided context when answering follow-up questions. If a question is unrelated to the context, say so."),
        ("system", f"Context for ticker {conv['ticker']} on {conv['date']}:\n{context}"),
        ("human", req.message),
    ]

    try:
        resp = llm.invoke(messages)
        text = getattr(resp, "content", str(resp))
        conv["history"].append({"role": "user", "content": req.message})
        conv["history"].append({"role": "assistant", "content": text})
        return ChatResponse(conv_id=req.conv_id, reply=text, agent="Assistant")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
