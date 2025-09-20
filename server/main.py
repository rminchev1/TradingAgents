import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

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

        return AnalyzeResponse(
            ticker=req.ticker,
            date=date,
            decision=final_state.get("final_trade_decision", ""),
            decision_processed=processed,
            reports=reports,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/healthz")
def healthz():
    return {"ok": True}
