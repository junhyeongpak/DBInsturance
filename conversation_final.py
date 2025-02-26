import openai
import json
import langgraph
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import os
import re
import random
import pandas as pd
from dotenv import load_dotenv
import graphviz

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


json_path = r"./10K_reports_summary.json"
file_path = "./dow_30_news.csv"
news_df = pd.read_csv(file_path)
financial_df = pd.read_csv("./All_Financial_Ratios.csv")

dow30_ticker_mapping = {
        "Apple": "AAPL", "Amgen": "AMGN", "Amazon": "AMZN", "Cisco": "CSCO", "Microsoft": "MSFT", "NVIDIA": "NVDA",
        "American Express": "AXP", "Boeing": "BA", "Caterpillar": "CAT", "Salesforce": "CRM", "Chevron": "CVX",
        "Disney": "DIS", "Goldman Sachs": "GS", "Home Depot": "HD", "Honeywell": "HON", "IBM": "IBM",
        "Johnson & Johnson": "JNJ", "JPMorgan Chase": "JPM", "Coca-Cola": "KO", "McDonald's": "MCD", "3M": "MMM",
        "Merck": "MRK", "Nike": "NKE", "Procter & Gamble": "PG", "Sherwin-Williams": "SHW", "Travelers": "TRV",
        "UnitedHealth": "UNH", "Visa": "V", "Verizon": "VZ", "Walmart": "WMT"
    }

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

def organize_data_by_year(data):
    company_data = {}
    for report in data:
        file_name = report["file_name"]
        year = "2023" if "2023" in file_name else "2024"
        company_name = file_name.replace("_2023", "").replace("_2024", "").replace(".pdf", "")

        if company_name not in company_data:
            company_data[company_name] = {}
        
        summary = report["summary"]
        
        if isinstance(summary, str):
            try:
                summary = json.loads(summary)
            except json.JSONDecodeError:
                summary = {}  
        
        company_data[company_name][year] = summary

    return company_data

company_data = organize_data_by_year(data)

# OpenAI LLM 설정 (변경 금지)
llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

class InvestmentState(TypedDict):
    views: str
    prev_views: str
    sentiment_views: str  
    financial_views: str  
    iteration: int


# **📌 투자 견해 초기 생성 함수 (JSON 데이터 기반)**
def generate_initial_views(state: InvestmentState):
    context = ""
    for company, years in company_data.items():
        context += f"\n📌 {company}\n"
        for year, summary in years.items():
            context += f"🔹 {year}년 데이터:\n"
            context += f"- **Business Overview**: {summary.get('Business Overview', 'N/A')}\n"
            context += f"- **Key Risk Factors**: {summary.get('Key Risk Factors', 'N/A')}\n"
            context += f"- **Financial Summary**: {summary.get('Financial Summary', 'N/A')}\n"
            context += f"- **Management Insights**: {summary.get('Management Insights', 'N/A')}\n"

    prompt = f"""
    You are a financial analyst specializing in Black-Litterman model-based investment insights.
    Based on the summarized 10-K reports from 2023 and 2024, generate **investment viewpoints** 
    using a comparative analysis of companies and their financial performance.

    **Company Data:**
    {context}

    **Instructions:**
    - Generate **investment viewpoints** that must **always include specific companies**.
    - The number of investment views generated cannot exceed the number of companies.
    - Each viewpoint **must contain an expected return percentage change (increase or decrease) and which company is**.
    - The expected return change **must be between 1% and 8% (rounded to one decimal place)., it can be negative**.
    - Ensure that each viewpoint **directly compares two companies** (e.g., "Microsoft vs. Google") or **focuses on a single company's expected return change**.
    - Use financial trends, key risks, and management insights to justify each viewpoint.
    - Avoid listing multiple stocks per viewpoint.
    - Let the number of single views and multiple views be drawn somewhat similar
    - Ensure that single-stock views and comparative views (A vs. B) do not include overlapping companies. Each company should only appear in either a single-stock view or a comparative view, not both
    
    **Example Format:**
    - **Apple** is expected to see a **+4.2%** return due to strong revenue growth and risk mitigation.
    - **Microsoft vs. Google**: Microsoft is expected to outperform Google by **+3.5%** due to AI investments.
    
    **Output Format:**
    - **Viewpoint 1**: ...
    - **Viewpoint 2**: ...
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    print("\n🔵 [초기 투자 견해 생성 완료] 🔵\n", response.content)
    
    return {
    "views": response.content,
    "prev_views": "",
    "sentiment_views": "", 
    "financial_views": "",  
    "iteration": 0
    }


def sentiment_analysis_agent(state: InvestmentState):

    mentioned_tickers = [ticker for company, ticker in dow30_ticker_mapping.items() if company in state["views"]]
    if not mentioned_tickers:
        print("⚠️ No relevant stocks found for sentiment analysis.")
        return {"sentiment_views": state["sentiment_views"]}  

    filtered_news_df = news_df[news_df["ticker"].isin(mentioned_tickers)].dropna(subset=["summary"])
    filtered_news_df["summary"] = filtered_news_df["summary"].fillna("").astype(str)

    if filtered_news_df.empty:
        print("⚠️ No relevant news articles found.")
        return {"sentiment_views": state["sentiment_views"]}  

    filtered_news_df = filtered_news_df.sort_values(by="datetime", ascending=False)

    summarized_news = {}
    for ticker in mentioned_tickers:
        subset = filtered_news_df[filtered_news_df["ticker"] == ticker]
        if subset.empty:
            continue  
        num_articles = random.randint(3, min(5, len(subset)))  
        selected_articles = subset.sample(n=num_articles, random_state=random.randint(1, 1000))["summary"].tolist()
        summarized_news[ticker] = " ".join(selected_articles)

    if not summarized_news:
        print("⚠️ No sampled news articles found.")
        return {"sentiment_views": state["sentiment_views"]}  
    sentiment_prompt = f"""
    Analyze the sentiment of the following news summaries and classify them as Positive, Negative, or Neutral.

    **News Summaries:**
    {summarized_news}
    """

    response = llm.invoke([HumanMessage(content=sentiment_prompt)])
    sentiment_results = response.content.strip()

    print("\n🟡 [Sentiment Analysis Results] 🟡")
    print(sentiment_results)

    return {"sentiment_views": sentiment_results}  

# def sentiment_analysis_agent(state: InvestmentState):
#     """Perform sentiment analysis on news data and update investment insights."""

#     dow30_ticker_mapping = {
#         "Apple": "AAPL", "Amgen": "AMGN", "Amazon": "AMZN", "Cisco": "CSCO", "Microsoft": "MSFT", "NVIDIA": "NVDA",
#         "American Express": "AXP", "Boeing": "BA", "Caterpillar": "CAT", "Salesforce": "CRM", "Chevron": "CVX",
#         "Disney": "DIS", "Goldman Sachs": "GS", "Home Depot": "HD", "Honeywell": "HON", "IBM": "IBM",
#         "Johnson & Johnson": "JNJ", "JPMorgan Chase": "JPM", "Coca-Cola": "KO", "McDonald's": "MCD", "3M": "MMM",
#         "Merck": "MRK", "Nike": "NKE", "Procter & Gamble": "PG", "Sherwin-Williams": "SHW", "Travelers": "TRV",
#         "UnitedHealth": "UNH", "Visa": "V", "Verizon": "VZ", "Walmart": "WMT"
#     }

#     mentioned_tickers = [ticker for company, ticker in dow30_ticker_mapping.items() if company in state["views"]]
#     if not mentioned_tickers:
#         print("⚠️ No relevant stocks found for sentiment analysis.")
#         return state

#     filtered_news_df = news_df[news_df["ticker"].isin(mentioned_tickers)].dropna(subset=["summary"])
#     filtered_news_df["summary"] = filtered_news_df["summary"].fillna("").astype(str)

#     if filtered_news_df.empty:
#         print("⚠️ No relevant news articles found.")
#         return state

#     filtered_news_df = filtered_news_df.sort_values(by="datetime", ascending=False)

#     summarized_news = {}
#     for ticker in mentioned_tickers:
#         subset = filtered_news_df[filtered_news_df["ticker"] == ticker]
#         if subset.empty:
#             continue  
#         num_articles = random.randint(3, min(5, len(subset)))  
#         selected_articles = subset.sample(n=num_articles, random_state=random.randint(1, 1000))["summary"].tolist()
#         summarized_news[ticker] = " ".join(selected_articles)

#     if not summarized_news:
#         print("⚠️ No sampled news articles found.")
#         return state

#     sentiment_prompt = f"""
#     Analyze the sentiment of the following news summaries and classify them as Positive, Negative, or Neutral.

#     **News Summaries:**
#     {summarized_news}
#     """

#     response = llm.invoke([HumanMessage(content=sentiment_prompt)])
#     sentiment_results = response.content.strip()

#     print("\n🟡 [Sentiment Analysis Results] 🟡")
#     print(sentiment_results)

#     adjustment_prompt = f"""
#     Modify the following investment viewpoints based on the sentiment analysis results.

#     **Current Investment Viewpoints:**
#     {state["views"]}

#     **Sentiment Analysis Results:**
#     {sentiment_results}

#     **Instructions:**
#     - Adjust expected return percentages based on sentiment.
#     - Ensure that the updated viewpoints remain within the 1% to 8% range.
#     """

#     adjusted_response = llm.invoke([HumanMessage(content=adjustment_prompt)])
#     adjusted_views = adjusted_response.content.strip()

#     print("\n🟡 [Updated Investment Views after Sentiment Analysis] 🟡")
#     print(adjusted_views)

#     return {
#         "sentiment_views": sentiment_results
#     }


def financial_analysis_agent(state: InvestmentState):

    viewpoints = state["views"].split("\n")
    adjusted_viewpoints = []

    for viewpoint in viewpoints:
        match = re.search(r'\*\*(.*?)\*\*.*?([+-]?\d+(?:\.\d+)?)%', viewpoint)
        if not match:
            adjusted_viewpoints.append(viewpoint)
            continue

        company, expected_return = match.groups()
        expected_return = float(expected_return)
        ticker = dow30_ticker_mapping.get(company)

        if not ticker:
            adjusted_viewpoints.append(viewpoint)
            continue

        latest_year = max(financial_df["연도"].astype(str).unique())
        company_financials = financial_df[(financial_df["종목명"] == ticker) & (financial_df["연도"].astype(str) == latest_year)]

        if company_financials.empty:
            adjusted_viewpoints.append(viewpoint)
            continue

        financial_metrics = company_financials.iloc[0].to_dict()

        prompt = f"""
        Given the financial indicators of {company}, adjust the expected return accordingly.

        **Financial Indicators:**
        Growth: {financial_metrics.get("총자산 증가율", "N/A")}, {financial_metrics.get("유형자산 증가율", "N/A")}, {financial_metrics.get("매출액 증가율", "N/A")}
        Profitability: {financial_metrics.get("매출액 영업이익률", "N/A")}, {financial_metrics.get("매출액 순이익률", "N/A")}
        Liquidity: {financial_metrics.get("유동비율", "N/A")}, {financial_metrics.get("당좌비율", "N/A")}
        Stability: {financial_metrics.get("부채비율", "N/A")}, {financial_metrics.get("유동부채비율", "N/A")}

        Adjust the expected return ({expected_return}%) based on the above indicators.

        Return the adjusted expected return as a single numerical value (rounded to one decimal place) followed by a percentage sign.
        """

        response = llm.invoke([HumanMessage(content=prompt)])
        financial_expected_return = response.content.strip()

        try:
            financial_expected_return = float(financial_expected_return.replace("%", ""))
            financial_expected_return = max(1, min(8, financial_expected_return))
        except ValueError:
            financial_expected_return = expected_return

        viewpoint = re.sub(r'([+-]?\d+(?:\.\d+)?)%', f"{financial_expected_return:.1f}%", viewpoint)
        adjusted_viewpoints.append(viewpoint)

    updated_financial_views = "\n".join(adjusted_viewpoints)

    print("\n🔵 [Financial Analysis Updated Views] 🔵")
    print(updated_financial_views)

    return {"financial_views": updated_financial_views}


# def update_views(state: InvestmentState):
    
#     prompt = f"""
#     You are a financial expert integrating sentiment analysis and financial data insights.
#     Based on the revised insights from sentiment analysis and financial metrics, revise the investment viewpoints.
    
#     **Previous Investment Viewpoints:**
#     {state["prev_views"]}
    
#     **Updated Sentiment Insights:**
#     {state["sentiment_views"]}
    
#     **Updated Financial Insights:**
#     {state["financial_views"]}
    
#     **Instructions:**
#     - Adjust expected return percentages if needed.
#     - If both sentiment and financial data indicate strong negative/positive signals, reflect them in the updates.
#     - Ensure updated viewpoints remain within the 1% to 8% range.
#     - 감성 분석과 재무 분석 에이전트의 견해를 반영해서 업데이트를 할 때, 제발 기존 견해의 개수랑 종목과 순서를 바꾸지마!

    
#     **Formatting Guidelines:**
#     - **For single-stock views**: 
#         - Format: "**[TICKER]** (Expected Return: [X]%) - Key reason: [Summary]"
#         - Example: "**AAPL** (Expected Return: +4.2%) - Strong revenue growth and stable financials."
#     - **For comparative views (A vs. B)**:
#         - Format: "**[TICKER1] vs. [TICKER2]** (Expected Outperformance: [X]%) - Key reason: [Summary]"
#         - Example: "**MSFT vs. AAPL** (Expected Outperformance: +3.5%) - Microsoft expected to outperform Apple due to AI investments and positive market sentiment."
    
#     **Final Output:**
#     - Summarize the updated viewpoints following the formatting above.
#     """

#     response = llm.invoke([HumanMessage(content=prompt)])
#     updated_views = response.content.strip()
    
#     print("\n🔵 [업데이트된 투자 견해] 🔵")
#     print(updated_views)
    
#     # ✅ 최종 결과만 출력
#     if state["iteration"] >= 4:
#         print("\n🟢 **최종 투자 견해 (Final Investment Views):** 🟢")
#         print(updated_views)

#     return {
#         "views": updated_views,
#         "prev_views": state["views"],
#         "iteration": state["iteration"] + 1
#     }

def update_views(state: InvestmentState):
    """LLM이 감성 분석과 재무 분석을 비교하여 조기 종료 여부를 결정"""

    prompt = f"""
    You are a financial expert evaluating investment viewpoints based on sentiment analysis and financial data insights.
    Your goal is to determine whether updates are necessary based on the alignment of sentiment and financial analysis.

    **Step 1: Independent Evaluation**
    - Sentiment analysis reflects how investors and the media perceive the company (based on news and social sentiment).
    - Financial analysis reflects the company's fundamental performance (based on earnings reports, growth metrics, and risk factors).

    **Step 2: Compare Both Analyses**
    - If sentiment and financial analysis align strongly (e.g., both suggest an increase or decrease in expected return), updates may not be needed.
    - If they conflict significantly (e.g., positive sentiment but weak financials, or vice versa), adjustments **must be made**.

    **Step 3: Expected Return Changes**
    - If the expected return change for most stocks is **less than 0.5%**, updates may not be necessary.
    - However, if sentiment analysis and financial insights contradict each other, **you must update the expected returns accordingly**.

    **Step 4: Decision Criteria**
    - If no further updates are needed, return "**STOP UPDATES**".
    - If updates are needed, return the revised investment viewpoints while ensuring the same structure and order.

    **Previous Investment Viewpoints:**
    {state["prev_views"]}

    **Sentiment Analysis Insights:**
    {state["sentiment_views"]}

    **Financial Analysis Insights:**
    {state["financial_views"]}

    **Instructions:**
    - If sentiment and financial analysis **do not align**, you **must adjust** the expected return percentages.
    - If updates are NOT needed, return "**STOP UPDATES**".
    - If updates are needed, return:
      - **[TICKER]** (Expected Return: [X]%) - Key reason: [Summary]
      - **[TICKER1] vs. [TICKER2]** (Expected Outperformance: [X]%) - Key reason: [Summary]
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    updated_views = response.content.strip()

    print("\n🔵 [업데이트된 투자 견해] 🔵")
    print(updated_views)

    # ✅ LLM이 STOP UPDATES를 반환하면 조기 종료
    if "STOP UPDATES" in updated_views:
        print("\n🟢 **LLM determined no further updates needed. Early termination.** 🟢")
        return {"views": state["prev_views"], "prev_views": state["views"], "iteration": 5}

    # ✅ 최종 결과만 출력
    if state["iteration"] >= 4:
        print("\n🟢 **최종 투자 견해 (Final Investment Views):** 🟢")
        print(updated_views)

    return {"views": updated_views, "prev_views": state["views"], "iteration": state["iteration"] + 1}




def check_convergence(state: InvestmentState):
    if state["iteration"] >= 5:
        print("\n🟢 **최대 반복 횟수 도달. 실행 종료.** 🟢")
        return END  
    return "analyze_sentiment_and_financials"  

def analyze_sentiment_and_financials(state: InvestmentState):
    """감성 분석과 재무 분석을 동시에 실행하는 노드"""
    sentiment_result = sentiment_analysis_agent(state)
    financial_result = financial_analysis_agent(state)

    return {
        "sentiment_views": sentiment_result["sentiment_views"],
        "financial_views": financial_result["financial_views"]
    }

# 📌 최종 결과 저장 함수
def save_results_to_json(result):
    """최종 투자 견해를 JSON 파일로 저장"""
    file_path = "C:/Users/Owner/Desktop/invest portfolio/final_result.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"\n✅ **Final investment views saved to {file_path}**")


def draw_langgraph():
    """LangGraph를 시각화하여 PNG 파일로 저장"""
    dot = graphviz.Digraph()

    dot.node("START", shape="ellipse", color="green")
    dot.node("Generate Views", shape="box", style="filled", fillcolor="lightblue")
    dot.node("Sentiment Analysis", shape="box", style="filled", fillcolor="lightyellow")
    dot.node("Financial Analysis", shape="box", style="filled", fillcolor="lightgray")
    dot.node("Update Views", shape="box", style="filled", fillcolor="lightcoral")
    dot.node("Check Convergence", shape="diamond", style="filled", fillcolor="white")
    dot.node("END", shape="ellipse", color="red")

    # 초기 투자 견해 생성 후 감성 분석 & 재무 분석 실행
    dot.edge("START", "Generate Views")
    dot.edge("Generate Views", "Sentiment Analysis")
    dot.edge("Generate Views", "Financial Analysis")

    # 감성 분석과 재무 분석 결과가 업데이트로 이동
    dot.edge("Sentiment Analysis", "Update Views")
    dot.edge("Financial Analysis", "Update Views")

    # 업데이트 후 종료 조건 평가
    dot.edge("Update Views", "Check Convergence")

    # 5회 반복 후 종료
    dot.edge("Check Convergence", "END", label="If iteration >= 5")

    # 반복이 필요한 경우 다시 감성 & 재무 분석 실행
    dot.edge("Check Convergence", "Sentiment Analysis", label="If iteration < 5")
    dot.edge("Check Convergence", "Financial Analysis")

    # 시각화 저장
    dot.render("C:/Users/Owner/Desktop/invest portfolio/langgraph", format="png", cleanup=True)
    print("\n✅ **LangGraph visualization updated and saved as 'langgraph.png'**")

workflow = StateGraph(InvestmentState)

workflow.add_node("generate_views", generate_initial_views)
workflow.add_node("analyze_sentiment_and_financials", analyze_sentiment_and_financials)
workflow.add_node("update_views", update_views)

workflow.add_edge(START, "generate_views")

workflow.add_edge("generate_views", "analyze_sentiment_and_financials")

workflow.add_edge("analyze_sentiment_and_financials", "update_views")

workflow.add_conditional_edges("update_views", check_convergence)

# **📌 그래프 실행**
app = workflow.compile()
result = app.invoke({
    "views": "",
    "prev_views": "",
    "sentiment_views": "",
    "financial_views": "",
    "iteration": 0
})

save_results_to_json(result)
draw_langgraph()