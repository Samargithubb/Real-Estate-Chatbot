import os
import re
from typing import TypedDict, Annotated, Optional, Literal
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END, START

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

GEMINI_MODEL_NAME = "gemini-2.0-flash" 
MAX_REASSIST_LOOPS = 3
ELIGIBLE_POSTCODES_FILE = "uk_postcodes 1.csv"

ELIGIBLE_POSTCODES = set()

def load_postcodes(filename=ELIGIBLE_POSTCODES_FILE):
    global ELIGIBLE_POSTCODES
    ELIGIBLE_POSTCODES.clear() 

    try:
        df = pd.read_csv(filename, encoding='utf-8')

        for row in df.itertuples(index=False):
            if hasattr(row, 'Postcode') and pd.notna(row.Postcode):
                formatted_postcode = row.Postcode.strip().upper().replace(" ", "")
                ELIGIBLE_POSTCODES.add(formatted_postcode)

    except Exception as e:
        print(f"An unexpected error occurred while loading postcodes using pandas from '{ELIGIBLE_POSTCODES_FILE}': {e}")

def is_postcode_covered(postcode: Optional[str]) -> bool:
    if not postcode:
        return False
    return postcode.strip().upper().replace(" ", "") in ELIGIBLE_POSTCODES

class IntentResponse(BaseModel):
    intent: Literal["buy", "sell", "unknown"] = Field(description="User's intent, either 'buy' or 'sell'.")

class BuyTypeResponse(BaseModel):
    buy_type: Literal["new home", "re-sale home", "unknown"] = Field(description="Type of home user wants: 'new home' or 're-sale home'.")

class NameResponse(BaseModel):
    name: Optional[str] = Field(description="The user's name. Null if not provided.")

class PhoneResponse(BaseModel):
    phone_number: Optional[str] = Field(description="The user's phone number. Null if not provided or invalid format.")

class EmailResponse(BaseModel):
    email_address: Optional[str] = Field(description="The user's email address. Null if not provided or invalid format.")

class BudgetResponse(BaseModel):
    budget: Optional[float] = Field(description="The user's budget as a numerical value. Null if not a number.")
    error: Optional[str] = Field(description="Error message if budget cannot be extracted as a number.")

class PostcodeResponse(BaseModel):
    postcode: Optional[str] = Field(description="The user's postcode. Null if not provided.")

class YesNoResponse(BaseModel):
    answer: Literal["yes", "no", "unknown"] = Field(description="User's response, either 'yes' or 'no'.")


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]
    user_raw_input: Optional[str]

    intent: Optional[Literal["buy", "sell"]]
    buy_type: Optional[Literal["new home", "re-sale home"]]
    
    name: Optional[str]
    phone: Optional[str]
    email: Optional[str]
    budget: Optional[float]
    postcode: Optional[str]
    
    postcode_covered: Optional[bool]
    reassist_choice: Optional[Literal["yes", "no"]]
    
    loop_count: int
    final_bot_message: Optional[str]


llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL_NAME,
    temperature=0,
    convert_system_message_to_human=True,
    api_key=GOOGLE_API_KEY
)

def get_user_input(prompt_to_user: str) -> str:
    print(f"Bot: {prompt_to_user}")
    user_input = input("You: ")
    return user_input

def parse_with_llm(system_prompt: str, user_input: str, pydantic_model: type[BaseModel]):
    messages = [
        SystemMessage(content=system_prompt), 
        HumanMessage(content=f"User's response: {user_input}")
    ]
    try:
        structured_llm = llm.with_structured_output(pydantic_model)
        response = structured_llm.invoke(messages)
        print("---------------------",response, "----------------------")
        return response
    except Exception as e:
        return pydantic_model()


def node_start_conversation(state: ChatState) -> ChatState:
    user_input = get_user_input("How may I help you?")
    return {
        "messages": [AIMessage(content="How may I help you?"), HumanMessage(content=user_input)],
        "user_raw_input": user_input,
    }

def node_determine_intent(state: ChatState) -> ChatState:
    prompt = (
        "You are a helpful assistant. Your task is to determine if the user wants to 'buy' or 'sell' a home "
        "based on their response. Respond only with the determined intent."
    )
    response = parse_with_llm(prompt, state["user_raw_input"], IntentResponse)
    intent = response.intent if response.intent != "unknown" else None
    if not intent:
        raw_input_lower = state["user_raw_input"].lower()
        if "buy" in raw_input_lower or "purchase" in raw_input_lower:
            intent = "buy"
        elif "sell" in raw_input_lower or "list" in raw_input_lower:
            intent = "sell"
    return {"intent": intent}

def node_ask_buy_type(state: ChatState) -> ChatState:
    user_input = get_user_input("Are you looking for a new home or a re-sale home?")
    return {
        "messages": [AIMessage(content="Are you looking for a new home or a re-sale home?"), HumanMessage(content=user_input)],
        "user_raw_input": user_input,
    }

def node_determine_buy_type(state: ChatState) -> ChatState:
    prompt = (
        "The user was asked if they want a 'new home' or 're-sale home'. "
        "Determine their choice from their response. Respond only with the determined type."
    )
    response = parse_with_llm(prompt, state["user_raw_input"], BuyTypeResponse)
    buy_type = response.buy_type if response.buy_type != "unknown" else None
    if not buy_type:
        raw_input_lower = state["user_raw_input"].lower()
        if "new" in raw_input_lower:
            buy_type = "new home"
        elif "re-sale" in raw_input_lower or "resale" in raw_input_lower or "used" in raw_input_lower:
            buy_type = "re-sale home"
    return {"buy_type": buy_type}

def node_ask_name(state: ChatState) -> ChatState:
    user_input = get_user_input("Can I get your name?")
    name = user_input.strip() if user_input.strip() else None
    return {
        "messages": [AIMessage(content="Can I get your name?"), HumanMessage(content=user_input)],
        "name": name,
    }

def node_ask_phone(state: ChatState) -> ChatState:
    user_input = get_user_input("Can I get your phone number?")
    phone_match = re.search(r'(\+?\d[\d\s-]{8,}\d)', user_input)
    phone = phone_match.group(1).replace(" ", "").replace("-","") if phone_match else None
    return {
        "messages": [AIMessage(content="Can I get your phone number?"), HumanMessage(content=user_input)],
        "phone": phone,
    }

def node_ask_email(state: ChatState) -> ChatState:
    user_input = get_user_input("Can I get your email address?")
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', user_input)
    email = email_match.group(0) if email_match else None
    return {
        "messages": [AIMessage(content="Can I get your email address?"), HumanMessage(content=user_input)],
        "email": email,
    }

def node_ask_budget(state: ChatState) -> ChatState:
    user_input = get_user_input("What is your budget?")
    budget_val = None
    try:
        cleaned_input = user_input.lower().replace(',', '').strip()
        multiplier = 1
        if 'm' in cleaned_input:
            multiplier = 1_000_000
            cleaned_input = cleaned_input.replace('m', '')
        elif 'k' in cleaned_input:
            multiplier = 1_000
            cleaned_input = cleaned_input.replace('k', '')
        
        numeric_part = "".join(filter(lambda x: x.isdigit() or x == '.', cleaned_input))
        if numeric_part:
            budget_val = float(numeric_part) * multiplier
    except ValueError:
        budget_val = None
        
    return {
        "messages": [AIMessage(content="What is your budget?"), HumanMessage(content=user_input)],
        "budget": budget_val,
    }

def node_inform_low_budget_and_goodbye(state: ChatState) -> ChatState:
    message = "Sorry, we dont cater to any properties under 1 million. Please call the office on 1800 111 222 to get help. Thankyou for chatting with us. Goodbye"
    print(f"Bot: {message}")
    return {"messages": [AIMessage(content=message)], "final_bot_message": message}


def node_ask_postcode(state: ChatState) -> ChatState:
    prompt_text = "Can I know the postcode of your location of interest?" if state['intent'] == 'buy' else "What is your postcode?"
    user_input = get_user_input(prompt_text)
    return {
        "messages": [AIMessage(content=prompt_text), HumanMessage(content=user_input)],
        "postcode": user_input.strip().upper().replace(" ", ""),
    }

def node_check_postcode_coverage(state: ChatState) -> ChatState:
    covered = is_postcode_covered(state["postcode"])
    return {"postcode_covered": covered}

def node_inform_postcode_not_covered(state: ChatState) -> ChatState:
    message = "Sorry, we dont cater to Post codes that you provided. Please call the office on 1800 111 222 to get help"
    print(f"Bot: {message}")
    return {"messages": [AIMessage(content=message)]}

def node_assure_callback(state: ChatState) -> ChatState:
    callback_message = "I can expect someone to get in touch with you within 24 hours via phone or email."
    print(f"Bot: {callback_message}")
    return {"messages": [AIMessage(content=callback_message)]}

def node_offer_reassistance_prompt(state: ChatState) -> ChatState:
    user_input = get_user_input("Is there anything else I can help you with? (yes/no)")
    prompt = (
        "The user was asked if they need more help. Determine if their answer is 'yes' or 'no'. "
        "Respond only with 'yes' or 'no'."
    )
    response = parse_with_llm(prompt, user_input, YesNoResponse)
    reassist_choice = response.answer if response.answer != "unknown" else None
    if not reassist_choice:
        if "yes" in user_input.lower() or "yeah" in user_input.lower():
            reassist_choice = "yes"
        elif "no" in user_input.lower() or "nope" in user_input.lower():
            reassist_choice = "no"

    return {
        "messages": [AIMessage(content="Is there anything else I can help you with? (yes/no)"), HumanMessage(content=user_input)],
        "reassist_choice": reassist_choice,
    }

def node_prepare_for_reassist(state: ChatState) -> ChatState:
    return {
        "messages": [AIMessage(content="How can I help you?")],
        "user_raw_input": None,
        "intent": None,
        "buy_type": None,
        "name": None,
        "phone": None,
        "email": None,
        "budget": None,
        "postcode": None,
        "postcode_covered": None,
        "reassist_choice": None,
        "loop_count": state["loop_count"] + 1,
        "final_bot_message": None,
    }

def node_say_goodbye(state: ChatState) -> ChatState:
    message = "Thank you for chatting with us. Good bye"
    print(f"Bot: {message}")
    return {"messages": [AIMessage(content=message)], "final_bot_message": message}


def route_intent(state: ChatState):
    if state["intent"] == "buy":
        return "ask_buy_type"
    elif state["intent"] == "sell":
        return "ask_name_sell"
    else:
        print("Bot: Sorry, I didn't understand if you want to buy or sell. Let's try again.")
        return START

def route_buy_type(state: ChatState):
    if state["buy_type"] in ["new home", "re-sale home"]:
        return "ask_name_buy"
    else:
        print("Bot: Sorry, I didn't understand if you're looking for a new or re-sale home. Let's try that again.")
        return "ask_buy_type"

def route_after_email_buy(state: ChatState):
    return "ask_budget"

def route_after_email_sell(state: ChatState):
    return "ask_postcode_sell" 

def route_budget(state: ChatState):
    if state["budget"] is None:
        print("Bot: I couldn't understand your budget. Please provide a numerical value (e.g. 1500000 or 1.5m).")
        return "ask_budget" 

    if state["buy_type"] == "new home":
        if state["budget"] < 1_000_000:
            return "inform_low_budget_and_goodbye" 
        else:
            return "ask_postcode_buy" 
    elif state["buy_type"] == "re-sale home":
        return "ask_postcode_buy"
    else:
        print(f"Bot: Unexpected buy type ({state['buy_type']}) during budget routing. Proceeding as if for re-sale.")
        return "ask_postcode_buy"

def route_postcode_coverage(state: ChatState):
    if state["postcode_covered"]:
        return "assure_callback"
    else:
        return "inform_postcode_not_covered"

def route_reassistance(state: ChatState):
    if state["reassist_choice"] == "yes" and state["loop_count"] < MAX_REASSIST_LOOPS:
        return "prepare_for_reassist"
    elif state["reassist_choice"] == "yes" and state["loop_count"] >= MAX_REASSIST_LOOPS:
        print("Bot: I've tried to assist multiple times. For further help, please contact our support. Goodbye.")
        return "say_goodbye"
    elif state["reassist_choice"] == "no":
        return "say_goodbye"
    else:
        print("Bot: Sorry, I didn't catch that. Please say 'yes' or 'no'.")
        return "offer_reassistance_prompt"

load_postcodes()

workflow = StateGraph(ChatState)

workflow.add_node("node_start_conversation", node_start_conversation)
workflow.add_node("node_determine_intent", node_determine_intent)
workflow.add_node("node_ask_buy_type", node_ask_buy_type)
workflow.add_node("node_determine_buy_type", node_determine_buy_type)
workflow.add_node("node_ask_name_buy", node_ask_name)
workflow.add_node("node_ask_phone_buy", node_ask_phone)
workflow.add_node("node_ask_email_buy", node_ask_email)
workflow.add_node("node_ask_name_sell", node_ask_name)
workflow.add_node("node_ask_phone_sell", node_ask_phone)
workflow.add_node("node_ask_email_sell", node_ask_email)
workflow.add_node("node_ask_budget", node_ask_budget)
workflow.add_node("node_inform_low_budget_and_goodbye", node_inform_low_budget_and_goodbye)
workflow.add_node("node_ask_postcode_buy", node_ask_postcode)
workflow.add_node("node_ask_postcode_sell", node_ask_postcode)
workflow.add_node("node_check_postcode_coverage", node_check_postcode_coverage)
workflow.add_node("node_inform_postcode_not_covered", node_inform_postcode_not_covered)
workflow.add_node("node_assure_callback", node_assure_callback)
workflow.add_node("node_offer_reassistance_prompt", node_offer_reassistance_prompt)
workflow.add_node("node_prepare_for_reassist", node_prepare_for_reassist)
workflow.add_node("node_say_goodbye", node_say_goodbye)

workflow.set_entry_point("node_start_conversation")

workflow.add_edge("node_start_conversation", "node_determine_intent")
workflow.add_conditional_edges(
    "node_determine_intent", route_intent,
    {"ask_buy_type": "node_ask_buy_type", "ask_name_sell": "node_ask_name_sell", START: "node_start_conversation"}
)
workflow.add_edge("node_ask_buy_type", "node_determine_buy_type")
workflow.add_conditional_edges(
    "node_determine_buy_type", route_buy_type,
    {"ask_name_buy": "node_ask_name_buy", "ask_buy_type": "node_ask_buy_type"}
)
workflow.add_edge("node_ask_name_buy", "node_ask_phone_buy")
workflow.add_edge("node_ask_phone_buy", "node_ask_email_buy")
workflow.add_conditional_edges("node_ask_email_buy", route_after_email_buy, {"ask_budget": "node_ask_budget"})
workflow.add_conditional_edges(
    "node_ask_budget", route_budget,
    {"inform_low_budget_and_goodbye": "node_inform_low_budget_and_goodbye", "ask_postcode_buy": "node_ask_postcode_buy", "ask_budget": "node_ask_budget"}
)
workflow.add_edge("node_inform_low_budget_and_goodbye", END)
workflow.add_edge("node_ask_postcode_buy", "node_check_postcode_coverage")
workflow.add_edge("node_ask_name_sell", "node_ask_phone_sell")
workflow.add_edge("node_ask_phone_sell", "node_ask_email_sell")
workflow.add_conditional_edges("node_ask_email_sell", route_after_email_sell, {"ask_postcode_sell": "node_ask_postcode_sell"})
workflow.add_edge("node_ask_postcode_sell", "node_check_postcode_coverage")
workflow.add_conditional_edges(
    "node_check_postcode_coverage", route_postcode_coverage,
    {"assure_callback": "node_assure_callback", "inform_postcode_not_covered": "node_inform_postcode_not_covered"}
)
workflow.add_edge("node_assure_callback", "node_offer_reassistance_prompt")
workflow.add_edge("node_inform_postcode_not_covered", "node_offer_reassistance_prompt")
workflow.add_conditional_edges(
    "node_offer_reassistance_prompt", route_reassistance,
    {"prepare_for_reassist": "node_prepare_for_reassist", "say_goodbye": "node_say_goodbye", "offer_reassistance_prompt": "node_offer_reassistance_prompt"}
)
workflow.add_edge("node_prepare_for_reassist", "node_start_conversation")
workflow.add_edge("node_say_goodbye", END)

app = workflow.compile()

if __name__ == "__main__":
    print("Chatbot initialized Type 'exit' or 'quit' to end.")
    initial_state = ChatState(
        messages=[], user_raw_input=None, intent=None, buy_type=None, name=None,
        phone=None, email=None, budget=None, postcode=None, postcode_covered=None,
        reassist_choice=None, loop_count=0, final_bot_message=None
    )
    
    try:
        final_state = app.invoke(initial_state, {"recursion_limit": 200})
        if final_state.get("final_bot_message") and not any(m.content == final_state["final_bot_message"] for m in final_state.get("messages", [])):
            pass
        print("\nConversation ended.")
    except Exception as e:
        print(f"\nAn error occurred during the conversation: {e}")
        import traceback
        traceback.print_exc()