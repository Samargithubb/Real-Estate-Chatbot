import os
from dotenv import load_dotenv
import re
import json
import google.generativeai as genai
import pandas as pd


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

generation_config = {
  "temperature": 0.1,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

model = genai.GenerativeModel(model_name="gemini-2.0-flash",
                              generation_config=generation_config)

ELIGIBLE_POSTCODES_FILE = "uk_postcodes 1.csv"
ELIGIBLE_POSTCODES = set()

def load_postcodes():
    """
    Loads eligible postcodes from the CSV file using pandas.
    """
    global ELIGIBLE_POSTCODES
    ELIGIBLE_POSTCODES.clear() 

    try:
        df = pd.read_csv(ELIGIBLE_POSTCODES_FILE, encoding='utf-8')

        postcodes_series = df['Postcode'].dropna().astype(str).str.strip().str.upper()
        
        ELIGIBLE_POSTCODES.update(pc for pc in postcodes_series if pc)

    except Exception as e:
        print(f"An unexpected error occurred while loading postcodes using pandas from '{ELIGIBLE_POSTCODES_FILE}': {e}")

def is_postcode_covered(postcode: str) -> bool:
    """Checks if the given postcode is in the eligible list."""
    return postcode.strip().upper() in ELIGIBLE_POSTCODES

def _clean_llm_json_response(text: str) -> str:
    """Cleans potential markdown and leading/trailing whitespace from LLM JSON response."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def get_intent_from_llm(user_input: str) -> str:
    prompt = (
        "You are an intent classification assistant. Your task is to determine if the user wants to 'buy' or 'sell' a home based on their message. "
        "You MUST respond with a single JSON object. Do not include any other text, explanations, or conversational filler before or after the JSON object. "
        "The JSON object should have a single key 'intent' with one of the following string values: 'buy', 'sell', or 'unknown'.\n\n"
        f"User message: '{user_input}'\n\n"
        "JSON Response:"
    )
    try:
        response = model.generate_content(prompt)
        cleaned_text = _clean_llm_json_response(response.text)
        data = json.loads(cleaned_text)
        return data.get('intent', 'unknown')
    except Exception as e: 
        print(f"Unexpected error in get_intent_from_llm (Google SDK): {e}. Response: '{getattr(response, 'text', 'N/A')}'")

    print("Falling back to keyword matching for intent.")
    if "buy" in user_input.lower(): return "buy"
    if "sell" in user_input.lower() or "selling" in user_input.lower(): return "sell"
    return "unknown"


def get_home_type_from_llm(user_input: str) -> str:
    prompt = (
        "You are an assistant helping to classify home type interest. Determine if the user is interested in a 'new' home or a 'resale' (existing/second-hand) home. "
        "You MUST respond with a single JSON object. Do not include any other text before or after it. "
        "The JSON object should have a single key 'home_type' with one of the following string values: 'new', 'resale', or 'unknown'.\n\n"
        f"User message: '{user_input}'\n\n"
        "JSON Response:"
    )
    try:
        response = model.generate_content(prompt)
        cleaned_text = _clean_llm_json_response(response.text)
        data = json.loads(cleaned_text)
        return data.get('home_type', 'unknown')
    except Exception as e:
        print(f"Unexpected error in get_home_type_from_llm (Google SDK): {e}. Response: '{getattr(response, 'text', 'N/A')}'")

    print("Falling back to keyword matching for home type.")
    if "new" in user_input.lower(): return "new"
    if "re-sale" in user_input.lower() or "resale" in user_input.lower() or "existing" in user_input.lower(): return "resale"
    return "unknown"

def get_yes_no_from_llm(user_input: str) -> str:
    prompt = (
        "You are an assistant determining if a user's response is affirmative (yes) or negative (no). "
        "You MUST respond with a single JSON object. Do not include any other text before or after it. "
        "The JSON object should have a single key 'answer' with one of the following string values: 'yes', 'no', or 'unknown'.\n\n"
        f"User message: '{user_input}'\n\n"
        "JSON Response:"
    )
    try:
        response = model.generate_content(prompt)
        cleaned_text = _clean_llm_json_response(response.text)
        data = json.loads(cleaned_text)
        return data.get('answer', 'unknown')
    except Exception as e:
        print(f"Unexpected error in get_yes_no_from_llm (Google SDK): {e}. Response: '{getattr(response, 'text', 'N/A')}'")

    if user_input.lower().strip() in ["yes", "y", "yeah", "yep", "sure", "ok", "affirmative"]: return "yes"
    if user_input.lower().strip() in ["no", "n", "nope", "nah", "negative"]: return "no"
    return "unknown"

def extract_details_from_llm(user_input: str, detail_type: str) -> str | None:
    instruction = ""
    if detail_type == "name":
        instruction = "Extract the person's name from the following text. If no name is found, respond with the literal string 'None'."
    elif detail_type == "phone":
        instruction = "Extract the phone number from the following text. If no phone number is found, respond with the literal string 'None'."
    elif detail_type == "email":
        instruction = "Extract the email address from the following text. If no email address is found, respond with the literal string 'None'."
    elif detail_type == "postcode":
        instruction = "Extract the UK postcode from the following text. A UK postcode has a format like SW1A 1AA or M1 1AE. If no postcode is found, respond with the literal string 'None'."
    elif detail_type == "budget":
        instruction = "Extract the budget amount as a number (digits only, or using 'k' for thousands, 'm' for millions) from the following text. If no numerical budget is found, respond with the literal string 'None'."

    prompt = (
        f"{instruction} Your response should be ONLY the extracted information or the exact string 'None'. Do not add any other commentary or markdown.\n\n"
        f"User input: '{user_input}'\n\n"
        "Extracted Information:"
    )
    try:
        response = model.generate_content(prompt)
        extracted = response.text.strip()
        if extracted.lower() == 'none' or extracted == '"None"':
            return None
        return extracted if extracted else None
    except Exception as e:
        print(f"LLM extraction error for {detail_type} (Google SDK): {e}. Response: '{getattr(response, 'text', 'N/A')}'. Returning None.")
        return None


def parse_budget(budget_str: str | None) -> int | None:
    if not budget_str:
        return None
    cleaned_budget_str = re.sub(r'[£€$,\s]', '', budget_str)
    try:
        if 'k' in cleaned_budget_str.lower():
            return int(float(cleaned_budget_str.lower().replace('k', '')) * 1000)
        if 'm' in cleaned_budget_str.lower():
            return int(float(cleaned_budget_str.lower().replace('m', '')) * 1000000)
        return int(cleaned_budget_str)
    except ValueError:
        return None


class RealEstateChatbot:
    def __init__(self):
        self.state = "START"
        self.context = {}

    def greet(self):
        print("Bot: How may I help you?")
        self.state = "AWAIT_INITIAL_INTENT"

    def reset_for_reassist(self):
        print("-" * 30)
        print("Bot: Resetting conversation flow...")
        self.state = "START"
        self.context = {}
        self.greet()

    def handle_input(self, user_input: str):
        if self.state == "AWAIT_INITIAL_INTENT":
            intent = get_intent_from_llm(user_input) 
            if intent == "buy":
                self.context['intent'] = "buy"
                print("Bot: Check buy requirement. Are you looking for a new home or a re-sale home?")
                self.state = "AWAIT_HOME_TYPE"
            elif intent == "sell":
                self.context['intent'] = "sell"
                self.state = "SELL_ASK_NAME"
            else:
                print("Bot: I'm sorry, I didn't understand if you want to buy or sell. Could you please clarify? (e.g., 'I want to buy a house')")

        elif self.state == "AWAIT_HOME_TYPE":
            home_type = get_home_type_from_llm(user_input)
            if home_type == "new":
                self.context['home_type'] = "new"
                self.state = "BUY_NEW_ASK_NAME"
            elif home_type == "resale":
                self.context['home_type'] = "resale"
                self.state = "BUY_RESALE_ASK_NAME"
            else:
                print("Bot: Sorry, I didn't catch if that was a new home or a re-sale home. Please specify 'new' or 're-sale'.")

        elif self.state == "BUY_NEW_ASK_NAME":
            self._collect_detail("name", user_input, "BUY_NEW_ASK_PHONE")
        elif self.state == "BUY_NEW_ASK_PHONE":
            self._collect_detail("phone", user_input, "BUY_NEW_ASK_EMAIL")
        elif self.state == "BUY_NEW_ASK_EMAIL":
            self._collect_detail("email", user_input, "BUY_NEW_ASK_BUDGET")
        elif self.state == "BUY_NEW_ASK_BUDGET":
            self._collect_budget(user_input, "BUY_NEW_CHECK_BUDGET_TRIGGER")

        elif self.state == "BUY_RESALE_ASK_NAME":
            self._collect_detail("name", user_input, "BUY_RESALE_ASK_PHONE")
        elif self.state == "BUY_RESALE_ASK_PHONE":
            self._collect_detail("phone", user_input, "BUY_RESALE_ASK_EMAIL")
        elif self.state == "BUY_RESALE_ASK_EMAIL":
            self._collect_detail("email", user_input, "BUY_RESALE_ASK_BUDGET")
        elif self.state == "BUY_RESALE_ASK_BUDGET":
            self._collect_budget(user_input, "ASSURE_CALLBACK_RESALE_TRIGGER")

        elif self.state == "SELL_ASK_NAME":
            self._collect_detail("name", user_input, "SELL_ASK_PHONE")
        elif self.state == "SELL_ASK_PHONE":
            self._collect_detail("phone", user_input, "SELL_ASK_EMAIL")
        elif self.state == "SELL_ASK_EMAIL":
            self._collect_detail("email", user_input, "SELL_ASK_POSTCODE")
        elif self.state == "SELL_ASK_POSTCODE":
            self._collect_postcode(user_input, "SELL_CHECK_POSTCODE_TRIGGER")

        elif self.state == "BUY_NEW_CHECK_BUDGET_TRIGGER":
            self.state = "BUY_NEW_CHECK_BUDGET"
            self._handle_buy_new_check_budget() 
        elif self.state == "BUY_NEW_CHECK_POSTCODE_COVERAGE_TRIGGER":
            self.state = "BUY_NEW_CHECK_POSTCODE_COVERAGE"
            self._handle_buy_new_check_postcode_coverage()
        elif self.state == "SELL_CHECK_POSTCODE_TRIGGER":
            self.state = "SELL_CHECK_POSTCODE"
            self._handle_sell_check_postcode()
        elif self.state == "ASSURE_CALLBACK_RESALE_TRIGGER":
            self.state = "ASSURE_CALLBACK_RESALE"
            self._assure_callback_and_reassist()


        elif self.state == "BUY_NEW_ASK_POSTCODE":
            self._collect_postcode(user_input, "BUY_NEW_CHECK_POSTCODE_COVERAGE_TRIGGER")

        elif self.state.endswith("_AWAIT_REASSIST"): 
            self._assure_callback_and_reassist_response(user_input)

        elif self.state == "END":
            pass

    def _ask_next_detail(self) -> str:
        if "ASK_NAME" in self.state: return "Can I get your name?"
        if "ASK_PHONE" in self.state: return "Can I get your phone number?"
        if "ASK_EMAIL" in self.state: return "Can I get your email address?"
        if "ASK_BUDGET" in self.state: return "What is your budget?"
        if "ASK_POSTCODE" in self.state:
            if self.context.get('intent') == 'sell':
                return "What is your postcode (of the property you want to sell)?"
            else:
                return "What is the postcode of your location of interest?"
        return ""

    def _collect_detail(self, detail_type: str, user_input: str, next_state: str):
        extracted_value = extract_details_from_llm(user_input, detail_type)
        if extracted_value:
            self.context[detail_type] = extracted_value
            print(f"Bot: Got it ({detail_type}: {extracted_value}).")
            self.state = next_state
        else:
            print(f"Bot: I'm sorry, I couldn't understand your {detail_type}. Could you please provide it again?")

    def _collect_budget(self, user_input: str, trigger_next_state: str):
        raw_budget_str = extract_details_from_llm(user_input, "budget")
        budget_val = parse_budget(raw_budget_str)
        if budget_val is not None:
            self.context['budget'] = budget_val
            print(f"Bot: Budget noted: {budget_val}.")
            self.state = trigger_next_state
            self.handle_input("")
        else:
            print("Bot: I'm sorry, I couldn't understand the budget. Please provide a numerical value (e.g., 1500000, 1.5m, 200k).")

    def _collect_postcode(self, user_input: str, trigger_next_state: str):
        postcode = extract_details_from_llm(user_input, "postcode") 
        if postcode:
            if re.match(r"^[A-Z]{1,2}[0-9][A-Z0-9]? ?[0-9][A-Z]{2}$", postcode.upper().strip()):
                self.context['postcode'] = postcode.upper().strip()
                print(f"Bot: Postcode noted: {self.context['postcode']}.")
                self.state = trigger_next_state
                self.handle_input("")
            else:
                print("Bot: That doesn't look like a valid UK postcode format (e.g., SW1A 1AA). Could you please provide it again?")
        else:
            print("Bot: I'm sorry, I couldn't get the postcode. Could you please provide it again?")


    def _handle_buy_new_check_budget(self):
        budget = self.context.get('budget')
        if budget is None:
             print("Bot: We need your budget first. What is your budget?")
             self.state = "BUY_NEW_ASK_BUDGET"
             return

        if budget < 1000000:
            print("Bot: Sorry, we don't cater to any properties under 1 million. "
                  "Please call the office on 1800 111 222 to get help. Thankyou for chatting with us. Goodbye.")
            self.state = "END"
        else:
            print("Bot: Great. Your budget is suitable.")
            self.state = "BUY_NEW_ASK_POSTCODE"

    def _handle_buy_new_check_postcode_coverage(self):
        postcode = self.context.get('postcode')
        if not postcode:
            print("Bot: Can I know the Postcode of your location of interest?")
            self.state = "BUY_NEW_ASK_POSTCODE"
            return

        if is_postcode_covered(postcode):
            self.state = "ASSURE_CALLBACK_BUY_NEW"
            self._assure_callback_and_reassist()
        else:
            print(f"Bot: Sorry, we don't cater to Post codes like {postcode} that you provided. "
                  "Please call the office on 1800 111 222 to get help.")
            print("Bot: Thank you for chatting with us. Goodbye.")
            self.state = "END"

    def _handle_sell_check_postcode(self):
        postcode = self.context.get('postcode')
        if not postcode:
            print("Bot: We need the postcode of the property you want to sell. What is the postcode?")
            self.state = "SELL_ASK_POSTCODE"
            return

        if is_postcode_covered(postcode):
            self.state = "ASSURE_CALLBACK_SELL"
            self._assure_callback_and_reassist()
        else:
            print(f"Bot: Sorry, we don't cater to Post codes like {postcode} that you provided. "
                  "Please call the office on 1800 111 222 to get help.")
            print("Bot: Thank you for chatting with us. Goodbye.")
            self.state = "END"

    def _assure_callback_and_reassist(self):
        current_intent_path = self.state 
        print("Bot: I can expect someone will get in touch with you within 24 hours via phone or email.")
        print("Bot: Is there anything else I can help you with? (yes/no)")
        self.state = current_intent_path + "_AWAIT_REASSIST" 

    def _assure_callback_and_reassist_response(self, user_input: str):
        answer = get_yes_no_from_llm(user_input)
        if answer == "yes":
            self.reset_for_reassist()
        elif answer == "no":
            print("Bot: Goodbye. Thank you for chatting with us. Good bye.")
            self.state = "END"
        else:
            print("Bot: Sorry, I didn't catch that. Is there anything else I can help you with? (Please say yes or no)")


    def run(self):
        load_postcodes()
        self.greet()
        while True:
            if self.state != "AWAIT_INITIAL_INTENT" and \
               not self.state.endswith("_AWAIT_REASSIST") and \
               not self.state.endswith("_TRIGGER") and \
               self.state not in ["END", "BUY_NEW_CHECK_BUDGET", "BUY_NEW_CHECK_POSTCODE_COVERAGE", "SELL_CHECK_POSTCODE", "ASSURE_CALLBACK_RESALE"]:
                current_question = self._ask_next_detail()
                if current_question:
                    print(f"Bot: {current_question}")

            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit']:
                print("Bot: Goodbye. Thank you for chatting with us. Good bye.")
                break
            if user_input.lower() == 'restart':
                self.reset_for_reassist()
                continue

            if self.state == "END":
                 print("Bot: (Session ended. Type 'restart' to begin a new conversation or 'quit' to exit.)")
                 continue

            self.handle_input(user_input)

            if self.state == "END" and user_input.lower() not in ['quit', 'exit', 'restart']:
                print("Bot: (Session ended. Type 'restart' to begin a new conversation or 'quit' to exit.)")


if __name__ == "__main__":
    print("Starting Real Estate Chatbot")
    print("Type 'quit' or 'exit' to stop, 'restart' to start over.")
    print("-" * 30)
    bot = RealEstateChatbot()
    bot.run()