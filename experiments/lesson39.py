from app.settings import SUMMARY_PROMPT_VERSION
from app.services.prompt_registry import get_active_summary_prompt_builder


text = "Python is a programming language."

prompt_builder = get_active_summary_prompt_builder()
prompt = prompt_builder(text)

print("SUMMARY_PROMPT_VERSION =", SUMMARY_PROMPT_VERSION)
print()
print(prompt)