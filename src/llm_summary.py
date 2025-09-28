# src/llm_summary.py
import openai
import pandas as pd

openai.api_key = "YOUR_OPENAI_API_KEY"

def generate_summary(df_pred: pd.DataFrame) -> str:
    summary_prompt = f"""
    Given the following appliance ON/OFF data for a day:
    {df_pred.head(50).to_dict()}
    Generate a concise summary in plain English of appliance usage percentages.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content
