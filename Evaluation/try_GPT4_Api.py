import pandas as pd
from openai import OpenAI


def generate_evaluation(df, model):
    predictions = []
    for _, row in df.iterrows():
        prompt = row["input"].split("<</SYS>>")[1].split("[/INST]")[0]
        label = row["gold"]
        pred = row["pred"]
        print(pred)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an evaluation system and rate outputs by another model."},
                {"role": "user", "content": f"An LLM was queried with ### {prompt} ### and answered with ### {pred} "
                                            f"### The actual answer is ### {label} ###. Is the answer provided by the LLM "
                                            f"corrcet? Answer with 1, if yes, or 0, if no."}
            ]
        )
        predictions.append(completion.choices[0].message.content)
        print(predictions)
    return predictions


if __name__ == "__main__":
    input_file_path = "Output_files/answers/evaluation_results_apstudy_SFT_only.csv"
    output_path= "Output_files/evaluated_answers/evaluation_results_apstudy_SFT_only.csv"
    client = OpenAI(api_key="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")
    MODEL = "gpt-4o"

    df = pd.read_csv(input_file_path)
    eval_results = generate_evaluation(df, MODEL)
    df['evaluation'] = eval_results
    df.to_csv(output_path, index=False)

