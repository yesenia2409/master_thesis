import pandas as pd
from openai import OpenAI


def generate_evaluation(df, model):
    evaluation = []
    prompts = []
    predictions = []

    for idx, row in df.iterrows():
        prompt = row["input"].split("<</SYS>>")[1].split("[/INST]")[0]
        label = row["gold"]
        pred = row["pred"].split("[/INST]")[0]
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an evaluation system and rate outputs by another model."},
                {"role": "user", "content": f"An LLM answered with ### {pred} ### to an input ### {prompt} ###. "
                                            f"The actual answer is ### {label} ###."
                                            f" Is the answer provided by the LLM correct? "
                                            f"Answer with 1, if yes, or 0, if no."}
            ]
        )
        prompts.append(prompt)
        predictions.append(pred)
        evaluation.append(completion.choices[0].message.content)
        print(idx)
    return prompts, predictions, evaluation


if __name__ == "__main__":
    input_file_path = "Output_files/answers/output_for_evaluation_npee_mc_SFT_only.csv"
    output_path= "Output_files/evaluated_answers/evaluation_results_npee_mc_SFT_only.csv"
    client = OpenAI(api_key="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")
    MODEL = "gpt-4o"

    df = pd.read_csv(input_file_path)
    df = df.loc[df['id'].isin(["choice"])]

    # print(df)
    prompts, preds, eval_results = generate_evaluation(df, MODEL)
    df['evaluation'] = eval_results
    df['input'] = prompts
    df['pred'] = preds
    df.to_csv(output_path, index=False)

