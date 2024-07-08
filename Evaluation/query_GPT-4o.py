import pandas as pd
from openai import OpenAI
import numpy as np


def generate_evaluation(base, sft, rlhf, model):
    evaluation = []
    prompts = []
    base_predictions = []
    sft_predictions = []
    rlhf_predictions = []

    for idx, row in base.iterrows():
        prompt = row["input"].split("<</SYS>>")[1].split("[/INST]")[0]
        label = row["gold"]
        pred_base = row["pred"].split("[/INST]")[0]
        pred_sft = sft["pred"][idx].split("[/INST]")[0]
        pred_rlhf = rlhf["pred"][idx]
        # print("BASE: ", pred_base)
        # print("RLHF: ", pred_rlhf)
        # print("SFT: ", pred_sft)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an evaluation system and rate outputs by other models."},
                {"role": "user", "content": f"Three LLMs were queried with ### {prompt} ###. The actual answer is ### {label} ###."
                                            f"Which is the best answer?"
                                            f"Print only \"1\", if ### {base} ### is the best answer."
                                            f"Print only \"2\", if ### {sft} ### is the best answer."
                                            f"Print only \"3\", if ### {rlhf} ### is the best answer."}
                # {"role": "user", "content": f"An LLM answered with ### {pred} ### to an input ### {prompt} ###. "
                #                             f"The actual answer is ### {label} ###."
                #                             f" Is the answer provided by the LLM correct? "
                #                             f"Answer with 1, if yes, or 0, if no."}
            ]
        )

        prompts.append(prompt)
        base_predictions.append(pred_base)
        sft_predictions.append(pred_sft)
        rlhf_predictions.append(pred_rlhf)
        evaluation.append(completion.choices[0].message.content)
        print("EVAL: ", evaluation)
        # print(idx)
    return prompts, base_predictions, sft_predictions, rlhf_predictions, evaluation


if __name__ == "__main__":
    # input_file_path = "Output_files/answers/closed/output_for_evaluation_npee_tf_RLHF.csv"
    base_input = "Output_files/answers/output_for_evaluation_npee_qa_base.csv"
    sft_input = "Output_files/answers/output_for_evaluation_npee_qa_SFT_only.csv"
    rlhf_input = "Output_files/answers/output_for_evaluation_npee_qa_RLHF.csv"

    output_path = "Output_files/evaluated_answers/open/evaluation_results_npee_qa.csv"
    client = OpenAI(api_key="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")
    MODEL = "gpt-4o"

    indices = np.random.choice(range(445), size=50, replace=False)

    base = pd.read_csv(base_input)
    base = base.iloc[indices]
    print(base)
    sft = pd.read_csv(sft_input)
    sft = sft.iloc[indices]
    print(sft)
    rlhf = pd.read_csv(rlhf_input)
    rlhf = rlhf.iloc[indices]
    print(rlhf)

    prompts, base_pred, sft_pred, rlhf_pred, eval_results = generate_evaluation(base, sft, rlhf, MODEL)

    df = pd. DataFrame()
    df['input'] = prompts
    df['pred_base'] = base_pred
    df['pred_sft'] = sft_pred
    df['pred_rlhf'] = rlhf_pred
    df['evaluation'] = eval_results
    df.to_csv(output_path, index=False)
