# Master Thesis in Computational Linguistics (SoSe204, University of Tübingen)
### Title: Refinement of Large Language Models by Domain Specialiazation, using Reinforcement Learning from Human Feedback
### Author: Annica Skupch

## Aim 
This thesis aims to contribute to the field of domain specialization for LLMs. 
In particular, a new approach to refine a general-purpose open-source LLM in the domain of geoscience was tested. 
The domain of geoscience served as an example throughout this thesis, however, the goal was to propose a new approach for domain specialization that is generalizable and, therefore, applicable to all kinds of domains in the future.
In particular, the potential of using Reinforcement Learning with Human Feedback (RLHF), including Supervised Fine-Tuning (SFT), as well as prompt crafting for domain specialization has been explored.

## Project Description
This project provides code for a domain specialization approach to train and evaluate a general-purpose LLM, transforming it into a Geoscience expert. 
The code applies techniques like Prompt Crafting, SFT, and RLHF, all implemented using Python.

## File Structure
- Model Selection:
    - Annotated_files and Output_files: Store respective data and results.
    - model_selection.py and run_job.sh: Main scripts for model selection.

- Prompting:
    - Adjusting_Dataset: Includes input and output files for dataset adjustment.
    - Proof_Of:Concept: Contains annotated files and output files created during the proof of concept.
    - Selection_of_system_prompt: Contains annotated files and output files created during the selection of a system prompt.
    - Key scripts are adjusting_dataset.py, plot_results.py, proof_of_concept.py, run_job.sh, and system_prompt.py.

- SFT: 
    - Annotated_files, Input_files, Output_files: Organized for supervised fine-tuning tasks.
    - inference_tests, plots_failed_SFT_tries, plots_hyperparameter_tests, slurm_files in Output_files: Contain various results and configurations.
    - Key scripts include inference.py, plot_results.py, run_job.sh, and training.py.

- RLHF:
    - Annotated_files, Input_files, Output_files: Structured for reward modeling and policy training. 
    - Inference_tests, loss_plots, slurm_files in Output_files: Store test results and job configurations.
    - Main scripts: create_reward_model_dataset.py, extract_rewards_from_policy_training.py, ppo.py, reward_model.py, and run_job.sh.

- Evaluation: 
    - Contains subdirectories for annotated files, input files (in CSV, JSON, and PKL formats), and output files including answers, evaluated answers, and plots. 
    - Key scripts include inference.py, plots.py, prepare_evaluation_data.py, query_GPT-4o.py, and run_job.sh.

## Requirements
The project requires a single A100 GPU with a maximum of 50GB RAM. 
It is designed to run on Python 3.10 and needs all necessary packages specified in the requirements.txt file.