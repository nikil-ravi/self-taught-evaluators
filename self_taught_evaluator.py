import os
from together import Together
import random
from pathlib import Path
from datasets import load_from_disk, Dataset
from prompts import MODIFIED_INSTRUCTION_GENERATION_PROMPT, JUDGEMENT_ANNOTATION_PROMPT
import argparse
import logging
import json
from groq import Groq
from typing import Any, Dict
import time

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

NUM_JUDGEMENTS = 3
NUM_ITERATIONS = 3
GROQ_INFERENCE_MODEL_ITERATION_1 = "meta-llama/llama-4-scout-17b-16e-instruct"


def load_data(data_path: str) -> Dataset:
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Dataset path {data_path} does not exist")
    return load_from_disk(data_path)


def create_eval_set(dataset: Dataset, eval_size: int = 100) -> Dataset:
    """Create a fixed evaluation set from the original dataset."""
    eval_dataset = dataset[-eval_size:]
    return eval_dataset


def evaluate_model_as_judge(
    model_name: str, 
    client: Any,
    eval_dataset: Dataset,
    output_file: Path = None
) -> Dict[str, float]:
    """Evaluate a model as a judge on the eval dataset."""
    
    correct_judgments = 0
    total_examples = len(eval_dataset)
    
    eval_results = []
    
    for i, row in enumerate(eval_dataset):
        if i % 10 == 0:
            logger.info(f"Evaluating example {i}/{total_examples}")
        
        # Get conversation (assumes first message is user instruction)
        conversation = row.get("conversation", [])
        if not conversation or conversation[0].get("role") != "user":
            continue
            
        instruction = conversation[0].get("content", "")
        
        # Generate responses (good and bad)
        good_response = generate_response(instruction, client, model_name)
        bad_response = generate_bad_response(instruction, good_response, client, model_name)
        
        # Randomize order
        if random.random() > 0.5:
            response_a, response_b = good_response, bad_response
            expected_verdict = "[[A]]"
        else:
            response_a, response_b = bad_response, good_response
            expected_verdict = "[[B]]"
        
        # Get model's judgment
        judgments = generate_judgements(
            instruction,
            response_a,
            response_b,
            client,
            model_name,
            num_judgements=1
        )
        
        if not judgments:
            continue
            
        judgment = judgments[0]
        
        # Extract verdict
        if "[[A]]" in judgment:
            predicted_verdict = "[[A]]"
        elif "[[B]]" in judgment:
            predicted_verdict = "[[B]]"
        else:
            predicted_verdict = "[[UNKNOWN]]"
        
        # Check if correct
        is_correct = predicted_verdict == expected_verdict
        if is_correct:
            correct_judgments += 1
        
        # Store results
        eval_result = {
            "instruction": instruction,
            "response_a": response_a,
            "response_b": response_b,
            "expected_verdict": expected_verdict,
            "predicted_verdict": predicted_verdict,
            "correct": is_correct,
            "judgment": judgment
        }
        eval_results.append(eval_result)
    
    # Calculate accuracy
    accuracy = (correct_judgments / total_examples) * 100 if total_examples > 0 else 0
    
    # Save detailed results if output file specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in eval_results:
                f.write(json.dumps(result) + '\n')
        logger.info(f"Saved detailed results to {output_file}")
    
    results = {
        "accuracy": accuracy,
        "total_examples": total_examples,
        "correct_judgments": correct_judgments
    }
    
    return results

def generate_response(prompt: str, client: Groq, model: str, max_tokens: int = 512) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def generate_bad_response(instruction: str, baseline_response: str, client: Groq, model: str) -> str:
    modified_instruction = MODIFIED_INSTRUCTION_GENERATION_PROMPT.format(
        instruction=instruction,
        baseline_response=baseline_response
    )
    content = generate_response(modified_instruction, client, model, max_tokens=512)

    try:
        modified_response = content.split("Modified Instruction:")[1].split("High-Quality Response:")[1].strip()
        return modified_response
    except IndexError:
        logger.warning("Failed to parse bad response; using baseline as fallback")
        return baseline_response


def generate_judgements(
    instruction: str,
    response_a: str,
    response_b: str,
    client: Any,
    model: str,
    num_judgements: int,
    temperature: float = 0.5
) -> list[str]:
    prompt = JUDGEMENT_ANNOTATION_PROMPT.format(instruction=instruction, response_a=response_a, response_b=response_b)
    judgments = []
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=temperature,
            n=num_judgements,
        )
        
        # Extract all judgments from the response
        judgments = []
        for choice in response.choices:
            judgment_text = choice.message.content.strip()
            if judgment_text:
                judgments.append(judgment_text)
            else:
                logger.warning("Received empty judgment from model.")
        
        logger.info(f"Generated {len(judgments)} judgments in batch")
        return judgments
        
    except Exception as e:
        logger.error(f"Error generating batch judgments: {e}")
        return []


def annotate_data(training_data: Dataset, model: str, client: Any, iteration_num: int, output_filename: Path) -> list[tuple[str, str, str, str]]:

    annotated_data = []
    skipped = 0
    
    logger.info(f"Annotation output will be saved to: {output_filename}")

    for i, row in enumerate(training_data):

        logger.info(f"Processing row {i}")

        # get conversation
        # assumes the first message is the user's instruction, TODO could just get by role
        conversation = row.get("conversation", [])
        if not conversation or len(conversation) < 1 or conversation[0].get("role") != "user":
            skipped += 1
            continue

        instruction = conversation[0].get("content", "")
        if not instruction:
            skipped += 1
            continue

        # generate (good) response and bad response
        good_response = generate_response(instruction, client, model)
        logger.info(f"Good response: {good_response}")

        bad_response = generate_bad_response(instruction, good_response, client, model)
        logger.info(f"Bad response: {bad_response}")

        # randomize response order and generate judgements
        response_a, response_b, expected_verdict = (
            (good_response, bad_response, "[[A]]") if random.random() > 0.5
            else (bad_response, good_response, "[[B]]")
        )
        logger.info(f"Response A: {response_a}")
        logger.info(f"Response B: {response_b}")
        logger.info(f"Expected verdict: {expected_verdict}")
        
        judgements = generate_judgements(
            instruction,
            response_a,
            response_b,
            client,
            model,
            num_judgements=NUM_JUDGEMENTS
        )
        logger.info(f"Raw Judgements received: {judgements}")
        
        # Filter for judgements that contain the expected verdict string
        valid_full_judgements = [j_text for j_text in judgements if expected_verdict in j_text]
        
        logger.info(f"Valid full judgements (containing '{expected_verdict}'): {valid_full_judgements}")
        if valid_full_judgements:
            selected_full_judgement = random.choice(valid_full_judgements)
        else:
            logger.warning(f"No valid judgements found containing '{expected_verdict}' for instruction: {instruction[:100]}...")
            skipped += 1
            continue
        annotated_data.append((
            instruction,
            good_response,
            bad_response,
            selected_full_judgement
        ))

        # Prepare data for JSONL
        finetuning_row = {
            "prompt": JUDGEMENT_ANNOTATION_PROMPT.format(
                instruction=instruction, 
                response_a=response_a, 
                response_b=response_b
            ),
            "completion": selected_full_judgement,
        }   

        # Write to JSONL file
        try:
            with open(output_filename, "a", encoding="utf-8") as f:
                f.write(json.dumps(finetuning_row) + "\n")
        except IOError as e:
            logger.error(f"Could not write to {output_filename}: {e}")

        logger.info("Fine-tuning row saved", finetuning_row)

    logger.info(f"Skipped {skipped} rows")
    logger.info(f"Annotated {len(annotated_data)} rows in total for iteration {iteration_num}")
    
    return annotated_data


def launch_finetuning_job(
    training_file_path: Path,
    model_name: str,
    suffix: str,
    n_epochs: int,
    n_checkpoints: int,
    lora: bool,
) -> str:
    if not training_file_path.exists():
        raise FileNotFoundError(f"Training file {training_file_path} not found")
    client = Together(api_key=os.getenv("TOGETHER_API_KEY") or "")
    file_resp = client.files.upload(str(training_file_path), check=True)
    ft_resp = client.fine_tuning.create(
        training_file=file_resp.id,
        model=model_name,
        n_epochs=n_epochs,
        n_checkpoints=n_checkpoints,
        suffix=suffix,
        lora=lora,
    )
    return ft_resp
    

def main():
    parser = argparse.ArgumentParser(description="Iteratively train self-taught evaluator")
    parser.add_argument("--data_path", type=Path, required=True, help="Path to raw dataset")
    parser.add_argument("--output_dir", type=Path, default="finetuning_data", help="Directory for prepared JSONL data")
    parser.add_argument("--model_name", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1", help="Base model for fine-tuning")
    parser.add_argument("--suffix", type=str, default="self_taught_eval_ft_500_samples", help="Base suffix for fine-tuned model names")
    parser.add_argument("--n_iterations", type=int, default=NUM_ITERATIONS, help="Number of iterations of finetuning/judgement")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs per iteration")
    parser.add_argument("--n_checkpoints", type=int, default=1, help="Number of checkpoints to save")
    parser.add_argument("--lora", type=bool, default=True, help="Use LoRA for fine-tuning")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples to use for response generation and judgement")
    parser.add_argument("--eval_set_size", type=int, default=50, help="Number of samples to use for evaluation")
    
    args = parser.parse_args()

    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY environment variable not set")

    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        raise ValueError("WANDB_API_KEY environment variable not set")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    together_client = Together(api_key=api_key)
    
    # Load the full dataset
    full_dataset = load_data(args.data_path)

    # Create a fixed eval set at the beginning
    eval_dataset = create_eval_set(full_dataset, args.eval_set_size)
    
    # Determine the number of samples to select
    num_samples_to_take = args.num_samples
    actual_num_samples = min(num_samples_to_take, len(full_dataset))
    
    if actual_num_samples < num_samples_to_take:
        logger.warning(
            f"Requested {num_samples_to_take} samples, but the full dataset only has {len(full_dataset)}. "
            f"Using {actual_num_samples} samples."
        )
    
    # Select the subset
    dataset = full_dataset.select(range(actual_num_samples))
    
    logger.info(f"Selected {len(dataset)} samples for processing: {dataset}")
    
    current_model = args.model_name #"nikilrav/Mixtral-8x7B-Instruct-v0.1:self_taught_eval_ft_iter1:ft-7a88998b-0b0e" #args.model_name
    logger.info(f"Starting iterations with model: {current_model}")

    for iteration in range(1, args.n_iterations + 1):
        logger.info(f"Starting iteration {iteration}/{args.n_iterations}")

        output_filename = args.output_dir / f"annotations_iter{iteration}.jsonl"
        if iteration == 1:
            model_for_annotation = GROQ_INFERENCE_MODEL_ITERATION_1
        else:
            model_for_annotation = current_model
        
        current_client = Together(api_key=api_key)
        
        logger.info(f"Annotating data using model: {model_for_annotation} via {'Groq' if iteration == 1 else 'Together'} API...")
        annotated_data = annotate_data(dataset, model_for_annotation, current_client, iteration, output_filename)

        logger.info(f"Annotated {len(annotated_data)} data points")
        if not annotated_data:
            logger.error("No valid annotated data generated; skipping iteration")
            continue

        logger.info("Launching fine-tuning job...")
        suffix = f"{args.suffix}_iter{iteration}"
        try:
            ft_resp = launch_finetuning_job(
                output_filename,
                current_model,
                suffix,
                args.n_epochs,
                args.n_checkpoints,
                args.lora,
                wandb_api_key
            )
            logger.info(f"Fine-tuning job launched for iteration {iteration}. Job ID: {ft_resp.id}")

            # Monitor fine-tuning job status
            job_to_monitor_id = ft_resp.id
            new_fine_tuned_model_id = None

            while True:
                try:
                    status_response = together_client.fine_tuning.retrieve(job_to_monitor_id)
                    logger.info(f"Polling Job ID {job_to_monitor_id}: Status = {status_response.status}")

                    if status_response.status == 'completed':
                        new_fine_tuned_model_id = ft_resp.output_name
                        logger.info(f"Fine-tuning job {job_to_monitor_id} completed. New model ID: {new_fine_tuned_model_id}")
                        break
                    elif status_response.status in ['error', 'failed', 'cancelled', 'ERROR', 'FAILED', 'CANCELLED']:
                        logger.error(f"Fine-tuning job {job_to_monitor_id} ended with status: {status_response.status}.")
                        if hasattr(status_response, 'events') and status_response.events:
                            logger.error(f"Events for job {job_to_monitor_id}: {status_response.events}")
                        new_fine_tuned_model_id = None
                        break
                except Exception as e:
                    logger.error(f"Error while retrieving status for job {job_to_monitor_id}: {e}. Retrying in 30s.")
                
                time.sleep(30)

            if new_fine_tuned_model_id:
                current_model = new_fine_tuned_model_id
            else:
                logger.error(f"Fine-tuning job {job_to_monitor_id} did not complete successfully. Continuing with previous model: {current_model}")
                continue
            
        except Exception as e:
            logger.error(f"Failed to launch or monitor fine-tuning job in iteration {iteration}: {e}")
            continue

    logger.info("Iterative training completed")

    # evaluate on eval set
    logger.info("Evaluating base model...")
    base_model_results = evaluate_model_as_judge(
        args.model_name,
        together_client,
        eval_dataset,
        args.output_dir / "base_model_eval.jsonl"
    )
    logger.info(f"Base model results: {base_model_results}")
    

    logger.info("Evaluating final model...")
    final_model_results = evaluate_model_as_judge(
        current_model,
        together_client,
        eval_dataset,
        args.output_dir / "final_model_eval.jsonl"
    )
    logger.info(f"Final model results: {final_model_results}")

    # Compare results
    logger.info("\n" + "="*50)
    logger.info("EVALUATION COMPARISON")
    logger.info("="*50)
    logger.info(f"Base model accuracy:  {base_model_results['accuracy']:.2f}%")
    logger.info(f"Final model accuracy: {final_model_results['accuracy']:.2f}%")
    logger.info(f"Improvement: {final_model_results['accuracy'] - base_model_results['accuracy']:.2f}%")
    logger.info("="*50)
    

if __name__ == "__main__":
    main()