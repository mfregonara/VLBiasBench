import os
import json
import logging
from datetime import datetime
from PIL import Image

from utils import setup_seeds
from models.load_model import load_model


def load_category_dataset(category_name, max_samples=100): #TODO: change to be a variable
    dataset_path = f"./data/close-ended/json/base/{category_name}.json"
    image_base_path = "./data/close-ended/images"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path, "r") as f:
        data = json.load(f)

    data = data[:max_samples]

    dataset = []
    for idx, sample in enumerate(data):
        image_path = os.path.join(image_base_path, sample["image_path"])

        instruction = f"{sample['context']} {sample['question']} Please choose from: (A) {sample['ans0']}, (B) {sample['ans1']}, (C) {sample['ans2']}"

        dataset_item = {
            "id": idx,
            "instruction": instruction,
            "images": [image_path],
            "context": sample["context"],
            "question": sample["question"],
            "choices": {"A": sample["ans0"], "B": sample["ans1"], "C": sample["ans2"]},
            "correct_label": sample["label"],
            "category": sample["category"],
            "condition": sample.get("condition", "unknown"),
        }
        dataset.append(dataset_item)

    return dataset


def evaluate_category(model, category_name, base_output_dir, max_samples=1000): #decided to use logger instead of print statements for consistency
    category_output_dir = os.path.join(
        base_output_dir, f"{category_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    os.makedirs(category_output_dir, exist_ok=True)

    logger = logging.getLogger(f"gemma-3-4b|{category_name}")
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(
        os.path.join(category_output_dir, "log.txt"), mode="a"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(""))

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(
        logging.Formatter("%(name)s - %(asctime)s - %(levelname)s - %(message)s")
    )

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    try:
        logger.info(f"=== Evaluating {category_name} ===")

        try:
            dataset = load_category_dataset(category_name, max_samples)
            logger.info(f"Loaded {len(dataset)} samples from {category_name} dataset")
        except FileNotFoundError as e:
            logger.error(f"Could not load {category_name}: {e}")
            return {
                "category": category_name,
                "total": 0,
                "successful": 0,
                "failed": 0,
                "output_dir": category_output_dir,
            }

        with open(os.path.join(category_output_dir, "config.json"), "w") as json_file:
            json.dump(model.config, json_file, indent=4)

        result_json = []
        successful_predictions = 0

        for idx, test_case in enumerate(dataset):
            try:
                pred = model.generate(
                    instruction=test_case["instruction"],
                    images=test_case["images"],
                )
                successful_predictions += 1

            except Exception as error:
                pred = ""
                logger.error(
                    f"An exception occurred for {category_name} sample {idx}: {error}"
                )

            logger.info(f"ID:\t{idx}")
            logger.info(f'Category:\t{test_case["category"]}')
            logger.info(f'Condition:\t{test_case["condition"]}')
            logger.info(f'Instruction:\t{test_case["instruction"]}')
            logger.info(f'Images:\t{test_case["images"]}')
            logger.info(f'Correct Label:\t{test_case["correct_label"]}')
            logger.info(f"Answer:\t{pred}")
            logger.info("-" * 60)

            result_case = {
                "id": test_case["id"],
                "instruction": test_case["instruction"],
                "in_images": test_case["images"],
                "answer": pred,
                "context": test_case["context"],
                "question": test_case["question"],
                "choices": test_case["choices"],
                "correct_label": test_case["correct_label"],
                "category": test_case["category"],
                "condition": test_case["condition"],
            }
            result_json.append(result_case)

            if (idx + 1) % 25 == 0:
                logger.info(
                    f"Progress: {idx + 1}/{len(dataset)} samples processed for {category_name}"
                )

        with open(os.path.join(category_output_dir, "result.json"), "w") as json_file:
            json.dump(result_json, json_file, indent=4)

        logger.info(
            f"{category_name} complete: {successful_predictions}/{len(dataset)} successful predictions"
        )
        logger.info(f"Results saved to: {category_output_dir}")

        return {
            "category": category_name,
            "total": len(dataset),
            "successful": successful_predictions,
            "failed": len(dataset) - successful_predictions,
            "output_dir": category_output_dir,
        }

    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


def main():
    print("=== Running Gemma 3 4B Evaluation ===")

    categories = [
        "Age_x_gender",
    ]
    max_samples_per_category = 100 #TODO: change to be a variable

    setup_seeds(seed=42)
    model_name = "gemma-3-4b"

    base_output_dir = os.path.join("./outputs", model_name)
    os.makedirs(base_output_dir, exist_ok=True)

    main_logger = logging.getLogger("main_evaluation")
    main_logger.setLevel(logging.INFO)

    main_stream_handler = logging.StreamHandler()
    main_stream_handler.setLevel(logging.INFO)
    main_stream_handler.setFormatter(
        logging.Formatter("MAIN - %(asctime)s - %(levelname)s - %(message)s")
    )
    main_logger.addHandler(main_stream_handler)

    try:
        main_logger.info("Loading Gemma 3 4B model...")
        model = load_model(model_name)
        main_logger.info("Model loaded successfully")

        category_summaries = []

        for i, category in enumerate(categories, 1):
            main_logger.info(f"\n{'='*80}")
            main_logger.info(f"Starting evaluation {i}/{len(categories)}: {category}")
            main_logger.info(f"{'='*80}")

            category_summary = evaluate_category(
                model, category, base_output_dir, max_samples_per_category
            )
            category_summaries.append(category_summary)

            main_logger.info(
                f"Completed {category}: {category_summary['successful']}/{category_summary['total']} successful"
            )

        main_logger.info("\n" + "=" * 80)
        main_logger.info("EVALUATION COMPLETE - SUMMARY")
        main_logger.info("=" * 80)

        total_samples = 0
        total_successful = 0

        for summary in category_summaries:
            main_logger.info(
                f"{summary['category']}: {summary['successful']}/{summary['total']} successful"
            )
            main_logger.info(f"  Output: {summary['output_dir']}")
            total_samples += summary["total"]
            total_successful += summary["successful"]

        main_logger.info("-" * 40)
        main_logger.info(f"OVERALL: {total_successful}/{total_samples} successful")
        main_logger.info(f"Success rate: {total_successful/total_samples*100:.1f}%")
        main_logger.info("=" * 80)

    except Exception as e:
        main_logger.error(f"Evaluation failed: {e}")
        raise

    finally:
        for handler in main_logger.handlers[:]:
            handler.close()
            main_logger.removeHandler(handler)


if __name__ == "__main__":
    main()

