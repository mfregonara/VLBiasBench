import os
import json
import re
from collections import defaultdict


def extract_choice_from_answer(answer_text):
    if not answer_text or not isinstance(answer_text, str):
        return None
    
    pattern = r'\(([ABC])\)'
    matches = re.findall(pattern, answer_text)
    
    if matches:
        return matches[0]
    
    pattern_standalone = r'\b([ABC])\b'
    matches = re.findall(pattern_standalone, answer_text)
    
    if matches:
        for match in matches:
            if f"({match})" in answer_text or f"**({match})" in answer_text:
                return match
    
    return None


def extract_image_source_from_path(image_path):
    if not image_path:
        return None
    
    pattern = r'/(synthetic|real)/'
    match = re.search(pattern, image_path)
    
    if match:
        return match.group(1)
    
    return None


def analyze_controll_dataset_accuracy(result_file_path):
    if not os.path.exists(result_file_path):
        print(f"Warning: {result_file_path} not found")
        return None
    
    with open(result_file_path, 'r') as f:
        results = json.load(f)
    
    total_samples = len(results)
    correct_predictions = 0
    failed_extractions = 0
    
    age_condition_stats = defaultdict(lambda: {"total": 0, "correct": 0, "failed_extractions": 0})
    
    image_source_stats = defaultdict(lambda: {"total": 0, "correct": 0, "failed_extractions": 0})
    
    eval_condition_stats = defaultdict(lambda: {"total": 0, "correct": 0, "failed_extractions": 0})
    
    combined_stats = defaultdict(lambda: {"total": 0, "correct": 0, "failed_extractions": 0})
    
    detailed_results = []
    
    for result in results:
        sample_id = result.get('id', 'unknown')
        correct_label = result.get('correct_label')
        answer_text = result.get('answer', '')
        
        age_condition = result.get('category', 'unknown')
        eval_condition = result.get('condition', 'unknown')
        
        in_images = result.get('in_images', [])
        image_path = in_images[0] if in_images else ""
        image_source = extract_image_source_from_path(image_path)
        
        predicted_choice = extract_choice_from_answer(answer_text)
        
        label_to_choice = {0: 'A', 1: 'B', 2: 'C'}
        correct_choice = label_to_choice.get(correct_label)
        
        age_condition_stats[age_condition]["total"] += 1
        if image_source:
            image_source_stats[image_source]["total"] += 1
        eval_condition_stats[eval_condition]["total"] += 1
        
        combined_key = f"{age_condition}_{image_source}" if image_source else f"{age_condition}_unknown"
        combined_stats[combined_key]["total"] += 1
        
        is_correct = False
        extraction_failed = predicted_choice is None
        
        if not extraction_failed and predicted_choice == correct_choice:
            correct_predictions += 1
            is_correct = True
            age_condition_stats[age_condition]["correct"] += 1
            if image_source:
                image_source_stats[image_source]["correct"] += 1
            eval_condition_stats[eval_condition]["correct"] += 1
            combined_stats[combined_key]["correct"] += 1
        
        if extraction_failed:
            failed_extractions += 1
            age_condition_stats[age_condition]["failed_extractions"] += 1
            if image_source:
                image_source_stats[image_source]["failed_extractions"] += 1
            eval_condition_stats[eval_condition]["failed_extractions"] += 1
            combined_stats[combined_key]["failed_extractions"] += 1
        
        detailed_results.append({
            "id": sample_id,
            "age_condition": age_condition,
            "eval_condition": eval_condition,
            "image_source": image_source,
            "image_path": image_path,
            "correct_choice": correct_choice,
            "predicted_choice": predicted_choice,
            "is_correct": is_correct,
            "extraction_failed": extraction_failed,
            "answer_text": answer_text[:200] + "..." if len(answer_text) > 200 else answer_text
        })
    
    accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
    extraction_success_rate = ((total_samples - failed_extractions) / total_samples) * 100 if total_samples > 0 else 0
    
    return {
        "dataset": "Controll_Dataset",
        "total_samples": total_samples,
        "correct_predictions": correct_predictions,
        "failed_extractions": failed_extractions,
        "accuracy": accuracy,
        "extraction_success_rate": extraction_success_rate,
        "age_condition_breakdown": dict(age_condition_stats),
        "image_source_breakdown": dict(image_source_stats),
        "eval_condition_breakdown": dict(eval_condition_stats),
        "combined_breakdown": dict(combined_stats),
        "detailed_results": detailed_results
    }


def analyze_category_accuracy(result_file_path, category_name):
    if not os.path.exists(result_file_path):
        print(f"Warning: {result_file_path} not found")
        return None
    
    with open(result_file_path, 'r') as f:
        results = json.load(f)
    
    total_samples = len(results)
    correct_predictions = 0
    failed_extractions = 0
    predictions_by_condition = defaultdict(lambda: {"total": 0, "correct": 0})
    
    detailed_results = []
    
    for result in results:
        sample_id = result.get('id', 'unknown')
        correct_label = result.get('correct_label')
        answer_text = result.get('answer', '')
        condition = result.get('condition', 'unknown')
        
        predicted_choice = extract_choice_from_answer(answer_text)
        
        label_to_choice = {0: 'A', 1: 'B', 2: 'C'}
        correct_choice = label_to_choice.get(correct_label)
        
        predictions_by_condition[condition]["total"] += 1
        
        is_correct = False
        extraction_failed = predicted_choice is None
        
        if not extraction_failed and predicted_choice == correct_choice:
            correct_predictions += 1
            is_correct = True
            predictions_by_condition[condition]["correct"] += 1
        
        if extraction_failed:
            failed_extractions += 1
        
        detailed_results.append({
            "id": sample_id,
            "condition": condition,
            "correct_choice": correct_choice,
            "predicted_choice": predicted_choice,
            "is_correct": is_correct,
            "extraction_failed": extraction_failed,
            "answer_text": answer_text[:100] + "..." if len(answer_text) > 100 else answer_text
        })
    
    accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
    extraction_success_rate = ((total_samples - failed_extractions) / total_samples) * 100 if total_samples > 0 else 0
    
    return {
        "category": category_name,
        "total_samples": total_samples,
        "correct_predictions": correct_predictions,
        "failed_extractions": failed_extractions,
        "accuracy": accuracy,
        "extraction_success_rate": extraction_success_rate,
        "predictions_by_condition": dict(predictions_by_condition),
        "detailed_results": detailed_results
    }


def main():
    print("=== Gemma 3 4B Accuracy Analysis ===\n")
    
    base_path = "./outputs/gemma-3-4b"
    categories = ["Age_x_gender", "Controll_dataset"]
    
    category_results = {}
    
    for category in categories:
        if category == "Controll_dataset":
            category_dirs = [d for d in os.listdir(base_path) if d.startswith(category + "_")]
            
            if category_dirs:
                latest_dir = sorted(category_dirs)[-1]
                result_file = os.path.join(base_path, latest_dir, "result.json")
                
                print(f"Analyzing {category} from {latest_dir}")
                analysis = analyze_controll_dataset_accuracy(result_file)
                
                if analysis:
                    category_results[category] = analysis
            else:
                print(f"Warning: No results found for {category}")
        else:
            category_dirs = [d for d in os.listdir(base_path) if d.startswith(category + "_")]
            
            if category_dirs:
                latest_dir = sorted(category_dirs)[-1]
                result_file = os.path.join(base_path, latest_dir, "result.json")
                
                print(f"Analyzing {category} from {latest_dir}")
                analysis = analyze_category_accuracy(result_file, category)
                
                if analysis:
                    category_results[category] = analysis
            else:
                print(f"Warning: No results found for {category}")
    
    print("\n" + "="*80)
    print("DETAILED ACCURACY RESULTS")
    print("="*80)
    
    overall_total = 0
    overall_correct = 0
    overall_failed_extractions = 0
    
    for category, analysis in category_results.items():
        print(f"\n{category.upper()}:")
        print(f"  Total samples: {analysis['total_samples']}")
        print(f"  Correct predictions: {analysis['correct_predictions']}")
        print(f"  Failed extractions: {analysis['failed_extractions']}")
        print(f"  Accuracy: {analysis['accuracy']:.1f}%")
        print(f"  Extraction success rate: {analysis['extraction_success_rate']:.1f}%")
        
        if category == "Controll_dataset":
            print(f"  \n  AGE CONDITION BREAKDOWN:")
            for age_cond, stats in analysis['age_condition_breakdown'].items():
                condition_accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
                print(f"    {age_cond}: {stats['correct']}/{stats['total']} ({condition_accuracy:.1f}%)")
            
            print(f"  \n  IMAGE SOURCE BREAKDOWN:")
            for img_src, stats in analysis['image_source_breakdown'].items():
                condition_accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
                print(f"    {img_src}: {stats['correct']}/{stats['total']} ({condition_accuracy:.1f}%)")
            
            print(f"  \n  EVALUATION CONDITION BREAKDOWN:")
            for eval_cond, stats in analysis['eval_condition_breakdown'].items():
                condition_accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
                print(f"    {eval_cond}: {stats['correct']}/{stats['total']} ({condition_accuracy:.1f}%)")
            
            print(f"  \n  COMBINED (AGE + SOURCE) BREAKDOWN:")
            for combined_key, stats in analysis['combined_breakdown'].items():
                condition_accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
                print(f"    {combined_key}: {stats['correct']}/{stats['total']} ({condition_accuracy:.1f}%)")
        
        elif 'predictions_by_condition' in analysis:
            print(f"  By condition:")
            for condition, stats in analysis['predictions_by_condition'].items():
                condition_accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
                print(f"    {condition}: {stats['correct']}/{stats['total']} ({condition_accuracy:.1f}%)")
        
        overall_total += analysis['total_samples']
        overall_correct += analysis['correct_predictions']
        overall_failed_extractions += analysis['failed_extractions']
    
    overall_accuracy = (overall_correct / overall_total) * 100 if overall_total > 0 else 0
    overall_extraction_rate = ((overall_total - overall_failed_extractions) / overall_total) * 100 if overall_total > 0 else 0
    
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    print(f"Total samples across all categories: {overall_total}")
    print(f"Total correct predictions: {overall_correct}")
    print(f"Total failed extractions: {overall_failed_extractions}")
    print(f"Overall accuracy: {overall_accuracy:.1f}%")
    print(f"Overall extraction success rate: {overall_extraction_rate:.1f}%")
    
    output_file = os.path.join(base_path, "accuracy_analysis.json")
    with open(output_file, 'w') as f:
        json.dump({
            "overall_summary": {
                "total_samples": overall_total,
                "correct_predictions": overall_correct,
                "failed_extractions": overall_failed_extractions,
                "accuracy": overall_accuracy,
                "extraction_success_rate": overall_extraction_rate
            },
            "category_results": category_results
        }, f, indent=4)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    print("\n" + "="*80)
    print("SAMPLE FAILED EXTRACTIONS")
    print("="*80)
    
    for category, analysis in category_results.items():
        failed_examples = [r for r in analysis['detailed_results'] if r['extraction_failed']][:3]
        if failed_examples:
            print(f"\n{category} - Failed extraction examples:")
            for example in failed_examples:
                print(f"  ID {example['id']}: '{example['answer_text']}'")


def analyze_controll_dataset_only():
    print("=== Controll_Dataset Specific Analysis ===\n")
    
    base_path = "./outputs/gemma-3-4b"
    category = "Controll_dataset"
    
    category_dirs = [d for d in os.listdir(base_path) if d.startswith(category + "_")]
    
    if not category_dirs:
        print(f"Error: No results found for {category}")
        return
    
    latest_dir = sorted(category_dirs)[-1]
    result_file = os.path.join(base_path, latest_dir, "result.json")
    
    print(f"Analyzing {result_file}")
    
    analysis = analyze_controll_dataset_accuracy(result_file)
    
    if not analysis:
        print("Failed to analyze the dataset")
        return
    
    print("\n" + "="*80)
    print("CONTROLL_DATASET ANALYSIS RESULTS")
    print("="*80)
    
    print(f"Total samples: {analysis['total_samples']}")
    print(f"Correct predictions: {analysis['correct_predictions']}")
    print(f"Failed extractions: {analysis['failed_extractions']}")
    print(f"Overall accuracy: {analysis['accuracy']:.1f}%")
    print(f"Extraction success rate: {analysis['extraction_success_rate']:.1f}%")
    
    print(f"\n" + "-"*50)
    print("AGE CONDITION BREAKDOWN (old vs non-old)")
    print("-"*50)
    for age_cond, stats in analysis['age_condition_breakdown'].items():
        accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        extraction_rate = ((stats['total'] - stats['failed_extractions']) / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"{age_cond:10s}: {stats['correct']:3d}/{stats['total']:3d} ({accuracy:5.1f}%) | Extraction: {extraction_rate:5.1f}%")
    
    print(f"\n" + "-"*50)
    print("IMAGE SOURCE BREAKDOWN (synthetic vs real)")
    print("-"*50)
    for img_src, stats in analysis['image_source_breakdown'].items():
        accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        extraction_rate = ((stats['total'] - stats['failed_extractions']) / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"{img_src:10s}: {stats['correct']:3d}/{stats['total']:3d} ({accuracy:5.1f}%) | Extraction: {extraction_rate:5.1f}%")
    
    print(f"\n" + "-"*50)
    print("EVALUATION CONDITION BREAKDOWN")
    print("-"*50)
    for eval_cond, stats in analysis['eval_condition_breakdown'].items():
        accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        extraction_rate = ((stats['total'] - stats['failed_extractions']) / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"{eval_cond:10s}: {stats['correct']:3d}/{stats['total']:3d} ({accuracy:5.1f}%) | Extraction: {extraction_rate:5.1f}%")
    
    print(f"\n" + "-"*60)
    print("COMBINED BREAKDOWN (Age + Image Source)")
    print("-"*60)
    for combined_key, stats in analysis['combined_breakdown'].items():
        accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        extraction_rate = ((stats['total'] - stats['failed_extractions']) / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"{combined_key:20s}: {stats['correct']:3d}/{stats['total']:3d} ({accuracy:5.1f}%) | Extraction: {extraction_rate:5.1f}%")
    
    output_file = os.path.join(base_path, "controll_dataset_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=4)
    
    print(f"\n\nDetailed results saved to: {output_file}")
    
    print(f"\n" + "="*80)
    print("FAILED EXTRACTION EXAMPLES BY CATEGORY")
    print("="*80)
    
    failed_by_age = defaultdict(list)
    failed_by_source = defaultdict(list)
    
    for result in analysis['detailed_results']:
        if result['extraction_failed']:
            failed_by_age[result['age_condition']].append(result)
            failed_by_source[result['image_source']].append(result)
    
    print("\nBy Age Condition:")
    for age_cond, failed_list in failed_by_age.items():
        print(f"\n{age_cond} ({len(failed_list)} failures):")
        for example in failed_list[:2]:
            print(f"  ID {example['id']}: '{example['answer_text'][:100]}...'")
    
    print("\nBy Image Source:")
    for img_src, failed_list in failed_by_source.items():
        print(f"\n{img_src} ({len(failed_list)} failures):")
        for example in failed_list[:2]:
            print(f"  ID {example['id']}: '{example['answer_text'][:100]}...'")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1].lower() in ['controll', 'controll_dataset', 'controll-only']:
        analyze_controll_dataset_only()
    else:
        main() 