import os
import sys
import glob
from pathlib import Path

# Make sure the TrackEval library is in your Python path
# You may need to adjust this path to where you have TrackEval installed
# TRACKEVAL_PATH = "/path/to/TrackEval"
# sys.path.insert(0, TRACKEVAL_PATH)

# Import TrackEval modules
import trackeval
from trackeval import eval
from trackeval import datasets
from trackeval import metrics

def run_evaluation(gt_folder_base, pred_folder_base, seqs_to_eval=None):
    """
    Evaluate MOT tracking performance using TrackEval
    
    Args:
        gt_folder_base: Base path to ground truth files
        pred_folder_base: Base path to prediction files
        seqs_to_eval: List of sequence names to evaluate (if None, all sequences are evaluated)
    """
    print(f"Evaluating tracking performance...")
    print(f"GT base folder: {gt_folder_base}")
    print(f"Prediction base folder: {pred_folder_base}")
    
    # Get list of all sequences if not specified
    if seqs_to_eval is None:
        # Get all sequence folders in the gt base directory
        seqs_to_eval = [os.path.basename(p) for p in glob.glob(os.path.join(gt_folder_base, "*")) 
                        if os.path.isdir(p)]
        print(f"Found {len(seqs_to_eval)} sequences: {seqs_to_eval}")
    
    # Create config for the evaluation
    config = {
        'USE_PARALLEL': True,
        'NUM_PARALLEL_CORES': 8,
        'BREAK_ON_ERROR': True,
        'PRINT_RESULTS': True,
        'PRINT_ONLY_COMBINED': False,
        'PRINT_CONFIG': True,
        'TIME_PROGRESS': True,
        'OUTPUT_SUMMARY': True,
        'OUTPUT_DETAILED': True,
        'PLOT_CURVES': False,
    }
    
    # Create dataset config for MOT Challenge format
    dataset_config = {
        'GT_FOLDER': gt_folder_base,
        'TRACKERS_FOLDER': pred_folder_base,
        'OUTPUT_FOLDER': os.path.join(pred_folder_base, "..", 'eval_results'),
        'TRACKERS_TO_EVAL': [''],  # This will look for prediction files directly
        'CLASSES_TO_EVAL': ['pedestrian'],  # Your class name
        'BENCHMARK': 'MOT',
        'SPLIT_TO_EVAL': 'val',  # This matches your file structure's "val" folder
        'INPUT_AS_ZIP': False,
        'PRINT_CONFIG': True,
        'TRACKER_SUB_FOLDER': '',  # No subfolder for predictions
        'OUTPUT_SUB_FOLDER': '',
        'SEQMAP_FOLDER': None,  # We'll create this dynamically
        'SEQMAP_FILE': None,  # We'll create this dynamically
        'SEQ_INFO': None,  # We'll create this manually
        'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',  # Format for GT files
        'SKIP_SPLIT_FOL': True,  # Skip split folder in ground truth path
    }
    
    # Create metrics config
    metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
    
    # Create sequence info based on the sequences to evaluate
    seq_info = {}
    for seq in seqs_to_eval:
        gt_file = os.path.join(gt_folder_base, seq, 'gt', 'gt.txt')
        if not os.path.isfile(gt_file):
            print(f"Warning: GT file not found: {gt_file}")
            continue
            
        # Check if corresponding prediction file exists
        pred_file = os.path.join(pred_folder_base, f"{seq}.txt")
        if not os.path.isfile(pred_file):
            print(f"Warning: Prediction file not found: {pred_file}")
            continue
            
        # Get sequence length (number of frames) from GT file
        try:
            with open(gt_file, 'r') as f:
                lines = f.readlines()
                frames = {int(line.split(',')[0]) for line in lines}
                seq_length = max(frames) - min(frames) + 1
                
            seq_info[seq] = {
                'seq_length': seq_length,
                'gt_file': gt_file,
                'pred_file': pred_file
            }
        except Exception as e:
            print(f"Error processing sequence {seq}: {e}")
    
    # Update dataset config with sequence info
    dataset_config['SEQ_INFO'] = seq_info
    
    # Create evaluator
    evaluator = eval.Evaluator(config)
    dataset_list = [datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    
    # Run evaluation
    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)
    
    return output_res, output_msg

if __name__ == "__main__":
    # Define your paths
    gt_base = "/mnt/beegfs/home/liu15/BASF/MOTRv2/data/val"
    pred_base = "/mnt/beegfs/home/liu15/BASF/MOTRv2/output/run3_prediction_txt/val"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(pred_base, "..", 'eval_results'), exist_ok=True)
    
    # Run evaluation (all sequences)
    results, msg = run_evaluation(gt_base, pred_base)
    
    # Print summary results
    print("\n===== EVALUATION RESULTS =====")
    for dataset in results.keys():
        for tracker in results[dataset].keys():
            for seq in results[dataset][tracker].keys():
                if seq == 'COMBINED_SEQ':
                    print(f"\nCOMBINED RESULTS:")
                    for metric, metric_results in results[dataset][tracker][seq].items():
                        if metric == 'HOTA':
                            print(f"HOTA: {metric_results['HOTA']:.3f}")
                        elif metric == 'CLEAR':
                            print(f"MOTA: {metric_results['MOTA']:.3f}")
                            print(f"IDF1: {metric_results['IDF1']:.3f}")
                            print(f"FP: {metric_results['FP']}")
                            print(f"FN: {metric_results['FN']}")
                            print(f"IDSw: {metric_results['IDSw']}")