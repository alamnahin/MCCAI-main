import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from tqdm import tqdm
from dataset import Config, RadGenDataset, get_transforms
from fusion_model import RadGen
from torch.utils.data import DataLoader

def calculate_metrics():
    # Load Data
    val_ds = RadGenDataset(Config.CSV_PATH, Config.IMG_DIR, fold=Config.TARGET_FOLD, split='val', transform=get_transforms('val'))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    # Load Model
    model = RadGen().to(Config.DEVICE)
    checkpoint = torch.load(f'best_radgen_fold{Config.TARGET_FOLD}.pth', map_location=Config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Setup Scorers
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    chencherry = SmoothingFunction()
    
    bleu1_scores = []
    bleu4_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    print("Generating reports and calculating metrics...")
    
    with torch.no_grad():
        for batch in tqdm(val_loader):
            images = batch['image'].to(Config.DEVICE)
            gt_report = batch['gt_report'][0]
            
            # Generate (no_repeat_ngram_size passed via **kwargs)
            gen_ids = model.report_gen.generate_report(images, max_length=64, num_beams=4, no_repeat_ngram_size=2)
            gen_report = model.report_gen.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            
            # BLEU scores
            ref_tokens = gt_report.lower().split()
            hyp_tokens = gen_report.lower().split()
            if len(hyp_tokens) > 0:
                b1 = sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0),
                                   smoothing_function=chencherry.method1)
                b4 = sentence_bleu([ref_tokens], hyp_tokens,
                                   smoothing_function=chencherry.method1)
                bleu1_scores.append(b1)
                bleu4_scores.append(b4)
            
            # ROUGE scores
            r_scores = scorer.score(gt_report, gen_report)
            rouge1_scores.append(r_scores['rouge1'].fmeasure)
            rouge2_scores.append(r_scores['rouge2'].fmeasure)
            rougeL_scores.append(r_scores['rougeL'].fmeasure)
    
    metrics = {
        'bleu1': sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0.0,
        'bleu4': sum(bleu4_scores) / len(bleu4_scores) if bleu4_scores else 0.0,
        'rouge1': sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
        'rouge2': sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0,
        'rougeL': sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0,
    }
    
    print("\n" + "="*30)
    print("FINAL REPORT METRICS")
    print("="*30)
    for k, v in metrics.items():
        print(f"{k}:  {v:.4f}")
    
    return metrics

if __name__ == '__main__':
    calculate_metrics()