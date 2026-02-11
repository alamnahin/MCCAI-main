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
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    chencherry = SmoothingFunction()
    
    bleu4_scores = []
    rouge_scores = []
    
    print("Generating reports and calculating metrics...")
    
    with torch.no_grad():
        for batch in tqdm(val_loader):
            images = batch['image'].to(Config.DEVICE)
            gt_report = batch['gt_report'][0]
            
            # Generate
            gen_ids = model.report_gen.generate_report(images, max_length=64, num_beams=4, no_repeat_ngram_size=2)
            gen_report = model.report_gen.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            
            # 1. BLEU-4
            ref_tokens = gt_report.lower().split()
            hyp_tokens = gen_report.lower().split()
            if len(hyp_tokens) > 0:
                b4 = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=chencherry.method1)
                bleu4_scores.append(b4)
            
            # 2. ROUGE-L
            r_score = scorer.score(gt_report, gen_report)['rougeL'].fmeasure
            rouge_scores.append(r_score)
            
    print("\n" + "="*30)
    print("FINAL REPORT METRICS")
    print("="*30)
    print(f"BLEU-4:  {sum(bleu4_scores)/len(bleu4_scores):.4f}")
    print(f"ROUGE-L: {sum(rouge_scores)/len(rouge_scores):.4f}")

if __name__ == '__main__':
    calculate_metrics()