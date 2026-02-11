## Usage Instructions

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run main RadGen training (produces best_radgen_fold4.pth)
python main.py

# 3. Run all experiments (generates all tables)
python run_all_experiments.py

# Or run individually:
python baselines.py              # Table 1: Baselines
python ablation_study.py         # Table 2: Ablations  
python report_metrics.py         # Table 3: Report quality

# 4. Check results
cat results_table1.json
cat results_table2.json
cat results_table3.json
```