python scripts/normalized_by_mito.py \
  --root_dir /Users/bstrope/Projects/Mariah/Data/batch_10 \
  --out_csv /Users/bstrope/Projects/Mariah/output_mito/Batch_10/colocalization_results.csv \
  --qa_dir /Users/bstrope/Projects/Mariah/output_mito/Batch_10/qa_out \
  --cellpose_csv /Users/bstrope/Projects/Mariah/CellPose/batch10_counts.csv \
  --normalize_by_green &

python scripts/normalized_by_mito.py \
  --root_dir /Users/bstrope/Projects/Mariah/Data/batch_11 \
  --out_csv /Users/bstrope/Projects/Mariah/output_mito/Batch_11/colocalization_results.csv \
  --qa_dir /Users/bstrope/Projects/Mariah/output_mito/Batch_11/qa_out \
  --cellpose_csv /Users/bstrope/Projects/Mariah/CellPose/batch11_counts.csv \
  --normalize_by_green &

python scripts/normalized_by_mito.py \
  --root_dir /Users/bstrope/Projects/Mariah/Data/batch_12 \
  --out_csv /Users/bstrope/Projects/Mariah/output_mito/Batch_12/colocalization_results.csv \
  --qa_dir /Users/bstrope/Projects/Mariah/output_mito/Batch_12/qa_out \
  --cellpose_csv /Users/bstrope/Projects/Mariah/CellPose/batch12_counts.csv \
  --normalize_by_green &
wait
echo "All batches completed."


python scripts/plot_results.py --csvs /Users/bstrope/Projects/Mariah/output_mito/Batch_10/colocalization_results.csv --outdir /Users/bstrope/Projects/Mariah/boxplots --metrics IWC_Cytoplasm_% OverlapIntensity_Cytoplasm Ch2_Total_Cytoplasm 
python scripts/plot_results.py --csvs /Users/bstrope/Projects/Mariah/output_mito/Batch_11/colocalization_results.csv --outdir /Users/bstrope/Projects/Mariah/boxplots --metrics IWC_Cytoplasm_% OverlapIntensity_Cytoplasm Ch2_Total_Cytoplasm 
python scripts/plot_results.py --csvs /Users/bstrope/Projects/Mariah/output_mito/Batch_12/colocalization_results.csv  --outdir /Users/bstrope/Projects/Mariah/boxplots --metrics IWC_Cytoplasm_% OverlapIntensity_Cytoplasm Ch2_Total_Cytoplasm 
