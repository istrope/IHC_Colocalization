python normalized_by_mito.py \
  --root_dir Data/Batch_5 \
  --out_csv output_mito/Batch_5/colocalization_results.csv \
  --qa_dir output_mito/Batch_5/qa_out \
  --cellpose_csv CellPose/batch5_counts.csv \
  --normalize_by_green &

python normalized_by_mito.py \
  --root_dir Data/Batch_6 \
  --out_csv output_mito/Batch_6/colocalization_results.csv \
  --qa_dir output_mito/Batch_6/qa_out \
  --cellpose_csv CellPose/batch6_counts.csv \
  --normalize_by_green &

python normalized_by_mito.py \
  --root_dir Data/Batch_7 \
  --out_csv output_mito/Batch_7/colocalization_results.csv \
  --qa_dir output_mito/Batch_7/qa_out \
  --cellpose_csv CellPose/batch7_counts.csv \
  --normalize_by_green &

python normalized_by_mito.py \
  --root_dir Data/Batch_8 \
  --out_csv output_mito/Batch_8/colocalization_results.csv \
  --qa_dir output_mito/Batch_8/qa_out \
  --cellpose_csv CellPose/batch3_counts.csv \
  --normalize_by_green &

wait
echo "All batches completed."


python plot_results.py --csvs output_mito/Batch_5/colocalization_results.csv --outdir boxplots --metrics IWC_Cytoplasm_% OverlapIntensity_Cytoplasm Ch2_Total_Cytoplasm 
python plot_results.py --csvs output_mito/Batch_6/colocalization_results.csv --outdir boxplots --metrics IWC_Cytoplasm_% OverlapIntensity_Cytoplasm Ch2_Total_Cytoplasm 
python plot_results.py --csvs output_mito/Batch_7/colocalization_results.csv  --outdir boxplots --metrics IWC_Cytoplasm_% OverlapIntensity_Cytoplasm Ch2_Total_Cytoplasm 
python plot_results.py --csvs output_mito/Batch_8/colocalization_results.csv --outdir boxplots --metrics IWC_Cytoplasm_% OverlapIntensity_Cytoplasm Ch2_Total_Cytoplasm 
