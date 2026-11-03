#!/bin/bash
. /mnt/data/work/export/env.sh
O=/mnt/data/work/export/efficientsam3-ev-m; R=/mnt/data/work/aivm-safety-mode/jetson-bench; B=$R/bench; LG=$R/logs
CLIP=$R/footage/raw/pexels-15170997-warehouse-team.mp4; CK=/mnt/data/work/weights/efficientsam3/efficientsam3_efficientvit.pt
cd /tmp
run() { local log=$1; shift; sync; echo 3 > /proc/sys/vm/drop_caches; sleep 2; { echo "# $(date -Is)"; echo "\$ $*"; "$@" 2>&1 | grep -v -i "warn\|setter\|W:onnxruntime"; } > $LG/$log; echo "done $log"; }
run trt-fixed4-ev-m-fp16_pexels-15170997-warehouse-team.log python $B/trt_sam.py --mode fixed --engine $O/trt/fixed4_mask_final_fp16.plan --clip $CLIP --tag efficientsam3-ev-m-trt
run trt-open-ev-m-fp16-4p_pexels-15170997-warehouse-team.log python $B/trt_sam.py --mode open --engine $O/trt/vision_decoder_final_fp16.plan --emb $O/text_embeddings.npz --clip $CLIP --tag efficientsam3-ev-m-trt-open4
run trt-open-ev-m-fp16-3p_pexels-15170997-warehouse-team.log python $B/trt_sam.py --mode open --engine $O/trt/vision_decoder_final_fp16.plan --emb $O/text_embeddings.npz --prompts "person,forklift,high-visibility vest" --frames 20 --clip $CLIP --tag efficientsam3-ev-m-trt-open3
run trt-open-ev-m-fp32-4p_pexels-15170997-warehouse-team.log python $B/trt_sam.py --mode open --engine $O/trt/vision_decoder_final_fp32.plan --emb $O/text_embeddings.npz --frames 20 --clip $CLIP --tag efficientsam3-ev-m-trt-open4-fp32
run pt-text-cost-ev-m_pexels-15170997-warehouse-team.log python $B/pt_text_cost.py --ckpt $CK --clip $CLIP --tag pt-text-cost-ev-m
cd $B; run combo4-efficientsam3-ev-m-fp32_pexels-15170997-warehouse-team.log python bench_sam.py --model efficientsam3 --tag combo4-efficientsam3-ev-m-fp32 --ckpt $CK --backbone efficientvit --variant b1 --clip $CLIP --combo --prompts "person,forklift,high-visibility vest,hard hat" --frames 20; cd /tmp
G=$O/open_vocab/vision_decoder_noif.onnx
run ort-cpu-default-ev-m_pexels-15170997-warehouse-team.log python $B/ort_sam.py --onnx $G --emb $O/text_embeddings.npz --ep cpu --clip $CLIP --frames 10 --tag ort-cpu-default-ev-m
run ort-cpu-4t-ev-m_pexels-15170997-warehouse-team.log python $B/ort_sam.py --onnx $G --emb $O/text_embeddings.npz --ep cpu --threads 4 --clip $CLIP --frames 10 --tag ort-cpu-4t-ev-m
run ort-cuda-ev-m_pexels-15170997-warehouse-team.log python $B/ort_sam.py --onnx $G --emb $O/text_embeddings.npz --ep cuda --clip $CLIP --frames 10 --tag ort-cuda-ev-m
echo ALLDONE
