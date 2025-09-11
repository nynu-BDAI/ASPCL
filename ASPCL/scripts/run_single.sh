export CUDA_VISIBLE_DEVICES=0
train_data='../generated_data/coco/'
upper_bound=10
threshold_type=mean # fixed or mean
fixed_threshold_value=10
lr=5e-06
bs=128
cd ../src/

for cmr_weight in 0.3
do
    for imc_weight in 0.3
    do
        if [ "$threshold_type" == "fixed" ]; then
            output_name=coco_hn_imc$imc_weight-atr$cmr_weight-fixed$fixed_threshold_value-$lr
        else
            output_name=coco_hn_imc$imc_weight-cmr$cmr_weight-mean-ub$upper_bound-$lr-test_single
        fi
        output_file=./Outputs/$output_name

        if [[ -d "$output_file" ]];then
            echo "$output_name already exists"
        else
            echo "running $output_name"
            python main.py \
            --wandb-project-name open_clip \
            --train-data $train_data \
            --seed 42 \
            --dataset-type npy \
            --save-frequency 1 \
            --report-to wandb \
            --warmup 50 \
            --batch-size $bs \
            --lr $lr \
            --wd 0.1 \
            --epochs 1 \
            --workers 32 \
            --pretrained openai \
            --model ViT-B-32 \
            --logs Outputs \
            --beta1 0.9 \
            --precision amp \
            --beta2 0.98 \
            --eps 1e-06 \
            --log-every-n-steps 10 \
            --imc-loss \
            --cmr-loss \
            --hardnegative \
            --threshold-type $threshold_type \
            --fixed-threshold-value $fixed_threshold_value \
            --cmr-loss-weight $cmr_weight \
            --imc-loss-weight $imc_weight \
            --positive-margin-loss \
            --positive-margin-loss-weight 0.3 \
            --upper-bound $upper_bound \
            --analogy-loss \
            --analogy-loss-weight 0.3 \
            --save-frequency -1
            --name $output_name

            # Check if the training command was successful
          #  if [ $? -ne 0 ]; then
           #     echo "Training failed. Cleaning up..."
                # Delete the output folder
            #    rm -rf $output_file
                # Remove the output name from the file
           # fi
        fi
    done
done