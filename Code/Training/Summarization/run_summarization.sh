export $DATA_PATH=./data
export $SUMMARIES_PATH=./output


python run_summarization.py \
    --documents_dir $DATA_PATH \
    --summaries_output_dir $SUMMARIES_PATH \
    --no_cuda false \
    --batch_size 4 \
    --min_length 50 \
    --max_length 200 \
    --beam_size 5 \
    --alpha 0.95 \
    --block_trigram true \ |& tee ./output/logs.txt