dataset=$3
PYTHONIOENCODING=utf-8 \
CUDA_VISIBLE_DEVICES=$1 \
python main.py --status $2 \
    --dataset $dataset \
		--train ./data/$dataset/train.char.bmes \
		--dev ./data/$dataset/dev.char.bmes \
		--test ./data/$dataset/test.char.bmes \
		--savemodel ./data/ckpt/$dataset/saved_model \
		--epochs 100 \
		--new_tag_scheme 1

# python main.py --status decode \
# 		--raw ../data/onto4ner.cn/test.char.bmes \
# 		--savedset ../data/onto4ner.cn/saved_model \
# 		--loadmodel ../data/onto4ner.cn/saved_model.13.model \
# 		--output ../data/onto4ner.cn/raw.out \
