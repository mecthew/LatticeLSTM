dataset=$1
if [ $1 == demo -o $1 == weibo -o $1 == resume ]; then
  train_file=./data/$dataset/train.char.bmoes
  dev_file=./data/$dataset/dev.char.bmoes
  test_file=./data/$dataset/test.char.bmoes
  eps=20
elif [ $1 == ontonotes4 ]; then
  train_file=./data/$dataset/train.char.clip256.bmoes
  dev_file=./data/$dataset/dev.char.clip256.bmoes
  test_file=./data/$dataset/test.char.clip256.bmoes
  eps=10
else
  train_file=./data/$dataset/train.char.clip256.bmoes
  dev_file=./data/$dataset/test.char.clip256.bmoes
  test_file=./data/$dataset/test.char.clip256.bmoes
  eps=10
fi

echo "dataset=$dataset, eps=$eps"
PYTHONIOENCODING=utf-8 \
CUDA_VISIBLE_DEVICES=$4 \
python main.py --status $2 \
    --dataset $dataset \
    --char_emb /home/qiumengchuan/NLP/corpus/embedding/chinese/lexicon/gigaword_chn.all.a2b.uni.11k.50d.vec \
    --gaz_file /home/qiumengchuan/NLP/corpus/embedding/chinese/lexicon/ctb.704k.50d.vec \
		--train ./data/$dataset/train.char.bmoes \
		--dev ./data/$dataset/dev.char.bmoes \
		--test ./data/$dataset/test.char.bmoes \
		--loadmodel ./output/ckpt/$dataset/best_.model \
		--epochs $eps \
		--new_tag_scheme $3

# python main.py --status decode \
# 		--raw ../data/onto4ner.cn/test.char.bmes \
# 		--savedset ../data/onto4ner.cn/saved_model \
# 		--loadmodel ../data/onto4ner.cn/saved_model.13.model \
# 		--output ../data/onto4ner.cn/raw.out \
