dataset=$1
if [ $1 == resume ]; then
  train_file=./data/$dataset/train.char.bmoes
  dev_file=./data/$dataset/dev.char.bmoes
  test_file=./data/$dataset/test.char.bmoes
  eps=50
  max_len=200
elif [ $1 == weibo ]; then
  train_file=./data/$dataset/train.char.bmoes
  dev_file=./data/$dataset/dev.char.bmoes
  test_file=./data/$dataset/test.char.bmoes
  eps=50
  max_len=200
elif [ $1 == ontonotes4 ]; then
  train_file=./data/$dataset/train.char.clip256.bmoes
  dev_file=./data/$dataset/dev.char.clip256.bmoes
  test_file=./data/$dataset/test.char.clip256.bmoes
  eps=50
  max_len=250
elif [ $1 == msra ]; then
  train_file=./data/$dataset/train.char.clip256.bmoes
  dev_file=./data/$dataset/test.char.clip256.bmoes
  test_file=./data/$dataset/test.char.clip256.bmoes
  eps=50
  max_len=250
else
  train_file=./data/$dataset/train.char.bmoes
  dev_file=./data/$dataset/dev.char.bmoes
  test_file=./data/$dataset/test.char.bmoes
  eps=1
  max_len=200
fi

echo "dataset=$dataset, eps=$eps"
PYTHONIOENCODING=utf-8 \
CUDA_VISIBLE_DEVICES=$4 \
python main.py --status $2 \
    --dataset $dataset \
    --char_emb /home/qiumengchuan/NLP/corpus/embedding/chinese/lexicon/gigaword_chn.all.a2b.uni.11k.50d.vec \
    --gaz_file /home/qiumengchuan/NLP/corpus/embedding/chinese/lexicon/ctb.704k.50d.vec \
		--train $train_file\
		--dev $dev_file \
		--test $test_file \
		--loadmodel ./output/ckpt/$dataset/tagscheme$3/best.model \
		--epochs $eps \
		--max_len $max_len \
		--new_tag_scheme $3 \
		--latticelstm_num 3

# python main.py --status decode \
# 		--raw ../data/onto4ner.cn/test.char.bmes \
# 		--savedset ../data/onto4ner.cn/saved_model \
# 		--loadmodel ../data/onto4ner.cn/saved_model.13.model \
# 		--output ../data/onto4ner.cn/raw.out \
