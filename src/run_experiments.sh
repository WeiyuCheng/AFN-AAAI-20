#!/bin/bash

filename="AFN";

# run AFN+
python -u ./${filename}.py --hidden_size=600 --data_dir=../data/frappe/ --model_dir=./checkpoint/frappe/ --field_size=10 --feature_size=5500 --instance_size=288609;
python -u ./${filename}.py --hidden_size=800 --data_dir=../data/movielens/ --model_dir=./checkpoint/movielens/ --field_size=3 --feature_size=92000 --instance_size=2006859;
python -u ./${filename}.py --hidden_size=1500 --data_dir=../data/criteo/ --model_dir=./checkpoint/criteo/ --field_size=39 --feature_size=2100000 --instance_size=45840617;
python -u ./${filename}.py --hidden_size=1200 --data_dir=../data/avazu/ --model_dir=./checkpoint/avazu/ --field_size=22 --feature_size=1600000 --instance_size=40428967;


## run AFN
python -u ./${filename}.py --ensemble=False --hidden_size=600 --data_dir=../data/frappe/ --model_dir=./checkpoint/frappe/ --field_size=10 --feature_size=5500 --instance_size=288609;
python -u ./${filename}.py --ensemble=False --hidden_size=800 --data_dir=../data/movielens/ --model_dir=./checkpoint/movielens/ --field_size=3 --feature_size=92000 --instance_size=2006859;
python -u ./${filename}.py --ensemble=False --hidden_size=1500 --data_dir=../data/criteo/ --model_dir=./checkpoint/criteo/ --field_size=39 --feature_size=2100000 --instance_size=45840617;
python -u ./${filename}.py --ensemble=False --hidden_size=1200 --data_dir=../data/avazu/ --model_dir=./checkpoint/avazu/ --field_size=22 --feature_size=1600000 --instance_size=40428967;


