# run at repo root path [pass]
# export RUN_SLOW=1 # uncomment to run slow tests
python -m unittest tests.models.bert.test_modeling_bert

# run a specific test case
# python -m unittest tests.models.bert.test_modeling_bert.BertModelTest.test_config
