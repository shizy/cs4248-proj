# Instructions
- Ensure that `torch` and `transformers==4.56.1` and `accelerate` are installed (preferably within a virtual environment)
- Download the `dev-v1.1.json` and `train-v1.1.json` files into the `./data` directory
- Run `train.py` without any arguments (will be improved on)

# Todos
1. Add argparse into the train script to allow for hyperparameter/arch changes via the command line
2. Abstract out the linear layer in the model to make it extensible. Make it modifiable in the training args
3. Move the dataset creation logic into its own file so that we can convert the dataset once and save it in persistent memory instead of converting it once every time we run training
4. Support more interesting LR scheduling algorithms. Abstract that out to be modifable in the training args
5. Support more interesting losses. Abstract that out to be modifiable in the training args