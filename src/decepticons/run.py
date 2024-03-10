from ingestion import run_model

root_dir = r"C:/Users/antho/Desktop/ml4physim/src/" #TODO: Add the root directory of the project here
model = "thrust_the_process" # megatron / thrust_the_process / Jazzx4
model_path = "models/" + model

# %%
if __name__ == "__main__":
    run_model(root_dir, model_path)