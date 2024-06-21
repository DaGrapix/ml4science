from ingestion import run_model

root_dir = r"C:/Users/antho/Desktop/ml4physim/src/run/"                 # Add the path to the run directory of the project here
model_path = "bi-transformer"                                           # Use the name of the model's directory name, eg. megatron / thrust_the_process / Jazzx4 / physical_heads...
BENCHMARK_PATH = r"C:/Users/antho/Desktop/ml4physim/src/benchmark.pkl"  # Add the path to the pickled benchmark file here. If the file does not exist, it will be created there.

# %%
if __name__ == "__main__":
    run_model(root_dir, model_path, BENCHMARK_PATH)
