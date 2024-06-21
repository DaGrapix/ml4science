from ingestion import run_model

root_dir = r"C:/Users/antho/Desktop/ml4physim/src/run/"                 # Add the path to the run directory of the project here
model_name = "bi-transformer"                                           # Use the name of the model's directory name, eg. bi_transformer
model_path = "models/" + model_name
BENCHMARK_PATH = r"C:/Users/antho/Desktop/ml4physim/src/benchmark.pkl"  # Add the path to the pickled benchmark file here. If the file does not exist, it will be created there.

# %%
if __name__ == "__main__":
    run_model(root_dir, model_path, BENCHMARK_PATH)
