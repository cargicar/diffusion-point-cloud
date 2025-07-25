import kagglehub

# Download latest version
path = kagglehub.dataset_download("jeremy26/shapenet-core")

print("Path to dataset files:", path)