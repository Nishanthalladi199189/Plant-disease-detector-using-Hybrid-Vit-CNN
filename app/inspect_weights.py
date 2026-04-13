import zipfile, h5py, tempfile, os

# Extract weights from .keras
z = zipfile.ZipFile('trained_model/hybrid_vit_cnn_plant_disease_model.keras')
weights_file = z.extract('model.weights.h5', tempfile.gettempdir())
z.close()

# Read weight names
f = h5py.File(weights_file, 'r')

def print_weights(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"  {name}: {obj.shape}")
    elif isinstance(obj, h5py.Group):
        if 'weight_names' in obj.attrs:
            for wn in obj.attrs['weight_names']:
                print(f"  {wn}")

# Print top-level structure
print("Top-level groups:", list(f.keys()))
print("\nWeight names in model_weights:")
mw = f['model_weights']
for key in mw.keys():
    print(f"\nLayer group: {key}")
    g = mw[key]
    if 'weight_names' in g.attrs:
        for wn in g.attrs['weight_names']:
            print(f"  {wn}")
    for subkey in g.keys():
        sg = g[subkey]
        if isinstance(sg, h5py.Group) and 'weight_names' in sg.attrs:
            for wn in sg.attrs['weight_names']:
                print(f"  {wn}")

f.close()
os.remove(weights_file)
