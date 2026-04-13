import zipfile, json

z = zipfile.ZipFile('trained_model/hybrid_vit_cnn_plant_disease_model.keras')
cfg = json.loads(z.read('config.json'))
layers = cfg['config']['layers']
print(f'Total layers: {len(layers)}')
for l in layers:
    cls = l['class_name']
    name = l['config'].get('name', '')
    # Print key config params
    extra = ''
    if cls == 'CustomCNNBlock':
        extra = f"filters={l['config'].get('filters')}, kernel_size={l['config'].get('kernel_size')}, dropout_rate={l['config'].get('dropout_rate')}"
    elif cls == 'PatchEmbedding':
        extra = f"patch_size={l['config'].get('patch_size')}, embed_dim={l['config'].get('embed_dim')}"
    elif cls == 'MultiHeadSelfAttention':
        extra = f"embed_dim={l['config'].get('embed_dim')}, num_heads={l['config'].get('num_heads')}"
    elif cls == 'TransformerBlock':
        extra = f"embed_dim={l['config'].get('embed_dim')}, num_heads={l['config'].get('num_heads')}, ff_dim={l['config'].get('ff_dim')}, dropout_rate={l['config'].get('dropout_rate')}"
    elif cls == 'Dense':
        extra = f"units={l['config'].get('units')}, activation={l['config'].get('activation')}"
    elif cls == 'Reshape':
        extra = f"target_shape={l['config'].get('target_shape')}"
    elif cls == 'Dropout':
        extra = f"rate={l['config'].get('rate')}"
    elif cls == 'LayerNormalization':
        extra = f"epsilon={l['config'].get('epsilon')}"
    elif cls == 'GlobalAveragePooling2D':
        extra = ""
    elif cls == 'Flatten':
        extra = ""
    elif cls == 'Concatenate':
        extra = ""
    print(f'  {cls}: {name} {extra}')

# Also print metadata
meta = json.loads(z.read('metadata.json'))
print(f'\nMetadata: {meta}')
z.close()
