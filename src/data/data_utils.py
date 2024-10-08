
def get_dataloaders(data_root_copy: str, batchsize: int):
    """function to create tensorflow data loaders for model training/validation/testing
    Args:
        data_root_copy (str): directory path of preprocessed raw input folder copy
        batchsize (int): batch size
    Returns:
        _type_: tensors for corresponding batched inputs/outputs
    """
    inputs = []
    outputs = []
    for filename in os.listdir(data_root_copy):
        input_, output_ = parse_input_output_pairs(data_root_copy, gold_root_copy, filename)
        inputs.append(input_)
        outputs.append(output_)

    inputs, outputs = tf.cast(tf.stack(inputs, axis=0), tf.float32), tf.cast(tf.stack(outputs, axis=0), tf.float32)
    print(type(inputs), inputs.shape)
    print(type(outputs), outputs.shape)

    gfg_inputs = tf.data.Dataset.from_tensor_slices(inputs)
    gfg_outputs = tf.data.Dataset.from_tensor_slices(outputs)
    gfg_inputs_loader = gfg_inputs.batch(batchsize)
    gfg_outputs_loader = gfg_outputs.batch(batchsize)

    return gfg_inputs_loader, gfg_outputs_loader


if __name__ == "__main__":
    data_root = r"C:/Users/takao/Desktop/denoising_collected_data/raw_imgs"
    gold_root = r"C:/Users/takao/Desktop/denoising_collected_data/gold_standards"
    preprocess(data_root, gold_root)
    print("FINISHED PREPROCESSING.")