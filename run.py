from source_code.models.unet.unet_class import Unet

# Directory containing your test images
test_images_root_dir = 'source_code/data/test_set_images'

model_pth = 'source_code/models/unet/Results/model_epoch39_fold_0.pth'
# Load my best model
best_model = load_model(model_pth)


for i in range(1, 51):

    print_advancement(i)
    dir_path = os.path.join(test_images_root_dir, f'test_{i}')
    image_path = os.path.join(dir_path, f'test_{i}.png')  # Update if the naming is different

    image = preprocess_image(image_path)
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        prediction = model(image)

    # Convert prediction to an image format
    # Update this part based on your prediction format
    prediction_image = prediction.squeeze().cpu().numpy()
    binary_prediction = apply_threshold(prediction_image)

    # Convert binary prediction to image and save
    binary_prediction_image = Image.fromarray((binary_prediction * 255).astype(np.uint8))
    save_path = os.path.join(save_dir, f'prediction_{i}.png')
    binary_prediction_image.save(save_path)

submission_filename = 'dummy_submission.csv'
image_filenames = []
for i in range(1, 51):

    image_filename = f'/content/source_code/models/unet/Results/preds/prediction_{i}.png'
    image_filenames.append(image_filename)

masks_to_submission(submission_filename, *image_filenames)