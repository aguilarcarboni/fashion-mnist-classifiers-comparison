import numpy as np
import skimage

def get_mnist_data_and_labels(images_filename, labels_filename):
    images_file = open(images_filename, "rb")
    labels_file = open(labels_filename, "rb")
    try:
        images_file.read(4)
        num_of_items = int.from_bytes(images_file.read(4), byteorder="big")
        num_of_rows = int.from_bytes(images_file.read(4), byteorder="big")
        num_of_colums = int.from_bytes(images_file.read(4), byteorder="big")
        num_of_image_values = num_of_rows * num_of_colums
        print("Number of images: ", num_of_items)
        print("Size of images: ", num_of_rows, "by", num_of_colums)
        labels_file.read(8)
        data = [[None for x in range(num_of_image_values)]
                for y in range(num_of_items)]
        labels = []
        for item in range(num_of_items):
            if (item % 1000) == 0:      
                  print("Current image number: %7d" % item)
            for value in range(num_of_image_values):
                data[item][value] = int.from_bytes(images_file.read(1),
                                                   byteorder="big")
            labels.append(int.from_bytes(labels_file.read(1), byteorder="big"))
            
        # Convert to NumPy arrays
        data = np.array(data).astype(np.uint8)
        labels = np.array(labels).astype(np.uint8)
        return data, labels
    finally:
        images_file.close()
        labels_file.close()
        print("Complete")



def resize_data(original_size, target_size, data):

    length = data.shape[0]

    # Resize all the training images
    data_resized = np.zeros( (length, target_size**2) )
    for img_idx in range(length):

        # Get the image
        img = data[img_idx].reshape(original_size,original_size)

        # Resize the image
        img_resized = skimage.transform.resize(img, (target_size,target_size), anti_aliasing=True)

        # Put it back in vector form
        data_resized[img_idx] = img_resized.reshape(1, target_size**2)

    return data_resized
