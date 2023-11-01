#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import wandb  # Import the wandb library for experiment tracking
from wandb.keras import WandbCallback

def make_generator_model():
    """
    Create a generator model for a DCGAN.

    Returns:
        A tf.keras.Sequential model representing the generator.
    """
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model

def make_discriminator_model():
    """
    Create a discriminator model for a DCGAN.

    Returns:
        A tf.keras.Sequential model representing the discriminator.
    """
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

def discriminator_loss(real_output, fake_output):
    """
    Calculate the discriminator loss.

    Args:
        real_output: Real output values from the discriminator.
        fake_output: Fake output values from the discriminator.

    Returns:
        Total loss for the discriminator.
    """
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    """
    Calculate the generator loss.

    Args:
        fake_output: Fake output values from the discriminator.

    Returns:
        Loss for the generator.
    """
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Define the optimizers for generator and discriminator
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Define the training step
@tf.function
def train_step(images, generator):
    """
    Perform a single training step.

    Args:
        images: Input images from the dataset.
        generator: The generator model.

    Returns:
        None
    """
    noise = tf.random.normal([config.batch_size, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def generate_and_save_images(model, epoch, test_input, variant):
    """
    Generate images using the generator and save them.

    Args:
        model: The generator model.
        epoch: The current training epoch.
        test_input: Input noise for generating images.
        variant: The variant of the model being used.

    Returns:
        None
    """
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(f'image_at_epoch_{epoch:04d}_variant_{variant}.png')
    plt.show()

def train(dataset, epochs, generator, discriminator, save_interval, variant):
    """
    Main training loop for the DCGAN.

    Args:
        dataset: The training dataset.
        epochs: The number of training epochs.
        generator: The generator model.
        discriminator: The discriminator model.
        save_interval: Interval for saving generated images.
        variant: The variant of the model being used.

    Returns:
        None
    """
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch, generator)
        if (epoch + 1) % save_interval == 0:
            noise = tf.random.normal([16, 100])
            generate_and_save_images(generator, epoch + 1, noise, variant)

# Set the interval at which you want to save images (e.g., every 10 epochs)
save_interval = 10

# Initialize Weights and Biases for experiment tracking
wandb.init(project="dcgan_mnist")
config = wandb.config
config.epochs = 80
config.batch_size = 256

# Load the MNIST dataset
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5

BUFFER_SIZE = 60000
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(config.batch_size)

# Define variations of DCGAN architectures
generator_variant_1 = make_generator_model()
discriminator_variant_1 = make_discriminator_model()
# Modify architecture for variant 1

generator_variant_2 = make_generator_model()
discriminator_variant_2 = make_discriminator_model()
# Modify architecture for variant 2

# Define additional variants with different architectures

# Experiment loop
for variant, (generator, discriminator) in enumerate([(generator_variant_1, discriminator_variant_1), (generator_variant_2, discriminator_variant_2)]):
    # Initialize Weights and Biases for experiment tracking
    wandb.init(project=f"dcgan_mnist_variant_{variant}")

    # Train the DCGAN with the modified architecture
    train(train_dataset, config.epochs, generator, discriminator, save_interval, variant)

    # Compare and evaluate the results with the baseline
    # You can compare generated images, losses, and other relevant metrics

