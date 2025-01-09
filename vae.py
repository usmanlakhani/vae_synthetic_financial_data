# Step 1: Import needed Python libraries.
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import sys

def initiate(startDate, endDate, latSpaceAttributes,ep,numSamples):

    # Step 2: Gather input data.
    #start_date = '2010-01-01'
    start_date = startDate
    #end_date   = '2025-01-01'
    end_date = endDate

    data = np.array((pdr.get_data_fred('SP500', start = start_date, end = end_date)).dropna())

    # Step 3: Initiate the encoder.
    latent_space_attributes = int(latSpaceAttributes)

    encoder = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(len(data),)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(latent_space_attributes, activation='relu'),
        ]
    )

    # Step 4: Configure the decoder.
    decoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(latent_space_attributes,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),    
        tf.keras.layers.Dense(len(data), activation='linear'),
    ])

    # Step 5: Wire up the VAE model.
    vae = tf.keras.Model(encoder.inputs, decoder(encoder.outputs))

    # Step 6: Compile the model.
    vae.compile(optimizer = 'adam', loss = 'mse')

    # Step 7: Normalize the input data array.
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    normalized_data = normalized_data.reshape(1, -1)

    # Step 8: Train the model.
    history = vae.fit(normalized_data, normalized_data, epochs = int(ep), verbose = 2)

    # Step 9: Test the model.
    # Step 9a: Create a numpy array that will hold 3 rows and 2 columns of data
    num_samples = int(numSamples)
    random_latent_vectors = np.random.normal(size=(num_samples, latent_space_attributes))

    # Step 9b: Generate synthetic data
    decoded_data = decoder.predict(random_latent_vectors)
    decoded_data = decoded_data * (np.max(data) - np.min(data)) + np.min(data)

    # Step 10: Visualize the results.
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(data, label = 'Original Data', color = 'black')
    axs[0].legend()
    axs[0].grid()
    for i in range(num_samples):
        plt.plot(decoded_data[i], label = f'Synthetic Generated Data {i+1}', 
                linewidth = 1)
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Original (black) vs. Synthetic Generated Time Series Data (colored)')
    plt.show()
    plt.grid()

if __name__ == "__main__":
    
    startDate = sys.argv[1]
    endDate = sys.argv[2]
    latSpaceAttributes = sys.argv[3]
    ep = sys.argv[4]
    numSamples = sys.argv[5]
    initiate(startDate, endDate, latSpaceAttributes,ep,numSamples)
