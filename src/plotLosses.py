import json
import os
import pandas as pd
import matplotlib.pyplot as plt

# Folder containing the JSON files with loss data
loss_data_folder = "/home/msai/riemerpi001/finalNNModels/BCE"

# Function to load loss data from files
def load_loss_data(folder_path):
    loss_dicts = []
    names = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            # Get the full file path
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                loss_data = json.load(file)
                loss_dicts.append(loss_data)
                # Use the file name without extension as the model name
                model_name = os.path.splitext(file_name)[0]
                names.append(model_name)

    return loss_dicts, names

# Function to combine loss data
def combine_losses(loss_dicts, names):
    combined_data = {'Epoch': list(range(1, len(loss_dicts[0]["train_losses"]) + 1))}
    
    for i, loss_dict in enumerate(loss_dicts):
        name = names[i]
        combined_data[f'{name}_train_losses'] = loss_dict['train_losses']
        combined_data[f'{name}_val_losses'] = loss_dict['val_losses']
    
    # Convert to DataFrame for better visualization
    df = pd.DataFrame(combined_data)
    return df

# Load loss data from the specified folder
loss_dicts, names = load_loss_data(loss_data_folder)

# Combine the losses from different files
combined_df = combine_losses(loss_dicts, names)

# Save or display the combined data
combined_df.to_csv("combined_losses.csv", index=False)
print(combined_df)

# Plot the training losses
plt.figure(figsize=(10, 6))
for col in combined_df.columns:
    if 'train_losses' in col:
        plt.plot(combined_df['Epoch'], combined_df[col], label=col)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.grid(True)
plt.savefig("combined_train_losses_plot.png")
plt.show()

# Plot the validation losses
plt.figure(figsize=(10, 6))
for col in combined_df.columns:
    if 'val_losses' in col:
        plt.plot(combined_df['Epoch'], combined_df[col], label=col)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Losses')
plt.legend()
plt.grid(True)
plt.savefig("BCE_combined_val_losses_plot.png")
plt.show()
