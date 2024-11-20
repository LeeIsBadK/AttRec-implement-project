import os
import pickle
import pandas as pd
def create_train_test(df, seq_counts=5, target_counts=3, save_dir='processed_data', is_Save=True):
    """
    Splits the dataset into train/test sets with user-item sequences.

    Args:
        data_path (str): Path to the user-item interaction data file.
        seq_counts (int): Length of input sequences.
        target_counts (int): Number of items to predict.
        save_dir (str): Directory to save the train/test data.
        is_save (bool): Whether to save the datasets and metadata.

    Returns:
        train (pd.DataFrame): Training data.
        test (pd.DataFrame): Testing data.
        user_all_items (dict): Mapping of users to their full item interaction lists.
        all_user_count (int): Total number of unique users.
        all_item_count (int): Total number of unique items.
        user_map (dict): Mapping of original user IDs to remapped IDs.
        item_map (dict): Mapping of original item IDs to remapped IDs.
    """
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # # Load data
    data = df.copy()
    # Remap user and item IDs to start from 1
    user_map = {uid: i for i, uid in enumerate(data['userID'].unique())}
    item_map = {iid: i for i, iid in enumerate(data['itemID'].unique())}
    data['userID'] = data['userID'].map(user_map)
    data['itemID'] = data['itemID'].map(item_map)

    # Sort data by user and timestamp
    data = data.sort_values(by=['userID', 'timestamp']).reset_index(drop=True)

    # Group data by user
    user_sessions = data.groupby('userID')['itemID'].apply(list).reset_index()
    user_sessions.rename(columns={'itemID': 'item_list'}, inplace=True)

    user_all_items = {}
    train_users, train_seqs, train_targets = [], [], []
    test_users, test_seqs, test_targets = [], [], []

    for _, row in user_sessions.iterrows():
        user = row['userID']
        items = row['item_list']
        user_all_items[user] = items

        # Create training sequences
        for i in range(seq_counts, len(items) - target_counts):
            seqs = items[i - seq_counts:i]
            targets = items[i:i + target_counts]
            train_users.append(user)
            train_seqs.append(seqs)
            train_targets.append(targets)

        # Create testing sequence
        if len(items) > seq_counts + target_counts:
            test_seq = items[-seq_counts - target_counts:-target_counts]
            test_target = items[-target_counts:]
            test_users.append(user)
            test_seqs.append(test_seq)
            test_targets.append(test_target)

    # Convert to DataFrames
    train = pd.DataFrame({'user': train_users, 'seq': train_seqs, 'target': train_targets})
    test = pd.DataFrame({'user': test_users, 'seq': test_seqs, 'target': test_targets})

    # Metadata
    all_user_count = len(user_map)
    all_item_count = len(item_map)

    if is_Save:
        # Save datasets
        train.to_csv(os.path.join(save_dir, 'train.csv'), index=False)
        test.to_csv(os.path.join(save_dir, 'test.csv'), index=False)

        # Save mappings and metadata
        with open(os.path.join(save_dir, 'info.pkl'), 'wb') as f:
            pickle.dump(user_all_items, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_user_count, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_item_count, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(user_map, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(item_map, f, pickle.HIGHEST_PROTOCOL)

        print(f"Train and test datasets saved in '{save_dir}'")

    return train, test, user_all_items, all_user_count, all_item_count, user_map, item_map
