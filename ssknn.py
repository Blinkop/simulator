import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

from polara.preprocessing.dataframes import leave_one_out


def compute_similarity(type, m1, m2):
    if type == 'jaccard':
        similarity = jaccard_similarity(m1, m2)
    elif type == 'weighted_jaccard':
        similarity = weighted_jaccard_similarity(m1, m2)
    elif type == 'cosine':
        similarity = cosine_similarity(m1, m2, dense_output=False)
    else:
        raise ValueError(f'Unknown similarity type: {type}')
    return similarity

def truncate_similarity(similarity, k):
    '''
    For every row in similarity matrix, pick at most k entities
    with the highest similarity scores. Disregard everything else.
    '''
    similarity = similarity.tocsr()
    inds = similarity.indices
    ptrs = similarity.indptr
    data = similarity.data
    new_ptrs = [0]
    new_inds = []
    new_data = []
    for i in range(len(ptrs)-1):
        start, stop = ptrs[i], ptrs[i+1]
        if start < stop:
            data_ = data[start:stop]
            topk = min(len(data_), k)
            idx = np.argpartition(data_, -topk)[-topk:]
            new_data.append(data_[idx])
            new_inds.append(inds[idx+start])
            new_ptrs.append(new_ptrs[-1]+len(idx))
        else:
            new_ptrs.append(new_ptrs[-1])
    new_data = np.concatenate(new_data)
    new_inds = np.concatenate(new_inds)
    truncated = csr_matrix(
        (new_data, new_inds, new_ptrs),
        shape=similarity.shape
    )
    return truncated

def jaccard_similarity(A, B):
    '''
    Computes the jaccard similarity index between the rows of two input matrices.
    The matrices are binarized.
    Jaccard(u, v) = \frac{\sum_{i=1}^k \min(u_k, v_k)}{\sum_{i=1}^k \max(u_k, v_k)}
    
    Args:
        A (scipy.sparse.csr_matrix): n_users_A x n_items
        B (scipy.sparse.csr_matrix): n_users_B x n_items

    Returns:
        scipy.sparse.csr_matrix: A sparse matrix of shape (n_users_A, n_users_B) containing the similarities between users
    '''
    assert A.shape[1] == B.shape[1]
    A_bin = A.astype('bool').astype('int')
    B_bin = B.astype('bool').astype('int')

    numerator = A_bin @ B_bin.T
    denominator = A_bin.sum(axis=1) + B_bin.sum(axis=1).T - A_bin @ B_bin.T
    similarity = csr_matrix(numerator / denominator)
    return similarity


def weighted_jaccard_index(u, v):
    '''
    Computes weighted jaccard index between the rows of matrices u and v.
    
    '''
    numerator = u.minimum(v).sum(axis=1).A.squeeze()
    denominator = u.maximum(v).sum(axis=1).A.squeeze()
    return (numerator / denominator)


def weighted_jaccard_similarity(A, B):
    '''
    Computes the weighted jaccard similarity index between the rows of two input matrices.
    Weighted_jaccard(u, v) = \frac{\sum_{i=1}^k \min(u_k, v_k)}{\sum_{i=1}^k \max(u_k, v_k)}
    
    Args:
        A (scipy.sparse.csr_matrix): n_users_A x n_items
        B (scipy.sparse.csr_matrix): n_users_B x n_items

    Returns:
        scipy.sparse.csr_matrix: A sparse matrix of shape (n_users_A, n_users_B) containing the similarities between users
    '''
    assert A.shape[1] == B.shape[1]
    similarity = np.zeros((A.shape[0], B.shape[0]))
    for i in range(A.shape[0]):
        # construct a new matrix A_tile
        # of the same shape as B,
        # each row of matrix A_tile
        # is equal to the i-th row of matrix A
        row = csr_matrix(A[i, :])
        rows, cols = B.shape
        
        A_tile = csr_matrix((np.tile(row.data, rows), np.tile(row.indices, rows),
                                np.arange(0, rows*row.nnz + 1, row.nnz)), shape=B.shape)
        # compute the similarity between user i and users in matrix B
        similarity[i, :] = weighted_jaccard_index(A_tile, B)
    return csr_matrix(similarity)

def generate_sequential_matrix(data, data_description, rebase_users=False):
    '''
    Converts a pandas dataframe with user-item interactions into a sparse matrix representation.
    Allows reindexing user ids, which help ensure data consistency at the scoring stage
    (assumes user ids are sorted in the scoring array).

    Args:
        data (pandas.DataFrame): The input dataframe containing the user-item interactions.
        data_description (dict): A dictionary containing the data description with the following keys:
            - 'n_users' (int): The total number of unique users in the data.
            - 'n_items' (int): The total number of unique items in the data.
            - 'users' (str): The name of the column in the dataframe containing the user ids.
            - 'items' (str): The name of the column in the dataframe containing the item ids.
            - 'feedback' (str): The name of the column in the dataframe containing the user-item interaction feedback.
            - 'timestamp' (str): The name of the column in the dataframe containing the user-item interaction timestamp.
        rebase_users (bool, optional): Whether to reindex the user ids to make contiguous index starting from 0. Defaults to False.

    Returns:
        scipy.sparse.csr_matrix: A sparse matrix of shape (n_users, n_items) containing the user-item interactions with reciprocal weighting.
    '''

    data_sorted = data.sort_values(by=[data_description['order']], ascending=False)
    data_sorted['reciprocal_rank'] = (1.0 / (data_sorted.groupby(data_description['users']).cumcount() + 1))

    n_users = data_description['n_users']
    n_items = data_description['n_items']
    # get indices of observed data
    user_idx = data_sorted[data_description['users']].values
    if rebase_users: # handle non-contiguous index of test users
        # This ensures that all user ids are contiguous and start from 0,
        # which helps ensure data consistency at the scoring stage.
        user_idx, user_index = pd.factorize(user_idx, sort=True)
        n_users = len(user_index)
    item_idx = data_sorted[data_description['items']].values
    ranks = data_sorted['reciprocal_rank'].values
    # construct the matrix
    return csr_matrix((ranks, (user_idx, item_idx)), shape=(n_users, n_items), dtype='f8')

class S_SKNN:
    def __init__(self, model_config=None) -> None:
        self.similarity_type = model_config['similarity']
        self.n_neighbors = model_config['n_neighbors']

    def build(self, data, data_description):
        interactions = generate_sequential_matrix(data, data_description)
        self.interactions = interactions

    def recommend(self, test_data, data_description):
        test_interactions = generate_sequential_matrix(test_data, data_description, rebase_users=True)
        full_similarity = compute_similarity(self.similarity_type, test_interactions, self.interactions)
        similarity = truncate_similarity(similarity=full_similarity, k=self.n_neighbors)
        scores = similarity.dot(self.interactions).toarray()
        return scores
    
    def get_action_embeddings(self, data_description):
        d = pd.DataFrame({
            data_description['users'] : np.arange(data_description['n_items']),
            data_description['items'] : np.arange(data_description['n_items']),
            data_description['order'] : np.zeros(data_description['n_items']).astype(np.int64)
        })

        dummy_interactions = generate_sequential_matrix(d, data_description, rebase_users=True)
        full_similarity = compute_similarity(self.similarity_type, dummy_interactions, self.interactions)
        return full_similarity.toarray()

    def get_current_state(self, user_test_interactions, item_popularity, data_description, calculate_subseqs, top_pop=300):
        
        if calculate_subseqs and (user_test_interactions.userid.nunique() == 1):
            targets = []
            seqs = pd.DataFrame()
            
            for i in range(user_test_interactions.shape[0] - 1):
                user_test_interactions['userid'] = i
                
                user_test_interactions, target = leave_one_out(user_test_interactions, target='timestamp', sample_top=True, random_state=0)
                seqs = pd.concat([seqs, user_test_interactions])
                targets.append(target.itemid.values[0])
            
            user_test_interactions = seqs
        
        test_interactions_matrix = generate_sequential_matrix(user_test_interactions, data_description, rebase_users=True)
        
        full_similarity = compute_similarity(
            self.similarity_type, 
            test_interactions_matrix, 
            self.interactions
        )
    
        similarity = truncate_similarity(similarity=full_similarity, k=self.n_neighbors)
    
        popular_item_scores = similarity.dot(self.interactions[:, item_popularity[:top_pop].index]).toarray()
        
        return popular_item_scores
