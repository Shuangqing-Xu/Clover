import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import numpy as np
from Crypto.Cipher import AES
import time

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_privacy.privacy.dp_query import test_utils
from scipy.stats import ks_2samp

from distributed_dp import discrete_gaussian_utils
from distributed_dp import distributed_discrete_gaussian_query

class DistributedDiscreteGaussianQueryTest(tf.test.TestCase,
                                        parameterized.TestCase):


    def local_noise_sampling(self, scaling, d):
        sigma = 0.95 / np.sqrt(2)
        local_scale = scaling * sigma
        record = tf.zeros([d], dtype=tf.int32)

        local_noise = discrete_gaussian_utils.sample_discrete_gaussian(
            scale=tf.cast(tf.round(local_scale), record.dtype),
            shape=tf.shape(record),
            dtype=record.dtype)
        return local_noise
    
    def reference_noise_sampling(self, scaling, d):
        sigma = 0.95 / np.sqrt(2)
        local_scale = scaling * sigma
        record = tf.zeros([d], dtype=tf.int32)

        central_scale = np.sqrt(2) * local_scale 

        reference_noise = discrete_gaussian_utils.sample_discrete_gaussian(
            scale=tf.cast(tf.round(central_scale), record.dtype),
            shape=tf.shape(record),
            dtype=record.dtype)
        return reference_noise
    
            
    def local_ks_test(self, scaling, d, num_records):
        sigma = 0.95
        local_scale = scaling * sigma
        record = tf.zeros([d], dtype=tf.int32)
        sample = [record] * num_records
        query = ddg_sum_query(l2_norm_bound=1.0, local_scale=local_scale)
        query_result, _ = test_utils.run_query(query, sample)

        central_scale = np.sqrt(num_records) * local_scale 
        central_noise = discrete_gaussian_utils.sample_discrete_gaussian(
            scale=tf.cast(tf.round(central_scale), record.dtype),
            shape=tf.shape(record),
            dtype=record.dtype)

        agg_noise, central_noise = self.evaluate([query_result, central_noise])
        agg_noise, central_noise = agg_noise/ tf.cast(scaling, agg_noise.dtype), central_noise/ tf.cast(scaling, central_noise.dtype)

        
        ks_statistic, p_value = ks_2samp(agg_noise, central_noise)

ddg_sum_query = distributed_discrete_gaussian_query.DistributedDiscreteGaussianSumQuery

BASE = 2

PRECISION_INTEGRAL = 0
PRECISION_FRACTIONAL = 15

Q = 2**61-1 
# Q = 2**64 - 2**32 + 1
PRECISION = PRECISION_INTEGRAL + PRECISION_FRACTIONAL

assert(Q > BASE**PRECISION)

def encode(rational):
    if isinstance(rational, np.ndarray):
        upscaled = (rational * BASE**PRECISION_FRACTIONAL).astype(np.int64)  
        field_element = upscaled % Q

        return field_element.tolist()  
    else:

        upscaled = int(rational * BASE**PRECISION_FRACTIONAL)
        field_element = upscaled % Q

        return int(field_element)  

def decode(field_element):
    upscaled = field_element if field_element <= Q/2 else field_element - Q
    rational = upscaled / BASE**PRECISION_FRACTIONAL
    return rational

def share(secret):
    first  = random.randrange(Q)
    second = random.randrange(Q)
    third  = (secret - first - second) % Q
    return [first, second, third]

def share_vector(secret_array):

    n = len(secret_array)
    first = np.random.randint(0, Q, n, dtype=np.int64)
    second = np.random.randint(0, Q, n, dtype=np.int64)
    third = (secret_array - first - second) % Q

    return np.stack([first, second, third], axis=1)  

def reconstruct(sharing):
    return sum(sharing) % Q

def reshare(x):
    Y = [ share(x[0]), share(x[1]), share(x[2]) ]
    return [ sum(row) % Q for row in zip(*Y) ]

def add(x, y):
    return [ (xi + yi) % Q for xi, yi in zip(x, y) ]

def sub(x, y):
    return [ (xi - yi) % Q for xi, yi in zip(x, y) ]
    
def imul(x, k):
    return [ (xi * k) % Q for xi in x ]

def aes_ctr_prg(key, counter, n):
    cipher = AES.new(key, AES.MODE_CTR, nonce=b"", initial_value=counter.to_bytes(16, byteorder="big"))

    random_bytes = cipher.encrypt(b"\x00" * (16 * n))  
    random_numbers = np.frombuffer(random_bytes, dtype=np.uint64)[:n]

    random_numbers_int = [int(num) % Q for num in random_numbers]
    

    return random_numbers_int  

def compute_mac(encoded_x, mac_key):
    assert len(encoded_x) == len(mac_key), "encoded_x length != mac_key"
    

    mac = [(key * value) % Q for key, value in zip(mac_key, encoded_x)]
    return mac


INVERSE = 70368744177664 
# INVERSE = pow(2**15, Q-2, Q)

KAPPA = 6 

assert((INVERSE * BASE**PRECISION_FRACTIONAL) % Q == 1)
assert(Q > BASE**(2*PRECISION + KAPPA))

def truncate(a):

    b = add(a, [BASE**(2*PRECISION+1), 0, 0])

    mask = random.randrange(Q) % BASE**(PRECISION + PRECISION_FRACTIONAL + KAPPA)
    mask_low = mask % BASE**PRECISION_FRACTIONAL
    b_masked = reconstruct(add(b, [mask, 0, 0]))

    b_masked_low = b_masked % BASE**PRECISION_FRACTIONAL
    b_low = sub(share(b_masked_low), share(mask_low))

    c = sub(a, b_low)

    d = imul(c, INVERSE)
    return d

def truncate_secure_ml(a):
    d = [0,0,0,]
    for i in range(3):
        sign = 1
        tmp = a[i]
        if i > Q // 2:
            sign = -1
            tmp = - a[i]
        d[i] = tmp* INVERSE % Q
        d[i] = d[i]*sign % Q
    return d


def mul(x, y):

    z0 = ((int(x[0]) * int(y[0]) + int(x[0]) * int(y[1]) + int(x[1]) * int(y[0])) % Q)
    z1 = ((int(x[1]) * int(y[1]) + int(x[1]) * int(y[2]) + int(x[2]) * int(y[1])) % Q)
    z2 = ((int(x[2]) * int(y[2]) + int(x[2]) * int(y[0]) + int(x[0]) * int(y[2])) % Q)
    

    Z = [share(z0), share(z1), share(z2)]
    w = [sum(row) % Q for row in zip(*Z)]
    

    v = truncate_secure_ml(w)
    return v

def truncate_secure_ml_np(a):
    d = np.zeros_like(a, dtype=int)  

    for i in range(a.shape[0]):  
        for j in range(a.shape[1]):  
            tmp = a[i, j]
            sign = 1

            if tmp > Q // 2:  
                sign = -1
                tmp = -tmp

            d[i, j] = (tmp * INVERSE) % Q
            d[i, j] = (d[i, j] * sign) % Q

    return d


def mul_np(x, y):
    x = np.array(x, dtype=object)
    y = np.array(y, dtype=object)

    z0 = (x[:, 0] * y[:, 0] + x[:, 0] * y[:, 1] + x[:, 1] * y[:, 0]) % Q
    z1 = (x[:, 1] * y[:, 1] + x[:, 1] * y[:, 2] + x[:, 2] * y[:, 1]) % Q
    z2 = (x[:, 2] * y[:, 2] + x[:, 2] * y[:, 0] + x[:, 0] * y[:, 2]) % Q

    Z = np.array([(z0), (z1), (z2)]).T  
    random_masks = np.random.randint(0, Q, (len(x), 3), dtype=np.int64)

    random_masks[:, 2] = (-random_masks[:, 0] - random_masks[:, 1]) % Q  
    reshared = (Z + random_masks) % Q

    v = truncate_secure_ml_np(reshared)

    return v


def apply_permutation(x, pi):
    sorted_indices = np.argsort(pi)  
    return x[sorted_indices]  


def inverse_permutation(pi):
    sorted_indices = np.argsort(pi)

    inv_pi = np.empty_like(sorted_indices)
    inv_pi[np.arange(len(pi))] = sorted_indices
    return inv_pi

def offline_generate_random_masks(n, num_masks, Q):
    random_masks = [np.random.randint(0, Q, (n, 3), dtype=np.int64) for _ in range(num_masks)]
    for mask in random_masks:
        mask[:, 2] = (-mask[:, 0] - mask[:, 1]) % Q  
    return random_masks

def offline_generate_random_masks_malicious_shuffle(n):
    row1 = np.random.randint(0, Q, size=(n, 2), dtype=np.int64)
    row2 = np.random.randint(0, Q, size=(n, 2), dtype=np.int64)

    row3 = (-row1 - row2) % Q

    random_masks = np.stack([row1, row2, row3], axis=1)
    
    return random_masks


def semi_shuffle(pi, a, random_masks):
    shares = np.array(a.shares)  
    reshuffled_shares = np.empty_like(shares)  

    def apply_shuffle_and_reshare(shares, perm, random_mask):
        for col in range(3):  
            reshuffled_shares[:, col] = apply_permutation(shares[:, col], perm)

        reshared = (reshuffled_shares + random_mask) % Q  
        return reshared

    for idx, p in enumerate(pi):
        shares = apply_shuffle_and_reshare(shares, p, random_masks[idx])

    z = SecureRationalVector()
    z.shares = shares  
    return z

def malicious_shuffle(pi, combined_groups, random_masks):
    def apply_shuffle_and_reshare(combined_groups, perm, random_mask):
        shuffled_combined_groups = apply_permutation(combined_groups, perm)

        reshared = (shuffled_combined_groups + random_mask) % Q  
        return reshared

    for idx, p in enumerate(pi):
        shares = apply_shuffle_and_reshare(combined_groups, p, random_masks)

    z = SecureRationalVector()
    z.shares = shares  
    return z


def combine_and_shuffle(pi, x, mac, random_masks):
    x_shares = np.array(x.shares, dtype=object)  
    mac_shares = np.array(mac.shares, dtype=object)  

    combined = np.stack((x_shares, mac_shares), axis=-1)

    combined_groups = combined.reshape(-1, 3, 2)

    shuffled_combined = malicious_shuffle(pi, combined_groups, random_masks)

    shuffled_combined_array = np.array(shuffled_combined.shares, dtype=object)

    shuffled_x_array = shuffled_combined_array[:, :, 0]  
    shuffled_mac_array = shuffled_combined_array[:, :, 1]  

    shuffled_x = SecureRationalVector()
    shuffled_x.shares = shuffled_x_array.tolist()
    shuffled_mac_key = SecureRationalVector()
    shuffled_mac_key.shares = shuffled_mac_array.tolist()

    return shuffled_x, shuffled_mac_key


def mod_sum_manual(arr):
    arr = np.array(arr)
    column_sums = [0] * arr.shape[1]  

    for i in range(arr.shape[0]):  
        for j in range(arr.shape[1]):  
            column_sums[j] += int(arr[i, j])  

    return [sum_value % Q for sum_value in column_sums]


class SecureRationalVector:
    def __init__(self):
        self.shares = None

    @staticmethod
    def secure(secret):
        z = SecureRationalVector()
        if isinstance(secret, (int, float)):
            secret = share(encode(secret))
            z.shares = secret.shares
        else:
            vector_shares = share_vector(encode(secret))
            z.shares = vector_shares
        return z

    def reveal(self):
        decoded_values = [decode(reconstruct(sharing)) for sharing in self.shares]
        return np.array(decoded_values)

    def __repr__(self):
        return f"SecureRationalVector({self.reveal()})"

    def __add__(x, y):
        assert x.shares.shape == y.shares.shape, "Shapes must match for addition"
        z = SecureRationalVector()
        z.shares = add(x.shares, y.shares)
        return z

    def __sub__(x, y):
        z = SecureRationalVector()
        z.shares = sub(x.shares, y.shares)
        return z

    def __mul__(x, y):
        z = SecureRationalVector()
        z.shares = mul_np(x.shares, y.shares)
        return z

    def __pow__(x, e):
        z = SecureRationalVector.secure(np.ones(x.shares.shape[0]))
        for _ in range(e):
            z = z * x
        return z

def generate_mac_key_shares(aes_keys, counter, d):
    return np.array([aes_ctr_prg(aes_keys[i], counter, d) for i in range(3)])


def shuffle_mali(d, alpha):
    k = round(alpha*d)

    x = np.random.randint(0, 100, size=d)

    pi_1 = np.random.permutation(d)
    pi_2 = np.random.permutation(d)
    pi_3 = np.random.permutation(d)
    pi = [pi_1, pi_2, pi_3]

    d = len(x)
    shared_x = SecureRationalVector.secure(x)
    encoded_x = encode(x)

    aes_key_0 = b"thisisakey123456"  
    aes_key_1 = b"thisisakey123213"  
    aes_key_2 = b"thisisakey432656"  
    counter = 0
    mac_key_share_0 = aes_ctr_prg(aes_key_0, counter, d)
    mac_key_share_1 = aes_ctr_prg(aes_key_1, counter, d)
    mac_key_share_2 = aes_ctr_prg(aes_key_2, counter, d)
    shared_mac_key = SecureRationalVector()
    shared_mac_key.shares = np.stack([mac_key_share_0, mac_key_share_1, mac_key_share_2], axis=1)
    mac_key = [(mac_key_share_0[i] + mac_key_share_1[i] + mac_key_share_2[i]) % Q 
            for i in range(len(mac_key_share_0))]
    proc_x = [(encoded_x[i]) // (BASE ** PRECISION_FRACTIONAL) for i in range(len(encoded_x))]
    mac = compute_mac(proc_x, mac_key)
    mac_sum = sum(mac)%Q
    shared_mac_sum = SecureRationalVector()
    shared_mac_sum.shares = share(mac_sum)
    d = len(x)  
    num_masks = len(pi)  
    random_masks = offline_generate_random_masks_malicious_shuffle(d)
    shared_pi_x, shared_pi_mac_key = combine_and_shuffle(pi, shared_x, shared_mac_key, random_masks)
    shared_mac_server = shared_pi_x * shared_pi_mac_key
    shared_mac_sum_server = SecureRationalVector()
    shared_mac_sum_server.shares = mod_sum_manual(shared_mac_server.shares)
    shared_flag = SecureRationalVector()
    shared_flag.shares = [(shared_mac_sum_server.shares[i] - shared_mac_sum.shares[i])%Q for i in range(3)]
    test_instance = DistributedDiscreteGaussianQueryTest()
    test_instance.local_noise_sampling(2**PRECISION_FRACTIONAL, d)
    test_instance.local_ks_test(2**PRECISION_FRACTIONAL, d, 2)
    print(sum(shared_flag.shares)%Q)
shuffle_mali(1000, 0.01)
