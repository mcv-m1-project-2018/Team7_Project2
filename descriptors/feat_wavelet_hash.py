import imagehash
from PIL import Image


def get_hash(image):
    """
    Computes the hash of the image using DWT (discrete wavelet transformation) based image hashing. The basis function
    can be changed but the Haar wavelet works pretty well. The hash size is important too.
    :param image: image
    :return: DWT based hash
    """
    return imagehash.whash(Image.fromarray(image), hash_size=32)


def compare_wavelet_hashing(q_hash, im_hash):
    """
    Computes the difference between 2 hashes. These are some results with different hash sizes:
    4x4 hash size: 0.54
    8x8 hash size: 0.85
    16x16 hash size: 0.92 (~26 seconds)
    32x32 hash size: 0.92

    We based our code on https://fullstackml.com/wavelet-image-hash-in-python-3504fdd282b5
    :param q_hash: hash of the query image
    :param im_hash: hash of the image
    :return: difference between the two hashes
    """

    return (q_hash-im_hash)/len(q_hash.hash) ** 2


def retrieve_best_results(q_image, database_imgs, database_hash, K=10):

    scores = []
    q_hash = get_hash(q_image)
    for d_image, d_name in database_imgs:
        scores.append((d_name, compare_wavelet_hashing(q_hash, database_hash[d_name])))

    scores.sort(key=lambda s: s[1], reverse=False)

    return scores[:K]
