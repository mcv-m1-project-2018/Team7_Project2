import imagehash


def compare_wavelet_hashing(q_image, image):
    """
    Computes the difference between 2 images using DWT (discrete wavelet transformation) based image hashes. The basis
    function can be changed but the Haar wavelet works pretty well. The hash size is important too. These are some
    results with different hash sizes:
    4x4 hash size: 0.54
    8x8 hash size: 0.85
    16x16 hash size: 0.92
    32x32 hash size: 0.92
    TODO: retrieve execution times

    We based our code on https://fullstackml.com/wavelet-image-hash-in-python-3504fdd282b5
    :param q_image: query image
    :param image: image
    :return: difference between the two hashes
    """
    q_hash = imagehash.whash(q_image, hash_size=16)
    im_hash = imagehash.whash(image, hash_size=16)

    return (q_hash-im_hash)/len(q_hash.hash) ** 2


def retrieve_best_results(q_image, database_imgs, K=10):

    scores = []
    for d_image, d_name in database_imgs:
        scores.append((d_name, compare_wavelet_hashing(q_image, d_image)))

    scores.sort(key=lambda s: s[1], reverse=False)

    return scores[:K]
