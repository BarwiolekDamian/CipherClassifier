import urllib.request
import os
import numpy as np
from ciphers_reduced import (encrypt_aes_reduced, encrypt_des_reduced, encrypt_3des_reduced, encrypt_rc4)

def download_plaintext(url: str, file_name: str = "raw_text.txt") -> bytes:

    if not os.path.exists(file_name):
        print(f"Pobieranie tekstu z: {url}...")
        urllib.request.urlretrieve(url, file_name)
        print("Pobieranie zakończone")

    with open(file_name, 'rb') as f:
        data_bytes = f.read()

    print(f"Wczytano {len(data_bytes) / 1024:.2f} KB surowych danych")
    return data_bytes

def propare_blocks(data_bytes: bytes, block_size: int = 1024) -> list[bytes]:
    blocks = []

    for i in range(0, len(data_bytes) - block_size + 1, block_size):
        blocks.append(data_bytes[i:i + block_size])

    print(f"Wygenerowano {len(blocks)} bloków tekstowych (każdy po {block_size} bajtów).")
    return blocks

if __name__ == "__main__":
    TEXT_URL = "https://wolnelektury.pl/media/book/txt/lalka-tom-pierwszy.txt"

    raw_bytes = download_plaintext(TEXT_URL)
    text_blocks = propare_blocks(raw_bytes, block_size = 1024)

    if not text_blocks:
        raise ValueError("Brak bloków tekstowych do przetworzenia!")

    os.makedirs("data/ciphertexts", exist_ok = True)

    X_data = []
    y_data = []

    print("Rozpoczynanie szyfrowania próbek...")

    samples_per_class = 5000

    block_idx = 0
    for class_id in range(4):
        for _ in range(samples_per_class):
            if block_idx >= len(text_blocks):
                block_idx = 0

            block = text_blocks[block_idx]
            block_idx += 1

            if class_id == 0:
                encrypted = encrypt_aes_reduced(block, rounds = 2)
            elif class_id == 1:
                encrypted = encrypt_des_reduced(block, rounds = 2)
            elif class_id == 2:
                encrypted = encrypt_3des_reduced(block)
            elif class_id == 3:    
                encrypted = encrypt_rc4(block)
                            
            sample_array = np.frombuffer(encrypted, dtype = np.uint8)

            X_data.append(sample_array)
            y_data.append(class_id)

    X = np.array(X_data)
    y = np.array(y_data)

    np.save("data/ciphertexts/X_ciphertexts.npy", X)
    np.save("data/ciphertexts/y_labels.npy", y)

    print(f"Wygenerowano i zapisano pomyślnie!")
    print(f"Kształt macierzy X (dane): {X.shape}")
    print(f"Kształt macierzy y (etykiety): {y.shape}")