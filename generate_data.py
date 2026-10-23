import urllib.request
import os

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

    if text_blocks:
        print(f"Długość pierwszej próbki: {len(text_blocks[0])} bajtów (oczekiwano: 1024)")