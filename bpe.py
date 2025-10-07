from pretokenization_example import *


class BPETokenizer:

    def __init__(self, **kwargs):
        self.input_path: str = kwargs.get("input_path", None)
        self.special_tokens: list[str] = kwargs.get("special_tokens", None)
        self.vocab_size: int = kwargs.get("vocab_size", None)
        print(kwargs)
        assert self.special_tokens and self.vocab_size and self.input_path

        # mapping from token ID to token bytes
        self.vocab: tuple[dict[int, bytes]] = None
        # mapping from tuples of bytes
        self.merges: list[tuple[bytes, bytes]] = None

        # utils
    
    def _regex_match(self, se):
        s, e = se
        with open(self.input_path, "rb") as f:
            f.seek(s)
            chunk = f.read(e - s).decode("utf-8", errors="ignore")
            pretoken_list = re.finditer(PAT, chunk)
            return Counter(match.group(0).encode("utf-8") for match in pretoken_list)  # return a local Counter


    def _pretokenize(self, word_counter, num_processes = 1):
        END_OF_TEXT = b"<|endoftext|>" # TODO
        with open(self.input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, END_OF_TEXT)

        with multiprocessing.Pool(processes=num_processes) as pool:
            inputs = zip(boundaries[:-1], boundaries[1:])
            outputs = pool.map(self._regex_match, inputs)

        for partial in tqdm(outputs):
            word_counter.update(partial)
            
        print("max count: ", max(word_counter, key=word_counter.get))
        print("length: ", len(word_counter))
        print(f"word_count: %s" % word_counter)


    def train(self):
        # pretokenize to get word count
        word_count = Counter()
        self._pretokenize(word_counter=word_count)

        # iterate pairs to compute max freq pairs

if __name__ == "__main__":
    bpe = BPETokenizer(
        input_path = "./data/toy.txt",
        special_tokens = [b"<|endoftext|>"],
        vocab_size = 256 
    )
    bpe.train()
