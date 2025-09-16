from NGramLanguageModel import NGramLanguageModel


def main():
    
    n = 1   # the degree of the n-gram. e.g., unigram -> n=1
    smoothing = 1   # the degree of the smoothing. e.g., "Add 1" -> 1
    unknown_epsilon = 10 # log2 count distance below which a token is mapped to '<UNK>'
    with open('../A1_DATASET/train.txt', 'r') as f:
        train = f.readlines()
        
        with open('../A1_DATASET/val.txt', 'r') as f:
            val = f.readlines()
        
            model = NGramLanguageModel(n, smoothing, unknown_epsilon, train, val)
            pp = model.get_perplexity()
            print(pp)
    

if __name__ == "__main__":
    main()
