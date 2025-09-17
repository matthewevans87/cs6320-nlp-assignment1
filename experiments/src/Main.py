from NGramLanguageModel import NGramLanguageModel


def main():
    
    n = 2   # the degree of the n-gram. e.g., unigram -> n=1
    smoothing = 1   # the degree of the smoothing. e.g., "Add 1" -> 1
    coverage = 0.99
    model:  NGramLanguageModel
    with open('../A1_DATASET/train.txt', 'r') as f:
        train = f.readlines()
        model = NGramLanguageModel(n, smoothing, coverage, train)
        
    with open('../A1_DATASET/val.txt', 'r') as f:
        val = f.readlines()
        pp = model.get_mean_perplexity(val)
        print(pp)
    

if __name__ == "__main__":
    main()
