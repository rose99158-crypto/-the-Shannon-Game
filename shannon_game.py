import nltk
import re
from collections import Counter
from nltk.corpus import brown
from collections import defaultdict

# Download Brown corpus if not already downloaded
nltk.download('brown')

class NgramModel:
    def __init__(self, n=1):
        self.n = n
        self.ngram_counts = defaultdict(Counter)
        self.total_words = 0
        
    def train(self, corpus):
        """Train the n-gram model on a corpus"""
        for sentence in corpus:
            words = ['<s>'] * (self.n - 1) + sentence + ['</s>']
            
            for i in range(len(words) - self.n + 1):
                context = tuple(words[i:i+self.n-1])
                next_word = words[i+self.n-1]
                
                self.ngram_counts[context][next_word] += 1
                self.total_words += 1
                
    def predict_next_word(self, context, top_k=5):
        """Predict next words given context"""
        # Convert context string to tuple
        if isinstance(context, str):
            context_words = context.lower().split()
        else:
            context_words = context
            
        # Take only last n-1 words for context
        if len(context_words) >= self.n - 1:
            context_tuple = tuple(context_words[-(self.n-1):])
        else:
            # Pad with start tokens if context is too short
            padding = ['<s>'] * (self.n - 1 - len(context_words))
            context_tuple = tuple(padding + context_words)
            
        # Get predictions for this context
        if context_tuple in self.ngram_counts:
            predictions = self.ngram_counts[context_tuple].most_common(top_k)
            return [(word, count) for word, count in predictions]
        
        return []
    
    def get_probability(self, context, next_word):
        """Get probability of next word given context"""
        if isinstance(context, str):
            context_words = context.lower().split()
        else:
            context_words = context
            
        if len(context_words) >= self.n - 1:
            context_tuple = tuple(context_words[-(self.n-1):])
        else:
            padding = ['<s>'] * (self.n - 1 - len(context_words))
            context_tuple = tuple(padding + context_words)
            
        if context_tuple in self.ngram_counts:
            total = sum(self.ngram_counts[context_tuple].values())
            return self.ngram_counts[context_tuple][next_word] / total
        
        return 0.0

def preprocess_text(text):
    """Simple preprocessing: lowercase and split into words"""
    # Remove punctuation and lowercase
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # Split into words
    words = text.split()
    return words

def train_models():
    """Train unigram, bigram, and trigram models on Brown corpus"""
    print("Training models on Brown Corpus...")
    
    # Get sentences from Brown corpus
    sentences = brown.sents()
    
    # Create and train models
    unigram = NgramModel(n=1)
    bigram = NgramModel(n=2)
    trigram = NgramModel(n=3)
    
    unigram.train(sentences)
    bigram.train(sentences)
    trigram.train(sentences)
    
    print(f"Models trained on {len(sentences)} sentences")
    print(f"Unigram vocabulary size: {len(unigram.ngram_counts[()])}")
    print(f"Bigram contexts: {len(bigram.ngram_counts)}")
    print(f"Trigram contexts: {len(trigram.ngram_counts)}")
    
    return unigram, bigram, trigram

def run_shannon_game(contexts, unigram, bigram, trigram):
    """Run Shannon Game for given contexts"""
    results = []
    
    for context in contexts:
        # Preprocess context
        context_words = preprocess_text(context)
        context_str = ' '.join(context_words)
        
        # Get predictions from each model
        uni_pred = unigram.predict_next_word(context_words)
        bi_pred = bigram.predict_next_word(context_words)
        tri_pred = trigram.predict_next_word(context_words)
        
        # Format predictions
        uni_top5 = [word for word, _ in uni_pred[:5]] if uni_pred else []
        bi_top5 = [word for word, _ in bi_pred[:5]] if bi_pred else []
        tri_top5 = [word for word, _ in tri_pred[:5]] if tri_pred else []
        
        # Pad if less than 5 predictions
        uni_top5 += [''] * (5 - len(uni_top5))
        bi_top5 += [''] * (5 - len(bi_top5))
        tri_top5 += [''] * (5 - len(tri_top5))
        
        results.append({
            'context': context_str,
            'unigram': uni_top5,
            'bigram': bi_top5,
            'trigram': tri_top5
        })
    
    return results

def print_results_table(results):
    """Print results in a formatted table"""
    print("\n" + "="*100)
    print("SHANNON GAME RESULTS")
    print("="*100)
    print(f"{'Context':<25} | {'Unigram':<35} | {'Bigram':<35} | {'Trigram':<35}")
    print("-"*100)
    
    for result in results:
        context = result['context']
        if len(context) > 24:
            context = context[:21] + "..."
        
        uni_str = ', '.join(result['unigram']) if any(result['unigram']) else 'No predictions'
        bi_str = ', '.join(result['bigram']) if any(result['bigram']) else 'No predictions'
        tri_str = ', '.join(result['trigram']) if any(result['trigram']) else 'No predictions'
        
        print(f"{context:<25} | {uni_str:<35} | {bi_str:<35} | {tri_str:<35}")
    
    print("="*100)

def analyze_models(results, example_contexts):
    """Analyze and compare model performance"""
    print("\n" + "="*100)
    print("MODEL ANALYSIS")
    print("="*100)
    
    print("\n1. WHICH PREDICTIONS FEEL MORE 'REASONABLE'?")
    print("-" * 50)
    print("Bigram and trigram models generally provide more 'reasonable' predictions")
    print("because they consider context. Unigram only predicts frequent words")
    print("without regard to context, which can be nonsensical.")
    
    print("\nExamples:")
    for i, context in enumerate(example_contexts):
        print(f"\nContext: '{context}'")
        print(f"Unigram: {results[i]['unigram']}")
        print(f"Bigram:  {results[i]['bigram']}")
        print(f"Trigram: {results[i]['trigram']}")
        
        # Show which model might be best
        trigram_preds = [p for p in results[i]['trigram'] if p]
        bigram_preds = [p for p in results[i]['bigram'] if p]
        
        if trigram_preds:
            print(f"Trigram suggests plausible continuations: {trigram_preds}")
        elif bigram_preds:
            print(f"Bigram suggests plausible continuations: {bigram_preds}")
        else:
            print("No contextual predictions available")
    
    print("\n2. WHICH MODEL BETTER CAPTURES LOCAL CONTEXT?")
    print("-" * 50)
    print("Trigram model best captures local context because it considers")
    print("the two previous words. However, it suffers from data sparsity -")
    print("many specific 3-word sequences don't appear in the training data.")
    print("\nBigram offers a good balance between context and coverage.")
    
    print("\n3. TRADE-OFFS:")
    print("-" * 50)
    print("• Unigram: High coverage, no context awareness")
    print("• Bigram: Good coverage, some context awareness")
    print("• Trigram: Better context, but suffers from sparse data")
    print("• Higher n-grams: Even more context-specific, but even sparser")

def main():
    """Main function to run the Shannon Game analysis"""
    
    # Define the contexts from the problem
    contexts = [
        "it is not my fault",
        "he was such",
        "i was discouraged",
        "the little prince never",
        "but seeds are invisible",
        "he was white with rage",
        "he had",
        "your cigarette has gone out",
        "let us look for",
        "it was"
    ]
    
    # Example from the problem for comparison
    example_context = "once when i was six"
    
    # Train models
    unigram, bigram, trigram = train_models()
    
    # Run Shannon Game
    print("\nRunning Shannon Game for given contexts...")
    results = run_shannon_game(contexts, unigram, bigram, trigram)
    
    # Print results table
    print_results_table(results)
    
    # Analyze model performance
    analyze_models(results, contexts[:3])  # Analyze first 3 examples
    
    # Show the example from the problem
    print("\n" + "="*100)
    print("EXAMPLE FROM PROBLEM: 'once when i was six'")
    print("="*100)
    
    example_words = preprocess_text(example_context)
    example_uni = unigram.predict_next_word(example_words)
    example_bi = bigram.predict_next_word(example_words)
    example_tri = trigram.predict_next_word(example_words)
    
    print(f"\nContext: {example_context}")
    print(f"Unigram: {[word for word, _ in example_uni[:5]]}")
    print(f"Bigram:  {[word for word, _ in example_bi[:5]]}")
    print(f"Trigram: {[word for word, _ in example_tri[:5]]}")
    
    print("\nNote: Your exact results may vary slightly due to:")
    print("1. Different Brown Corpus preprocessing")
    print("2. Case handling (our code lowercases everything)")
    print("3. Random variations in tie-breaking")

if __name__ == "__main__":
    main()
