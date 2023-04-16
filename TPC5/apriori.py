
class Apriori:
    def __init__(self, transactions, min_support, min_confidence):
        self.transactions = transactions
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.itemsets = {}
        self.rules = {}

    def fit(self):
        # Step 1: Get all unique items
        items = set()
        for transaction in self.transactions:
            for item in transaction:
                items.add(item)
        
        # Step 2: Get frequent itemsets of length 1
        frequent_itemsets = {}
        for item in items:
            count = sum([1 for transaction in self.transactions if item in transaction])
            support = count / len(self.transactions)
            if support >= self.min_support:
                frequent_itemsets[(item,)] = support
        self.itemsets = frequent_itemsets
        
        # Step 3: Generate candidate itemsets
        k = 2
        candidate_itemsets = list(frequent_itemsets.keys())
        while candidate_itemsets:
            itemset_counts = {}
            for transaction in self.transactions:
                for itemset in candidate_itemsets:
                    if set(itemset).issubset(set(transaction)):
                        if itemset in itemset_counts:
                            itemset_counts[itemset] += 1
                        else:
                            itemset_counts[itemset] = 1
            frequent_itemsets = {}
            for itemset, count in itemset_counts.items():
                support = count / len(self.transactions)
                if support >= self.min_support:
                    frequent_itemsets[itemset] = support
            candidate_itemsets = self.generate_candidates(list(frequent_itemsets.keys()), k)
            k += 1
            self.itemsets.update(frequent_itemsets)

        # Step 4: Generate association rules
        rules = {}
        for itemset in self.itemsets.keys():
            if len(itemset) > 1:
                for item in itemset:
                    antecedent = tuple(set(itemset) - set([item]))
                    consequent = (item,)
                    if antecedent in self.itemsets:
                        confidence = self.itemsets[itemset] / self.itemsets[antecedent]
                        if confidence >= self.min_confidence:
                            rules[(antecedent, consequent)] = confidence
        # Add generated rules to the dictionary
        self.rules = rules

    def generate_candidates(self, itemsets, k):
        candidates = []
        n = len(itemsets)
        for i in range(n):
            for j in range(i+1, n):
                itemset1 = set(itemsets[i])
                itemset2 = set(itemsets[j])
                if len(itemset1.union(itemset2)) == k:
                    candidate = tuple(sorted(itemset1.union(itemset2)))
                    candidates.append(candidate)           
        return set(candidates)

if __name__ == '__main__':

    transactions = [    
    ['milk', 'bread', 'eggs'],
    ['bread', 'sugar', 'coffee'],
    ['bread', 'milk', 'sugar', 'coffee'],
    ['bread', 'sugar', 'eggs'],
    ['milk', 'sugar', 'coffee'],
    ['milk', 'bread', 'sugar', 'coffee'],
    ['milk', 'bread', 'sugar', 'eggs'],
    ['milk', 'bread', 'sugar', 'coffee', 'eggs'],
    ['bread', 'sugar', 'eggs', 'coffee'],
    ['milk', 'sugar', 'eggs']
    ]

    min_support = 0.4
    min_confidence = 0.5

    apriori = Apriori(transactions, min_support, min_confidence)
    apriori.fit()

    
    # Print frequent itemsets
    print("Frequent itemsets:")
    for itemset, support in apriori.itemsets.items():
        print(f"{itemset}: {support}")
    
    
    # Print association rules
    print("\nAssociation rules:")
    for rule, confidence in apriori.rules.items():
        premise, conclusion = rule
        print(f"{premise} => {conclusion}: {confidence}")
