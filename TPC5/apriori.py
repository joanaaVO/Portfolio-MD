
class TransactionDataset:
    def __init__(self, transactions):
        """
        A class to represent a transaction dataset.

        Parameters:
            transactions (list): A list of transactions where each transaction is a list of items.

        Attributes:
            transactions (list): A list of transactions where each transaction is a list of items.
            frequent_items (dict): A dictionary where each key is an item and each value is its support in the dataset.

        """
        self.transactions = transactions
        self.frequent_items = {}
        self.build_frequent_items()

    def build_frequent_items(self):
        """
        A method to build frequent items for the transaction dataset.

        """
        item_counts = {}
        for transaction in self.transactions:
            for item in transaction:
                if item in item_counts:
                    item_counts[item] += 1
                else:
                    item_counts[item] = 1
        for item, count in item_counts.items():
            self.frequent_items[item] = count / len(self.transactions)
    
    def __iter__(self):
        """
        An iterator method to iterate over the transactions.

        """
        return iter(self.transactions)

    def __len__(self):
        """
        A method to get the number of transactions.

        """
        return len(self.transactions)

class Apriori:
    def __init__(self, transaction_dataset, min_support, min_confidence):
        """
        A class to represent the Apriori algorithm.

        Parameters:
            transaction_dataset (TransactionDataset): An instance of TransactionDataset class representing the transaction dataset.
            min_support (float): The minimum support threshold for frequent itemsets.
            min_confidence (float): The minimum confidence threshold for association rules.

        Attributes:
            transaction_dataset (TransactionDataset): An instance of TransactionDataset class representing the transaction dataset.
            min_support (float): The minimum support threshold for frequent itemsets.
            min_confidence (float): The minimum confidence threshold for association rules.
            itemsets (dict): A dictionary where each key is a frequent itemset and each value is its support in the dataset.
            rules (dict): A dictionary where each key is an association rule and each value is its confidence.

        """
        self.transaction_dataset = transaction_dataset
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.itemsets = {}
        self.rules = {}

    def fit(self):
        """
        A method to fit the Apriori algorithm to the transaction dataset.

        """

        # Step 1: Get all unique items
        items = set()
        for transaction in self.transaction_dataset:
            for item in transaction:
                items.add(item)

        # Step 2: Get frequent itemsets of length 1
        frequent_itemsets = {}
        for item in items:
            count = sum([1 for transaction in self.transaction_dataset if item in transaction])
            support = count / len(self.transaction_dataset)
            if support >= self.min_support:
                frequent_itemsets[(item,)] = support
        self.itemsets = frequent_itemsets

        # Step 3: Generate candidate itemsets
        k = 2
        candidate_itemsets = list(frequent_itemsets.keys())
        while candidate_itemsets:
            itemset_counts = {}
            for transaction in self.transaction_dataset:
                for itemset in candidate_itemsets:
                    if set(itemset).issubset(set(transaction)):
                        if itemset in itemset_counts:
                            itemset_counts[itemset] += 1
                        else:
                            itemset_counts[itemset] = 1
            frequent_itemsets = {}
            for itemset, count in itemset_counts.items():
                support = count / len(self.transaction_dataset)
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
        """
        Generates a list of candidate itemsets of length k based on a list of frequent itemsets.

        Args:
            itemsets (list): A list of frequent itemsets.
            k (int): The length of the candidate itemsets to generate.

        Returns:
            set: A set of candidate itemsets of length k.

        Raises:
            None
        """
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

    transaction_dataset = TransactionDataset(transactions)

    apriori = Apriori(transaction_dataset, min_support, min_confidence)
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
